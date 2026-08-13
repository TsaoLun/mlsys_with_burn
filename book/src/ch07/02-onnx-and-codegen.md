# ONNX、图转换与 Burn Rust 代码生成

## ONNX 解决什么问题

训练框架的模型对象通常带有自己的 module 类型、参数命名和执行语义。
推理系统如果为每个训练框架实现一个专用 runtime，维护成本会随框架和
backend 数量增长。ONNX（Open Neural Network Exchange）提供一种交换
图表示，但它不是“所有模型天然兼容”的魔法格式。

转换至少要回答两类问题：

- **拓扑问题**：输入、输出、节点顺序、子图、动态 shape 和控制流怎样
  表达；
- **算子语义问题**：属性、broadcast、layout、dtype、边界条件和版本
  语义怎样映射。

同名算子也不一定同义。例如一个框架的 `Slice` 可能与 ONNX 的
`Split`/`Slice` 组合相对应；训练框架中的融合节点可能需要拆开或重写。
如果两边的算子集合或语义没有交集，转换必须报错或由人工提供替代实现。

## `ModelGen` 的转换路线

`burn-onnx` 的 `ModelGen` 把转换分成几个有明确边界的阶段：

```text
ONNX file
   │ OnnxGraphBuilder::parse_file
   ▼
ONNX IR + simplify passes
   │ ParsedOnnxGraph::into_burn
   ▼
BurnGraph + node registry
   ├── Rust token/source generation
   └── tensor snapshots → .bpk Burnpack
```

在 `model_gen.rs` 中，`OnnxGraphBuilder::new().simplify(...)` 负责解析并
选择是否执行图简化；随后每个 ONNX node 注册到 `BurnGraph`，图的 input/
output 也被注册。`BurnGraph::codegen` 最终生成可读的 Rust source，而不是
在运行时解释 ONNX 文件。

这带来一个部署取舍：

- 转换和编译阶段承担更多工作，运行时不必携带完整 ONNX parser；
- 生成的 Rust source 可以进入 Rust 类型检查和普通构建流程；
- 输出代码仍依赖生成时所针对的 Burn API 和 backend 能力；
- 生成成功不等于 reference 数值、动态 shape 或所有目标 backend 都已
  通过验证。

`burn-onnx` 还支持大图 partition。`partition(true)` 时，超过阈值的图可能被
拆成多个 submodule，每个 submodule 有自己的 `forward`；这主要帮助生成
代码保持可编译和可读，不应直接解释成 runtime 的模型并行。

## 图简化与第 4 章的关系

`ModelGen` 的 simplify 选项属于导入前后的图处理入口，README 和
源码注释列出的方向包括常量折叠、公共子表达式消除、死代码消除、恒等元素
消除和部分 reshape/permute 识别。它们的共同条件是保持输入输出语义。

不要把三个“图”混在一起：

1. ONNX graph 是交换 artifact 的表示；
2. `BurnGraph` 是 `burn-onnx` 代码生成时的中间组织；
3. Burn IR/Fusion 是运行或后端优化路径中的表示。

导入阶段删掉一个常量节点，不意味着后端一定会融合所有相邻算子；后端是否
融合仍取决于 Burn feature、backend 和设备运行时。第 4 章讨论的
capture/register、analysis、lowering 和 device sync 仍然适用于部署推理。

## 一个算子的旅程：Gemm

上面的阶段图落到单个算子上是什么样？以 `Gemm`
（$Y=\alpha A'B'+\beta C$）为例，按本书 burn-onnx 版本走一遍：

**第 1 站：注册。** `onnx-ir/src/registry.rs` 把每种 ONNX 节点类型
绑定到一个处理器——算子在词汇表里有名字，解析器才认识它：

```rust,ignore
// onnx-ir/src/registry.rs
registry.register(NodeType::Gemm, Box::new(crate::node::gemm::GemmProcessor));
```

**第 2 站：属性与形状。** `onnx-ir/src/node/gemm.rs` 的
`GemmProcessor` 提取属性并做类型推断：`GemmConfig { alpha, beta,
trans_a, trans_b }`，缺省值按 ONNX 规范取 `alpha = 1.0`、
`beta = 1.0`；`infer_types` 根据输入秩推出输出形状。第 2 节说的
「算子语义问题」在这里变成具体字段。

**第 3 站：模式识别（图级 lowering）。** `onnx-ir` 的
node_conversion 阶段会识别特例：`Gemm(alpha=1, beta=1, transB=1)`
被转换成 Burn 专有的 `Linear` 节点（`MatMul + Add` 序列也会融合成
它）。`onnx-ir/src/node/linear.rs` 的注释同时记下了一个精细的语义
问题：Gemm 来源的权重是 `[out_features, in_features]` 布局，需要
`transpose_weight` 标志，MatMul 来源的不需要——**同名参数在不同
来源下布局不同**，这正是转换器必须携带元数据的原因。

**第 4 站：代码生成。** 没被特例吸收的一般 Gemm 走
`burn-onnx/src/burn/node/gemm.rs` 的 `NodeCodegen::forward`，把
配置拼成 Burn 张量 API 的 Rust token：

```rust,ignore
// burn-onnx/src/burn/node/gemm.rs（节选）
let product = quote! { #a.matmul(#b) };            // 必要时先 .transpose()
// alpha != 1 时： quote! { #product * #alpha }
// 有 C 时：      quote! { … + (#c) * #beta }
```

生成的不是对 ONNX 的解释执行，而是普通 Rust 源码——这就是第 2 节
「转换期做重活、运行时只剩 Burn API 调用」的含义。被识别成
`Linear` 的那一支则生成 `nn::Linear` 模块，权重作为 snapshot 进入
`.bpk`（下一节的装载入口）。

**第 5 站：语义测试。** `onnx-tests/tests/gemm/` 里有一组
`gemm*.py` 脚本生成的 `.onnx` fixture（含 `gemm_no_c`、
`gemm_non_unit_alpha_beta` 等变体），Rust 测试把生成模型的输出与
参考值比对——属性组合的每个分支都有 fixture 盯着。

把 `Gemm` 换成任何算子，五站不变：注册、属性与形状推断、可选的
模式识别、代码生成、fixture 测试。`SUPPORTED-ONNX-OPS.md` 列出的
每个「已支持」算子背后都是这样一条链；表里的空档则意味着五站中
至少缺一站。

## 生成代码如何加载权重

`BurnGraph::register_burnpack_loaders` 为生成的 `Model` 安排不同入口。
`LoadStrategy` 不是“性能等级”，而是权重生命周期和宿主环境选择：

- `File`：权重留在独立 `.bpk` 文件；生成 `from_file`、`from_bytes` 和
  `Default`。`Default` 使用生成时的文件路径，适合标准 host 文件系统；
- `Embedded`：通过对齐后的 `include_bytes!` 把 `.bpk` 放进 binary，
  生成 `from_embedded`、`from_bytes` 和 `Default`。这是减少运行时文件
  依赖的一种方式，但会把权重纳入 binary；
- `Bytes`：只生成 `from_bytes`，调用者从网络、固件分区或自定义存储取得
  bytes；
- `None`：不生成内置 loader，调用者必须自己初始化和装配参数。

这些入口内部都会创建 model、构造 `BurnpackStore`，再调用 `load_from`。
因此“生成了 Rust model”与“权重已经在设备上”是两个事件：后者发生在
loader 应用 snapshot 时，还可能触发 dtype 转换、设备分配和 backend copy。

## 为什么本书默认示例不直接依赖 `burn-onnx`

本书示例使用的 Burn revision 是 `976aa9...`，而 `burn-onnx` 的
manifest 把 `burn`、`burn-flex` 和 `burn-store` 指向 `78f10a...`。依赖图中
即使出现相同的 package name，Rust 也会把不同 revision 的类型视为不同
类型；例如旧 `burn::Tensor` 不能自动传给当前示例里的 `burn::Tensor`。

所以本章分两条阅读线：

1. 读取 `burn-onnx` 源码，对照 ONNX graph、codegen、Burnpack 和
   `LoadStrategy` 的行为；
2. 用本书示例里的 Burn `ModuleRecord` 做 CPU 往返保存与恢复，观察当前
   参数状态 API。

若将来 `burn-onnx` 与本书示例的 Burn 对齐，再增加小型 ONNX fixture 时，
应同时比较 ONNX Runtime reference、生成 model 的输出和不同 backend 的
输出，而不是只把 crate 塞进同一依赖图。

## 验证转换时建议记下什么

导入一个模型时，建议至少记下：

```text
source model checksum
→ importer revision + Burn revision
→ opset / input-output schema
→ simplification and partition options
→ generated source checksum + weight checksum
→ reference outputs and tolerances
→ target backend/device/dtype
```

这份记录能区分“转换器改变了图”“loader 改变了参数 dtype”“backend
计算有误差”和“前处理 schema 不一致”。没有它，部署故障常被错误归因于
模型文件本身。
