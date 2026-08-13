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
