# 2026-08-13 深度批次四：宏黑箱、ONNX 算子旅程、解剖对照

## 背景

延续「机制彻底搞懂、贡献为副产品」的定位（批次三），按 STATUS 记录
的顺序交付三项机制纵深，全部为内容深化，无范围决策需要。

## 交付

1. **宏黑箱打开（真实展开产物）**。本机有 cargo-expand 1.0.121，
   两处小节均引用真实展开而非从宏源码推断：
   - ch02/03「宏在替你写什么」：`derive(Module)` 为 TinyModel 生成
     的逐字段 num_params/visit/map（无运行时反射、
     enter/exit_module 字段名即第 7 章参数路径来源、map 按值重建
     满足所有权）；派生宏集合核实为 Module/Config/RecordState。
   - ch03/03「宏在替你写什么」：`#[cube(launch_unchecked)]` 生成
     expand 函数（ExpandType 签名 + `__expand_*` 链向 Scope 登记
     IR）、Kernel 类型（`define()` 产出 KernelDefinition，JIT 缓存
     键来源）、launch 装配入口；三个时刻（构造/首编/每次提交）；
     展开期控制流与 tape 只记执行路径同构。
2. **ch07/02「一个算子的旅程：Gemm」**：五站走读（注册
   registry.rs:444 → 属性/形状推断 → node_conversion 模式识别
   Gemm(α=1,β=1,transB=1)/MatMul+Add → Linear（权重布局与
   transpose_weight 元数据）→ NodeCodegen quote! 代码生成 →
   onnx-tests fixture 变体）；结尾把 SUPPORTED-ONNX-OPS 解释为
   「每行 = 一条五站链」。
3. **解剖页 add/sum 对照**：op-anatomy「换一个算子」段扩为真实
   摘录（add 的 State=(Shape,Shape) 与 broadcast_shape、sum 的
   ones×grad），点题「反向需要什么，checkpoint 就要负担什么」；
   `ch02-ch04-op-anatomy` 新增两断言（广播梯度形状/按列归约值、
   sum 梯度全 1），共 6 测试。

## 验证

- `cargo test -p ch02-ch04-op-anatomy --locked --offline`：6 通过；
  clippy 零警告；`cargo fmt --all --check` 通过；
- `cargo expand` 产物人工比对（/tmp/expand-ch02.rs 531 行、
  /tmp/expand-gemm.rs 1826 行），书内摘录标注「节选并简化」；
- burn-onnx 五站均以 `git show/grep <pin>` 核实；
- `mdbook build/test book`、`check_release.py --require-built-book
  --json`（`ok=true`）、`git diff --check` 通过。

## 边界

- 展开摘录做了排版简化（省略 lint 属性与部分限定路径），并在正文
  注明；`cargo expand` 作为读者可选工具给出命令，不进入默认检查。
- ONNX 旅程为固定 revision 源码走读，无书内可运行 fixture
  （D010 隔离不变）。

## 后续候选

- 二元/归约解剖已并入本批；剩余机制候选：`matmul` 在 CubeK
  tile/stage/global 的下钻走读（第 3 章已有六决策点走查，可与
  解剖页互链）、autotune 键的端到端追踪、`#[derive(CubeType)]`
  的展开对照。
