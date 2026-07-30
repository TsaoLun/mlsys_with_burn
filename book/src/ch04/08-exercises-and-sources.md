# 练习、延伸阅读与来源

## 小结

中间表示让不同阶段围绕合适粒度工作：Burn OperationIr 描述 Tensor 级
操作和资源状态，CubeCL Scope/KernelDefinition 描述设备 Kernel。自动微分
tape、Fusion IR、CubeCL IR 和设备 graph capture 目标不同。

Burn Fusion 按 stream 注册操作，搜索合法执行块，并选择 fused、unfused
或组合计划。同步和读回形成物化边界。TensorStatus、HandleContainer 与
CubeCL 内存池支撑生命周期和复用；ComputeClient 的 launch 通常异步，
read/sync 才是等待边界。

CubeCL Compiler 按目标执行优化和 lowering，再 JIT 编译并缓存。不同后端
不共享完全相同的优化管线，固定快照也不能被描述为完整统一 AOT 工具链。

## 练习

### 概念题

1. 为什么 autodiff tape 和 Fusion OperationIr 不能合并成一个概念？
2. 比较线性 IR、图 IR 和混合 IR 适合的优化粒度。
3. 常量传播为什么常与 DCE 组合？Pass 顺序如何影响结果？
4. 为什么融合能减少访存，却不一定总能提高性能？
5. `TensorStatus::ReadWrite` 为什么只是原地复用的必要条件之一？
6. 区分编译缓存、autotune cache 与设备 pipeline cache。
7. 为什么只测 host launch 调用不能得到设备执行时间？
8. 设备 graph capture 与 Burn Fusion 分别复用什么？

### Rust 与实验题

1. 将实验扩为 add→mul→exp，断言连续路径包含三操作 ElementWise block。
2. 分别在 add 后、mul 后插入同步，比较报告切分。
3. 增加一个 broadcast 输入，比较输出与 Fusion 计划。
4. 使用 `Device::flex()` 计算相同结果作为数值 reference，并解释 Inspector
   为什么没有对应报告。
5. 让两个测试使用显式不同 StreamId，验证报告互不污染。
6. 把 `FusionSummary` 序列化为稳定的教材快照；不要序列化完整 Debug 文本。

### 源码题

1. 找到 float add 构造 `BinaryOpIr` 和注册 OperationIr 的位置。
2. 比较 `TensorStatus::ReadOnly` 与 `ReadWrite` 在 HandleContainer 中的
   handle 获取行为。
3. 沿 `Device::sync()` 找到 Fusion stream drain。
4. 找出 burn-cubecl 注册的五类 fuser，并选择一类解释关闭条件。
5. 找到 CubeCL `KernelDefinition` 的字段及 KernelBuilder 构造路径。
6. 比较 SPIR-V Compiler 与 CPP Compiler 的优化入口。
7. 找到 CubeCL 编译缓存与 autotune cache，比较 key 和 value。

### 性能与系统题

1. 对较大 Tensor 分别测首次与稳态 add→exp；记录同步位置和缓存状态。
2. 比较连续表达式与人为同步版本，但先证明两个计划结构和数值一致。
3. 设计一个生命周期条带图，手工给出可复用 allocation 的贪心方案。
4. 解释多 stream 并行为什么可能增加内存峰值。

## 延伸阅读

固定上游中的权威入口：

- `burn/crates/burn-ir/src/`
- `burn/crates/burn-fusion/src/ops/`
- `burn/crates/burn-fusion/src/stream/`
- `burn/crates/burn-fusion/src/search/`
- `burn/crates/burn-fusion/src/inspect.rs`
- `burn/crates/burn-backend-tests/tests/fusion/fusion_shape.rs`
- `burn/crates/burn-cubecl/src/fusion.rs`
- `burn/crates/burn-cubecl-fusion/src/optim/`
- `burn/crates/burn-flex/ARCHITECTURE.md`
- `cubecl/crates/cubecl-ir/`
- `cubecl/crates/cubecl-opt/`
- `cubecl/crates/cubecl-runtime/src/kernel.rs`
- `cubecl/crates/cubecl-runtime/src/client.rs`
- `cubecl/crates/cubecl-runtime/src/memory_management/`

LLVM、MLIR、Halide、TVM/Ansor 和自动微分文献可用于比较 IR、schedule 与
搜索设计。在线文档必须记录版本，不能覆盖本书固定源码事实。

## 来源与改编说明

本章重组 OpenMLSys v1 两章内容：

### `chapter_frontend_and_ir/`

- `index.md`
- `overview_of_frontend.md`
- `ai_compiler_design_principle.md`
- `intermediate_representation.md`
- `ad.md`
- `type_system_and_static_analysis.md`
- `common_frontend_optimization_pass.md`
- `summary.md`

保留 IR 分类、多层编译、静态信息、经典 Pass 与 AD/IR 关系。TensorFlow、
JAX、MindIR 与 MindSpore 自动微分实现只作为原始范围核对，没有迁移其长
代码和框架结构图。

### `chapter_backend_and_runtime/`

- `index.md`
- `overview.md`
- `graph_optimizer.md`
- `kernel_selecter.md`
- `memory_allocator.md`
- `compute_schedule_and_execute.md`
- `op_compiler.md`
- `summary.md`

保留融合、layout/dtype、Kernel 选择、生命周期、内存池、异步调度和
compute/schedule 思想。MindSpore Graph Kernel、SOMAS、Ascend task 下沉
及厂商布局约束未映射为 Burn 能力；通信 buffer 内容后移第 6 章。

OpenMLSys v2 固定快照只有第 4 章 TODO，没有可迁移正文。本章没有复制
OpenMLSys ch04/ch05 图片或 Python/C++ 示例。新增 Burn OperationIr/Fusion、
CubeCL lowering/JIT/stream 内容。Rust 实验参考固定 Burn
`fusion_shape.rs` 的 add→exp 与同步切分回归模式，重新设计了独立 Stream、
可传播错误、稳定 summary、教学输出和双重结构/数值断言。

完整逐文件和固定源码映射见 `planning/chapter-sources/ch04.md`。
OpenMLSys 原作和改编正文采用 CC BY-NC-SA 4.0，原创 Rust 示例采用
MIT OR Apache-2.0。

