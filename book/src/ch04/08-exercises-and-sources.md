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
不共享完全相同的优化管线，本版也不能被描述为完整统一 AOT 工具链。

## 练习


练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。

### 概念题

1. 【基础】为什么 autodiff tape 和 Fusion OperationIr 不能合并成一个概念？

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

2. 【基础】比较线性 IR、图 IR 和混合 IR 适合的优化粒度。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【基础】常量传播为什么常与 DCE 组合？Pass 顺序如何影响结果？

<details>
<summary>提示</summary>

见第 4 章 Pass 契约与同步边界节。

</details>

4. 【基础】为什么融合能减少访存，却不一定总能提高性能？

<details>
<summary>提示</summary>

运行/阅读 `examples/ch04-fusion-inspector` 与第 4 章 Fusion 节。

</details>

5. 【进阶】`TensorStatus::ReadWrite` 为什么只是原地复用的必要条件之一？

<details>
<summary>提示</summary>

对照第 4 章「内存、Stream 与异步执行」中的生命周期、别名和 stream 依赖；
`ReadWrite` 只说明访问模式，不说明没有其他 handle 仍可能读它。

</details>

6. 【进阶】区分编译缓存、autotune cache 与设备 pipeline cache。

<details>
<summary>提示</summary>

对照第 4 章「CubeCL Lowering、JIT 与缓存」：编译缓存避免重新 lowering/
codegen，autotune cache 保存候选测量结果；不要把二者与硬件 pipeline 的
执行状态混为一谈。

</details>

7. 【进阶】为什么只测 host launch 调用不能得到设备执行时间？

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

8. 【进阶】设备 graph capture 与 Burn Fusion 分别复用什么？

<details>
<summary>提示</summary>

见第 4 章 Pass 契约与同步边界节。

</details>

9. 【进阶】为常量传播、DCE、CSE 和融合各写一条输入/输出不变量，并列出一个
   必须回退的副作用或别名场景。

<details>
<summary>提示</summary>

运行/阅读 `examples/ch04-fusion-inspector` 与第 4 章 Fusion 节。

</details>

10. 【进阶】沿一次 shape 改变追踪 Fusion 计划、tune key、编译 key、cache、
    launch 和 readback 哪些环节会失效或重新发生。

<details>
<summary>提示</summary>

见第 4 章 Pass 契约与同步边界节。

</details>


### Rust 与实验题

1. 【基础】运行已交付的 `inspect_add_mul_exp`，确认三操作 ElementWise block；再
   分别在 add 后、mul 后插入同步，比较报告切分。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【基础】增加一个 broadcast 输入，比较输出与 Fusion 计划。

<details>
<summary>提示</summary>

见第 4 章 Pass 契约与同步边界节。

</details>

3. 【进阶】使用 `Device::flex()` 计算相同结果作为数值 reference，并解释 Inspector
   为什么没有对应报告。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

4. 【进阶】让两个测试使用显式不同 StreamId，验证报告互不污染。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】把 `FusionSummary` 序列化为稳定的教材快照；不要序列化完整 Debug 文本。

<details>
<summary>提示</summary>

见第 4 章 Pass 契约与同步边界节。

</details>


### 源码题

1. 【进阶】找到 float add 构造 `BinaryOpIr` 和注册 OperationIr 的位置。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

2. 【进阶】比较 `TensorStatus::ReadOnly` 与 `ReadWrite` 在 HandleContainer 中的
   handle 获取行为。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

3. 【进阶】沿 `Device::sync()` 找到 Fusion stream drain。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

4. 【进阶】找出 burn-cubecl 注册的五类 fuser，并选择一类解释关闭条件。

<details>
<summary>提示</summary>

见第 3 章 GPU 并行层次节与配图。

</details>

5. 【进阶】找到 CubeCL `KernelDefinition` 的字段及 KernelBuilder 构造路径。

<details>
<summary>提示</summary>

见第 3 章 GPU 并行层次节与配图。

</details>

6. 【进阶】比较 SPIR-V Compiler 与 CPP Compiler 的优化入口。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

7. 【进阶】找到 CubeCL 编译缓存与 autotune cache，比较 key 和 value。

<details>
<summary>提示</summary>

见第 3 章 GPU 并行层次节与配图。

</details>


### 性能与系统题

1. 【进阶】对较大 Tensor 分别测首次与稳态 add→exp；记录同步位置和缓存状态。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【挑战】比较连续表达式与人为同步版本，但先证明两个计划结构和数值一致。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【挑战】设计一个生命周期条带图，手工给出可复用 allocation 的贪心方案。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【挑战】解释多 stream 并行为什么可能增加内存峰值。

<details>
<summary>提示</summary>

回看第 4 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


## 延伸阅读

本书固定版本源码中的权威入口：

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

## 本章系统结论

1. 编译器做保持语义的变换；运行时把计划落实到分配、调度、launch 与同步。
2. 前端关注 OperationIr / Pass / autodiff 边界；后端关注选择、内存、stream 与 JIT/cache。
3. 同一套 Fusion/CubeCL IR 可以落到不同 Runtime；同步/`read` 语义在设备上更昂贵。
4. CPU 上 FusionInspector 让你看到计划切分与数值等价，不是硬件 launch 计数器。
5. GPU 阅读时对照：设备 graph capture（若后端支持）、cache/JIT 与真实 launch 指标的区别。
6. Fusion block 数、cache hit、kernel launch 与墙钟时间不是同一个量。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-4-章)。
