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

两者的生命周期差异列在
[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
第 4 节；tape 的构造背景见第 2 章[「自动微分」](../ch02/05-autodiff.md)。
从“各自被谁消费、何时失效”入手：`backward()` 读 tape 生成梯度，
`Device::sync()` 只 drain Fusion stream，两个动作互不蕴含。

</details>

2. 【基础】比较线性 IR、图 IR 和混合 IR 适合的优化粒度。

<details>
<summary>提示</summary>

[「编译栈与中间表示」](01-stack-and-ir.md)第 2 节写了三种形态各自
擅长的分析，第 3 节的分层表格是现成对照素材。想一想：Tensor 级
子图搜索为什么落在 OperationIr 这类图表示，而 unit 级指令优化要
转成 CFG/SSA 风格做局部数据流分析；答案取决于各优化需要的信息。

</details>

3. 【基础】常量传播为什么常与 DCE 组合？Pass 顺序如何影响结果？

<details>
<summary>提示</summary>

把[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
第 2 节那段中性伪 IR 亲手推两遍：先折叠/传播再 DCE，与颠倒顺序，
各自还能删掉哪些值？关键在于常量传播会制造新的“不可观察”事实，
顺序决定这些事实在 DCE 运行前是否已经暴露。

</details>

4. 【基础】为什么融合能减少访存，却不一定总能提高性能？

<details>
<summary>提示</summary>

[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 5 节先算出
融合把全局内存流量从 20N 字节降到 12N 字节，又在结尾列出反向代价
（寄存器压力、编译时间、过大 Kernel）。想清楚“省流量约等于省时间”
依赖低算术强度这一前提，再构造一个前提不成立的场景。

</details>

5. 【进阶】`TensorStatus::ReadWrite` 为什么只是原地复用的必要条件之一？

<details>
<summary>提示</summary>

[「内存、Stream 与异步执行」](06-memory-streams-execution.md)第 2 节
列出了与 `ReadWrite` 并列的其他条件：shape、dtype、别名、Kernel
语义与是否仍有其他读者。逐项构造一个“状态是 ReadWrite 却不能原地
覆盖”的反例，就能说明它为什么只是必要条件之一。

</details>

6. 【进阶】区分编译缓存、autotune cache 与设备 pipeline cache。

<details>
<summary>提示</summary>

[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 5 节列出四类缓存。按三个维度比较：key 编码什么，value 是编译
产物、候选选择还是已加载对象，失效条件是什么；该节还提醒“调优
命中不保证编译产物已加载”，这句话适合用来检验你的答案。

</details>

7. 【进阶】为什么只测 host launch 调用不能得到设备执行时间？

<details>
<summary>提示</summary>

[「内存、Stream 与异步执行」](06-memory-streams-execution.md)第 4 节：
launch 是向 stream 提交，host 返回不代表设备完成，错误还可能拖到
read/sync 才报告。再对照
[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 4.1 节因果链的最后一步，说明哪一刻起耗时才算已观察事实。

</details>

8. 【进阶】设备 graph capture 与 Burn Fusion 分别复用什么？

<details>
<summary>提示</summary>

判据在[「内存、Stream 与异步执行」](06-memory-streams-execution.md)
第 6 节末段的对比句。回答时分三层核对：各自的输入对象是什么、
复用后省掉的是哪部分工作、失效条件有何不同；再解释同文第 3 节里
capture 窗口为什么要求 persistent pool 一类分配约束。

</details>

9. 【进阶】为常量传播、DCE、CSE 和融合各写一条输入/输出不变量，并列出一个
   必须回退的副作用或别名场景。

<details>
<summary>提示</summary>

套用[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
第 2.1 节的四元组模板：输入不变量、分析条件、输出不变量、不能变换
时的处理。回退场景可从该文第 5 节的随机数、I/O、原地更新中挑；
融合一条对照第 2.1 节末尾的 fuser 条件写，别只写泛泛的“不安全”。

</details>

10. 【进阶】沿一次 shape 改变追踪 Fusion 计划、tune key、编译 key、cache、
    launch 和 readback 哪些环节会失效或重新发生。

<details>
<summary>提示</summary>

沿[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 4.1 节的因果链逐箭头标注“必然重来/可能重来/不受影响”。shape
何时只是运行时元数据、何时进入编译键，参考
[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
第 3 节；记住 cache 命中只表示某一层结果可复用，不代表下游全部跳过。

</details>


### Rust 与实验题

1. 【基础】运行已交付的 `inspect_add_mul_exp`，确认三操作 ElementWise block；再
   分别在 add 后、mul 后插入同步，比较报告切分。

<details>
<summary>提示</summary>

`examples/ch04-fusion-inspector` 的 `inspect_add_exp` 已示范切分写法：
`split_by_sync` 分支里的 `device.sync()` 就是可移动的边界。把同样的
调用分别插到 `inspect_add_mul_exp` 的 add 之后、mul 之后，比较
`reports` 数与各块 `operations` 计数怎样变化；预期形态参考
[「实验：观察 Fusion 执行计划」](07-fusion-inspector-lab.md)第 7 节。

</details>

2. 【基础】增加一个 broadcast 输入，比较输出与 Fusion 计划。

<details>
<summary>提示</summary>

把 `inspect_add_exp` 某个输入改成可广播的 shape（例如 `[1, 4]`），
先手算预期数值再运行。计划方面不要预设结论：
[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 4 节说明
broadcast 是 fuser 可以拒绝候选的条件之一，接受与否都用
`BlockSummary` 的 fuser 名称和 operations 数如实记录。

</details>

3. 【进阶】使用 `Device::flex()` 计算相同结果作为数值 reference，并解释 Inspector
   为什么没有对应报告。

<details>
<summary>提示</summary>

背景在[「实验：观察 Fusion 执行计划」](07-fusion-inspector-lab.md)
第 5 节与[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 7 节。
对照点放在 `to_data()` 读回的数值上；解释报告缺失时，想清楚
Inspector 安装在哪条 Fusion stream 上、Flex 的 eager 路径有没有向
它注册过任何 OperationIr。

</details>

4. 【进阶】让两个测试使用显式不同 StreamId，验证报告互不污染。

<details>
<summary>提示</summary>

先看 `examples/ch04-fusion-inspector` 中 `StreamId::allocate()`、
`stream.executes(...)` 与 `FusionInspector::install(stream)` 的配合：
Inspector 按 stream 安装。给两个测试各自分配 stream、各装各的
Inspector，断言 `drain()` 结果互不包含对方的操作；stream 隔离的
背景见[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 2 节。

</details>

5. 【进阶】把 `FusionSummary` 序列化为稳定的教材快照；不要序列化完整 Debug 文本。

<details>
<summary>提示</summary>

`FusionSummary` 本身就是示范：只保留 fuser 名称、operations 计数等
稳定字段，而不是内部结构的 Debug 文本。沿同样思路做字段级序列化，
并说明理由——[「实验：观察 Fusion 执行计划」](07-fusion-inspector-lab.md)
第 2、6 节解释了 test-util 不是长期稳定接口、完整日志文本会随版本
漂移，不适合当作快照比对对象。

</details>

6. 【进阶】给 `ch04-mini-pass-pipeline` 增加恒等元素消除（`x+0 → x`、
   `x*1 → x`），并回答：对 `f32` 的 `-0.0`，`x+0 → x` 按位合法吗？

<details>
<summary>提示</summary>

先写「语义按位一致」测试再写 Pass（照抄现有 Pass 的测试骨架）。
`(-0.0) + 0.0` 在 IEEE 754 里等于 `+0.0`，用 `f32::to_bits` 而不是
`==` 检查这条改写在 `-0.0` 上的行为；对照
[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
「亲手写一个 Pass」段的合法性讨论决定它属于标准还是 fast-math 集。

</details>

7. 【挑战】让 CSE 识别交换律：`add %0 %1` 与 `add %1 %0` 合并为一个
   节点，并说明为什么 `Exp` 这类单输入算子不需要这一步。

<details>
<summary>提示</summary>

在 `common_subexpression_elimination` 的结构化 `Key` 上做操作数
归一化（如小编号在前）即可，`Mul` 同理；浮点加法交换律逐位成立
（结合律才不成立），依据见
[「静态信息、Pass 与自动微分边界」](02-static-analysis-and-passes.md)
的浮点语义段。改完后跑随机图语义测试确认按位一致。

</details>


### 源码题

1. 【进阶】找到 float add 构造 `BinaryOpIr` 和注册 OperationIr 的位置。

<details>
<summary>提示</summary>

在本书所用版本源码 `burn/crates/burn-fusion/src/ops/` 下搜索
`float_add`，看它构造 `BinaryOpIr` 后把描述交给谁；对照
[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 1 节的注册
链路，把“Tensor 操作到 client”的每一跳落到具体函数上。

</details>

2. 【进阶】比较 `TensorStatus::ReadOnly` 与 `ReadWrite` 在 HandleContainer 中的
   handle 获取行为。

<details>
<summary>提示</summary>

`HandleContainer` 定义在本章列出的 `burn/crates/burn-ir/src/` 里，
找到按 `TensorStatus` 分支取 handle 的方法，比较两种状态下 handle
的所有权去向与容器内条目的变化；再回到
[「内存、Stream 与异步执行」](06-memory-streams-execution.md)第 2 节，
核对“最后使用者可取得所有权”对应哪几行代码。

</details>

3. 【进阶】沿 `Device::sync()` 找到 Fusion stream drain。

<details>
<summary>提示</summary>

从本章列出的 `burn/crates/burn-fusion/src/stream/` 入手，在多 stream
管理模块里搜索 `drain`，记录哪些调用方会触发它；再对照
[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 6 节列出的
物化边界，确认同步、读回与跨 stream 共享各自走到哪条路径。

</details>

4. 【进阶】找出 burn-cubecl 注册的五类 fuser，并选择一类解释关闭条件。

<details>
<summary>提示</summary>

注册点在本章列出的 `burn/crates/burn-cubecl/src/fusion.rs`（搜索
`fusers`），五个实现体在 `burn/crates/burn-cubecl-fusion/src/optim/`
的子模块里。选一类后带着问题读：什么样的 shape、broadcast、layout
或 dtype 会让它拒绝候选？检索关键词可用
[「Burn IR 与运行时融合」](03-burn-ir-and-fusion.md)第 4 节的条件清单。

</details>

5. 【进阶】找到 CubeCL `KernelDefinition` 的字段及 KernelBuilder 构造路径。

<details>
<summary>提示</summary>

`KernelDefinition` 的字段就在本章列出的
`cubecl/crates/cubecl-runtime/src/kernel.rs`。先抄下字段清单，再反查
`KernelBuilder` 从哪里收集它们，与
[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 1 节列出的输入（Scope、参数、CubeDim、设置）逐项对上。

</details>

6. 【进阶】比较 SPIR-V Compiler 与 CPP Compiler 的优化入口。

<details>
<summary>提示</summary>

以[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 2 节为地图：两条路径都会用到本章列出的
`cubecl/crates/cubecl-opt/`，但接入位置与额外步骤不同。分别在两个
Compiler 的源码里搜索对 Optimizer 的调用，再确认 CPP 侧多出的
shared-memory 分析与 Scope post-processing 挂在哪一步。

</details>

7. 【进阶】找到 CubeCL 编译缓存与 autotune cache，比较 key 和 value。

<details>
<summary>提示</summary>

分类先看[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 5 节。源码从 `cubecl/crates/cubecl-runtime/src/` 入手，分别搜索
编译产物的存取与 `tune` 相关模块，比较两边的 key 各编码了什么、
value 是可执行产物还是“哪个候选胜出”的记录，失效条件有何不同。

</details>


### 性能与系统题

1. 【进阶】对较大 Tensor 分别测首次与稳态 add→exp；记录同步位置和缓存状态。

<details>
<summary>提示</summary>

首次成本的构成见
[「CubeCL Lowering、JIT 与缓存」](05-cubecl-lowering-and-jit.md)
第 4 节；计时终点必须落在 read/sync 之后，理由见
[「内存、Stream 与异步执行」](06-memory-streams-execution.md)第 4 节。
报告写明是第几次运行、同步放在哪一行、缓存冷热，并把结果当作
环境相关测量，不外推为普适结论。

</details>

2. 【挑战】比较连续表达式与人为同步版本，但先证明两个计划结构和数值一致。

<details>
<summary>提示</summary>

结构与数值证据可直接扩展 `examples/ch04-fusion-inspector` 的
`observes_fusion_and_sync_boundary` 测试：用 `blocks` 固定两版计划、
断言输出逐位相等。之后的计时才是新增部分，按
[「实验：观察 Fusion 执行计划」](07-fusion-inspector-lab.md)第 1 节
列出的污染源（首次 JIT、Fusion 搜索、调度）设计预热与重复，并把
计划结构差异与墙钟差异分开报告，不要用一方去证明另一方。

</details>

3. 【挑战】设计一个生命周期条带图，手工给出可复用 allocation 的贪心方案。

<details>
<summary>提示</summary>

模板就是[「内存、Stream 与异步执行」](06-memory-streams-execution.md)
第 2 节的两张条带图：横轴时间、纵轴 allocation，同步会拉长中间值
寿命。贪心可按创建顺序扫描，把新 Tensor 放进“最后读取已结束”的
旧槽；再用该节的 shape、dtype、别名条件说明哪些复用必须放弃。

</details>

4. 【挑战】解释多 stream 并行为什么可能增加内存峰值。

<details>
<summary>提示</summary>

[「内存、Stream 与异步执行」](06-memory-streams-execution.md)第 5 节
末段点了题：并发块的生命周期重叠。给同一组操作画“单 stream 串行”
与“双 stream 并行”两张条带图，数每个时刻同时存活的 allocation；
再联系第 2 节的复用条件，解释重叠为什么取消了原本可行的复用。

</details>


## 延伸阅读

TVM、MLIR、Halide 等编译系统论文见附录
[参考文献](../references.md#第-4-章-ai-编译器与运行时系统)。
本书所用版本源码中的权威入口：

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
2. 前端关注 OperationIr / Pass / autodiff 边界；后端关注选择、内存、stream 与 JIT。
3. 同一套 Fusion/CubeCL IR 可以落到不同 Runtime；同步在设备上更昂贵。
4. FusionInspector 让你看到计划切分；迷你 Pass 让你亲手写出非法 fast-math 的后果。
5. 改融合规则打开 `burn-fusion`，改 JIT 缓存键打开 CubeCL Compiler。
6. Fusion block 数、cache hit 和墙钟时间不是同一个量。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-4-章)。
