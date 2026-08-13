# 练习、延伸阅读与来源

## 小结

大规模 GPU 集群需要把控制面、训练通信数据面和设备运行时分开。控制面
负责队列、配额、成组准入、拓扑放置、租约、故障恢复和集群级遥测；数据面
负责已分配 rank 之间的 collective；设备运行时负责 stream、memory、
kernel 和同步边界。

OpenMLSys v1 提供了分布式训练的拓扑、通信、并行和参数服务器动机。固定
Burn/CubeCL 快照提供 `ExecutionStrategy`、`DistributedContext`、
`DistributedOps`、CUDA collective 和 `ComputeClient` 等局部入口，但没有
把这些入口组合成集群调度器。CPU 模拟器只验证资源、成本和恢复协议，不
替代真实集群 benchmark。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么“集群有足够 GPU”不等于一个同步作业可以立即启动？请分别从
   显存、成组调度和通信域回答。

<details>
<summary>提示</summary>

把[「作业队列、资源向量与成组调度」](03-job-queue-and-resource-scheduling.md)
的资源向量 `R_job` 和
[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)
区分的三类“容量”逐项过一遍：显存不够、凑不齐成组资源、只能拿到
跨机柜组合，分别卡在准入的哪个条件上、把代价记进哪个时间项。

</details>

2. 【基础】画出控制面、训练数据面、设备运行时三层，并把队列等待、
   `all_reduce`、`ComputeClient::sync` 放到正确层。

<details>
<summary>提示</summary>

对照[「集群负载、系统分层与能力边界」](01-cluster-workload-and-boundary.md)
的三层图和第 6 章
[「集合通信、DDP 与能力边界」](../ch06/06-collective-and-ddp.md)：
队列等待属于控制面，`all_reduce` 属于训练数据面，
`ComputeClient::sync` 属于设备运行时完成边界。

</details>

3. 【基础】用 `alpha + beta * bytes` 比较同机柜、跨机柜和跨 Spine 的消息成本；
   说明这个模型遗漏了哪些真实网络因素。

<details>
<summary>提示</summary>

[「GPU 节点、机柜与网络拓扑」](02-gpu-node-and-network-topology.md)
给出成本函数推导与超额认购、热点的讨论；示例
`examples/ch09-cluster-simulator` 的 `communication_cost` 只把
每个 GPU pair 分进三档通信域再乘确定性 multiplier。对照“超额
认购与热点”一段想一想：哪些现象是一条固定乘数表达不了的。

</details>

4. 【基础】FIFO 和 topology-aware placement 的公平性目标有什么不同？为什么
   “减少跨机柜 bytes”可能增加某个租户的 queue wait？

<details>
<summary>提示</summary>

[「作业队列、资源向量与成组调度」](03-job-queue-and-resource-scheduling.md)
列出了几种互不等价的“公平”定义和队首阻塞场景；再用
`examples/ch09-cluster-simulator` 对比两种 `PlacementPolicy` 的
`queue_wait_us`、`p95_queue_wait_us` 与 `cross_rack_bytes`。思考
方向：把作业挤进同一通信域会改变剩余空闲资源的形状，这对后续
作业的成组准入意味着什么。

</details>

5. 【进阶】构造一个 GPU 总容量足够但发生资源碎片的例子；说明保留大块资源和
   backfilling 的取舍。

<details>
<summary>提示</summary>

[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)
里“四张空闲 GPU 分布在两个机柜”就是一个起点；示例 crate 的
`Cluster::uniform_interleaved` 特意把相邻 id 放进不同机柜，便于
构造这类布局。取舍对照同节“打包、保留与抢占”：保留牺牲哪种
利用率，backfill 又必须守住哪条启动承诺。

</details>

6. 【进阶】为什么一个 `DistributedContext` 不能自动说明节点 membership、rank
   rendezvous 和故障恢复已经完成？

<details>
<summary>提示</summary>

[「拓扑感知放置与集合通信成本」](04-topology-aware-placement-and-communication.md)
的“Burn 的 collective 边界”指出 `DistributedContext` 保存的只是
传入的设备集合。对照本章列出的
`burn/crates/burn-tensor/src/tensor/distributed.rs`，数一数
`init` 的输入里有没有 job、attempt、租约或故障域字段。

</details>

7. 【进阶】设计一个 checkpoint commit 协议，防止旧 attempt 重复提交 optimizer
   step 或覆盖新版本。

<details>
<summary>提示</summary>

[「故障、检查点与可观测性」](06-faults-checkpoints-and-observability.md)
给出 checkpoint 字段清单（commit version、acknowledged）和幂等性
要求；[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)
的 allocation 记录提供 `lease_version` 语义。从“同一
`job_id + attempt + step` 只能生效一次”出发设计提交与拒绝规则。

</details>

8. 【进阶】分别解释 queue wait、collective wait、straggler time 和 recovery
   replay；为什么只报告 GPU utilization 不够定位瓶颈？

<details>
<summary>提示</summary>

这四个时间项分属
[「集群负载、系统分层与能力边界」](01-cluster-workload-and-boundary.md)
作业时间模型的不同分量；在哪一层观测它们，见
[「故障、检查点与可观测性」](06-faults-checkpoints-and-observability.md)
的四层指标聚合。反例思路：构造一个 GPU 一直“忙”、作业却因排队
或重放而变慢的场景。

</details>


## Rust 与 API 题

1. 【基础】为 `Job` 增加 tenant、priority 和 memory quota，保持资源不满足时不
   允许部分启动。

<details>
<summary>提示</summary>

从示例 crate 的 `Job` 结构、`validate_job` 和 `choose_placement`
的显存过滤入手，用测试
`gang_admission_does_not_start_partial_jobs` 保住成组不变量；
配额语义对照
[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)：
quota 检查应发生在准入，而不是放置成功之后。

</details>

2. 【基础】为 `PlacementPolicy` 实现一个 first-fit/backfill 变体，测试它不会
   让已承诺的 FIFO 作业无限饥饿。

<details>
<summary>提示</summary>

改 `examples/ch09-cluster-simulator` 的 `choose_placement` 时，
注意“队首放不下就停止准入”的逻辑在 `admit_jobs` 里；为队首阻塞、
资源归还和最终准入写虚拟时间测试。饥饿判据可借用
[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)
对 backfilling 的约束：被越过的队首作业的启动承诺不能被破坏。

</details>

3. 【进阶】将 `communication_cost` 抽象为 `CostModel` trait，写一个不依赖真实
   时间的 mock，测试 bytes 增加时成本单调。

<details>
<summary>提示</summary>

现有自由函数 `communication_cost` 的输入输出就是 trait 的候选
签名，测试 `communication_cost_is_monotonic_in_message_size` 是
单调性断言的模板。对照
[「拓扑感知放置与集合通信成本」](04-topology-aware-placement-and-communication.md)
想清楚 trait 边界：mock 应该在哪一层替换——bytes 分域之后，
还是虚拟时间计价之后。

</details>

4. 【进阶】为 `TraceEvent` 增加 lease version 和 network domain，测试旧 attempt
   的 completion 事件不能确认新 attempt。

<details>
<summary>提示</summary>

示例 crate 的 `TraceEvent`/`TraceRecord` 已带 `attempt` 和
schema 版本字段；`lease_version` 的语义见
[「多租户、配额与资源碎片」](05-multitenancy-and-fragmentation.md)
的 allocation 记录。测试思路：构造一个失败重试作业，断言携带旧
版本号的 completion 事件被拒绝，而不是写进新 attempt 的报告。

</details>

5. 【进阶】将 failure step 改为节点故障，要求同一故障域中的 GPU 一起释放，并从
   最近的有效 checkpoint 恢复。

<details>
<summary>提示</summary>

`Gpu` 已带 `node`/`rack` 字段，失败路径集中在示例 crate 的
`handle_failure` 和 `release`；对照
[「故障、检查点与可观测性」](06-faults-checkpoints-and-observability.md)
的故障状态机，把故障单位从单个作业换成 `(rack, node)` 域：受影
响的每个 placement 都要走“释放 → 定位 checkpoint → 重新入队”。

</details>

6. 【进阶】为模拟器加入 `Result` 错误边界：非法 GPU 数、显存不足、重复 job id、
   retry limit 和 scheduler deadlock 都必须有描述性错误。

<details>
<summary>提示</summary>

`SimulationError` 已有 `DuplicateJobId`、`JobDoesNotFit`、
`RetryLimitExceeded`、`SchedulerDeadlock` 等变体，入口校验集中
在 `validate_job` 与 `simulate` 主循环。参照
[「作业队列、资源向量与成组调度」](03-job-queue-and-resource-scheduling.md)
的状态机逐条问：哪些非法输入应在准入前拒绝、哪些只能在运行中
暴露，并为每个错误写最小触发夹具。

</details>

7. 【进阶】阅读固定 Burn 的 `ExecutionStrategy`，实现一个只打印设备集合和策略
   的 adapter；不要把它命名为 cluster scheduler。

<details>
<summary>提示</summary>

入口是本章列出的
`burn/crates/burn-train/src/learner/supervised/strategies/base.rs`
（以 `pins.toml` 的 revision 为准，而不是在线最新文档）；命名
边界见[「作业队列、资源向量与成组调度」](03-job-queue-and-resource-scheduling.md)
的“Burn 入口不是作业调度器”：adapter 拿到的只是调用者给的设备
列表，没有队列、租约和 rank rendezvous。

</details>


## 性能与系统题

1. 【进阶】固定 job 数和 step 数，比较 FIFO 与 topology-aware 的 queue wait、
   makespan、cross-rack bytes 和 collective time。

<details>
<summary>提示</summary>

[「实验：CPU 集群调度与故障模拟器」](07-cpu-cluster-simulator-lab.md)
的主程序已对同一组作业分别跑两种策略并打印这四个指标，测试
`topology_aware_placement_reduces_cross_rack_bytes` 是最小对照。
先预测每个指标的大小关系再跑模拟核对，重点解释与预测不一致的
那一项。

</details>

2. 【挑战】改变 `gradient_bytes_per_step`、`cross_node_multiplier` 与
   `cross_rack_multiplier`，验证通信成本模型随每步字节与三档拓扑域的
   单调性，并说明它不是实测带宽。

<details>
<summary>提示</summary>

用 `NetworkModel::with_multipliers` 构造参数，仿照测试
`communication_cost_distinguishes_three_placement_domains` 对同
节点、同机柜跨节点、跨机柜三种 placement 分别断言，每次只改一
个参数。“不是实测带宽”的理由见
[「实验：CPU 集群调度与故障模拟器」](07-cpu-cluster-simulator-lab.md)：
虚拟微秒来自确定性乘法，不来自任何 NIC。

</details>

3. 【挑战】设计一个 `p95` queue wait 报告，解释为什么平均等待时间会掩盖租户
   饥饿和队首阻塞。

<details>
<summary>提示</summary>

`SimulationResult` 已带 `p95_queue_wait_us` 字段（实现见示例
crate 的 `percentile_95`）；
[「作业队列、资源向量与成组调度」](03-job-queue-and-resource-scheduling.md)
的队首阻塞场景是现成的构造起点。让大多数作业等待很短、少数
作业等待极长，再比较均值和 p95 各自报告了什么。

</details>

4. 【挑战】为每个 rank 记录 compute、collective、wait、checkpoint 和 retry，
   设计一个能定位 straggler 的聚合指标。

<details>
<summary>提示</summary>

straggler 的定义和四层指标聚合见
[「故障、检查点与可观测性」](06-faults-checkpoints-and-observability.md)；
示例 crate 的 `TraceRecord` 目前只到 job 粒度，扩展时要把事件
降到 rank 并带上 node/rack。指标设计的关键是把“单个 rank 慢”
与“所有 rank 的 collective 一起变慢”区分开。

</details>

5. 【挑战】比较 checkpoint interval 较小和较大时的写入开销与 failure replay；
   给出 time-to-recovery 的成本模型。

<details>
<summary>提示</summary>

在示例 crate 里调 `Job::checkpoint_interval` 与
`SimulationConfig::checkpoint_cost_us`，配合 `with_failure_step`
注入故障，比较 `checkpoint_replay_steps` 和 `makespan_us` 的
变化。成本模型不必从零推：
[「故障、检查点与可观测性」](06-faults-checkpoints-and-observability.md)
已推导 Young 近似间隔，把它扩展到 time-to-recovery 即可。

</details>

6. 【挑战】为同一个 job 设计节点内、同 rack、跨 rack 三种 placement，列出需要
   真实实验记录的硬件、driver、通信库、rank 和同步信息。

<details>
<summary>提示</summary>

[「GPU 节点、机柜与网络拓扑」](02-gpu-node-and-network-topology.md)
的“迁移到真实集群时要记录什么”是这份清单的起点；先用
`Cluster::uniform_interleaved(2, 2, 2, ...)` 在模拟器里造出三种
placement（测试
`communication_cost_distinguishes_three_placement_domains` 的
夹具），再逐项想哪些字段是虚拟时间模型根本没有、必须真实实验
才能补上的。

</details>


## 源码题

1. 【进阶】阅读 `burn/crates/burn-train/src/learner/supervised/strategies/base.rs`，
   区分 `MultiDevice` 与 `DistributedDataParallel` 的设备范围和优化策略。

<details>
<summary>提示</summary>

第 6 章[「本机多设备与数据并行」](../ch06/05-local-data-parallel.md)
和[「集合通信、DDP 与能力边界」](../ch06/06-collective-and-ddp.md)
分别对应这两种策略，本章
[「集群负载、系统分层与能力边界」](01-cluster-workload-and-boundary.md)
概括了三种 `ExecutionStrategy`。读源码时盯住两点：设备集合从哪
里来，梯度在主设备聚合还是走 collective。

</details>

2. 【进阶】阅读 `burn/crates/burn-tensor/src/tensor/distributed.rs`，追踪
   `DistributedContext::init`、`all_reduce`、`resolve` 和
   `sync_collective` 的生命周期。

<details>
<summary>提示</summary>

调用顺序对照第 6 章
[「集合通信、DDP 与能力边界」](../ch06/06-collective-and-ddp.md)
的 AllReduce 语义与完成边界；注意第 9 章模拟器只建立成本协议，
不调用这些 Burn collective API。

</details>

3. 【进阶】阅读 `burn/crates/burn-backend/src/backend/distributed/ops.rs`，说明
   `register_sync_parameters`、`submit_gradient_sync` 和默认 collective
   实现之间的关系。

<details>
<summary>提示</summary>

[「拓扑感知放置与集合通信成本」](04-topology-aware-placement-and-communication.md)
的“Burn 的 collective 边界”列出这组入口的分工（启动/关闭
server、注册参数、提交同步、`all_reduce`/`sync_collective`）。
按 `pins.toml` 固定 revision 读 `ops.rs`，画“注册 → 提交 →
完成”的时序，标出哪些方法有默认实现、哪些必须由 backend 提供。

</details>

4. 【进阶】阅读 `burn/crates/burn-flex/src/ops/transaction.rs`，解释为什么 Flex
   CPU 不是本章 AllReduce 的运行验证 backend。

<details>
<summary>提示</summary>

[「拓扑感知放置与集合通信成本」](04-topology-aware-placement-and-communication.md)
的“Flex CPU 不能作为 collective 实验”就是这道题的正文依据：在
`transaction.rs` 里找到明确写明不支持 collective 的位置，再把
“trait 有默认实现、代码能编译”与“运行时真的能归约”区分开。

</details>

5. 【进阶】阅读 `cubecl/crates/cubecl-runtime/src/client.rs` 和
   `stream/scheduler.rs`，画出 `launch → flush → read/sync` 与本地 stream
   对齐的边界。

<details>
<summary>提示</summary>

完成边界的讲解在第 4 章
[「内存、Stream 与异步执行」](../ch04/06-memory-streams-execution.md)；
本章[「GPU 节点、机柜与网络拓扑」](02-gpu-node-and-network-topology.md)
的“Burn/CubeCL 的局部视角”解释这些边界为何只属于设备运行时。
画图时区分“提交到 stream”和“确认完成”分别发生在哪个调用上。

</details>

6. 【进阶】阅读 `cubecl/crates/cubecl-cpu/src/runtime.rs`，说明 CPU runtime 的
   `SERVER_COMM_ENABLED` 与集群通信之间的差异。

<details>
<summary>提示</summary>

`SERVER_COMM_ENABLED` 是 `ServerCommunication` trait 的常量，
描述 server 之间（如 peer-to-peer）的数据搬运是否可用；从
`runtime.rs` 找到 `CpuServer` 后，顺着它的 `ServerCommunication`
实现看取值与注释。再对照
[「集群负载、系统分层与能力边界」](01-cluster-workload-and-boundary.md)
的三层边界想：这个开关管到哪一层，rank rendezvous、成员管理和
故障恢复又在哪一层。

</details>


## OpenMLSys v1 来源

本章逐文件参考本书固定版本
`9c289782ccbb165ac8ad7c960ecffc12942a5560`：

- `openmlsys/v1/zh_chapters/chapter_distributed_training/index.md`：
  保留分布式训练的动机、并行方法和集合通信/参数服务器地图；
- `overview.md`：保留算力、内存、经济性和硬件故障动机；
- `methods.md`：保留数据、模型、混合和流水线并行，以及通信与微批次问题；
- `cluster.md`：保留 GPU 服务器、rack、ToR、Spine、跨机柜瓶颈和节点内
  互连层次；
- `collective.md`：保留 Broadcast、Reduce、AllReduce、AllGather、
  Scatter 和 `alpha + beta * bytes` 成本模型；
- `parameter_servers.md`：保留同步/异步更新、straggler、Push/Pull、
  副本、热点和一致性取舍；
- `summary.md`：保留规模化训练、AllReduce、参数服务器和故障动机。

旧章节中的硬件规格、厂商性能数字、图片、具体框架实现和外部链接不被
当作当前固定 Burn 能力。新增的 queue、quota、lease、failure detector、
cluster telemetry 和 CPU simulator 是本书的框架无关系统设计。

## Burn/CubeCL 固定源码入口

主 Burn revision 是 `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`，
CubeCL revision 是 `be278a1e76aed881e2cc6b165414ee6103ca4634`：

- `burn/crates/burn-train/src/learner/supervised/strategies/base.rs`
- `burn/crates/burn-train/src/learner/supervised/strategies/ddp/strategy.rs`
- `burn/crates/burn-tensor/src/tensor/distributed.rs`
- `burn/crates/burn-backend/src/backend/distributed/ops.rs`
- `burn/crates/burn-backend/src/backend/distributed/server.rs`
- `burn/crates/burn-autodiff/src/distributed.rs`
- `burn/crates/burn-flex/src/ops/transaction.rs`
- `burn/crates/burn-cubecl/src/ops/distributed.rs`
- `cubecl/crates/cubecl-runtime/src/client.rs`
- `cubecl/crates/cubecl-runtime/src/stream/scheduler.rs`
- `cubecl/crates/cubecl-runtime/src/memory_management/`
- `cubecl/crates/cubecl-cpu/src/runtime.rs`
- `cubecl/crates/cubecl-cuda/src/compute/server.rs`

固定源码可核验设备/通信/运行时入口；不能据此声称已有集群作业队列、
拓扑感知放置、多租户 quota、跨节点 rendezvous、elastic membership、
自动重试、分布式 checkpoint 共识或统一集群遥测。这些能力在真实系统
中的实现（Borg、Gandiva、Tiresias 等）见附录
[参考文献](../references.md#第-9-章-大规模-gpu-集群管理)。

## 本章系统结论

1. 集群控制面负责作业、资源与故障；训练数据面负责 rank 间通信——两层不能混称。
2. gang scheduling、按“同节点→同机柜→跨机柜”收紧的拓扑放置，与
   $\alpha+\beta$ 通信成本共同决定 makespan；链路每往外跨一档，带宽
   通常低 1–2 个数量级。
3. CPU 模拟器验证了队列、成组准入、故障重放与确定性 trace，不测量真实 GPU/NCCL。
4. GPU 阅读时应把第 6 章一次 AllReduce 的字节量放进机柜/链路模型重算成本。
5. Burn/CubeCL 源码可定位设备与 collective 数据面入口，但不提供作业队列实现。
6. 不能把虚拟时间或放置结果当成 GPU benchmark。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-9-章)。
