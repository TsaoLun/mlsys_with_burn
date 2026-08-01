# 9.8 练习、延伸阅读与来源

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

## 概念题

1. 为什么“集群有足够 GPU”不等于一个同步作业可以立即启动？请分别从
   显存、成组调度和通信域回答。
2. 画出控制面、训练数据面、设备运行时三层，并把队列等待、
   `all_reduce`、`ComputeClient::sync` 放到正确层。
3. 用 `alpha + beta * bytes` 比较同机柜、跨机柜和跨 Spine 的消息成本；
   说明这个模型遗漏了哪些真实网络因素。
4. FIFO 和 topology-aware placement 的公平性目标有什么不同？为什么
   “减少跨机柜 bytes”可能增加某个租户的 queue wait？
5. 构造一个 GPU 总容量足够但发生资源碎片的例子；说明保留大块资源和
   backfilling 的取舍。
6. 为什么一个 `DistributedContext` 不能自动说明节点 membership、rank
   rendezvous 和故障恢复已经完成？
7. 设计一个 checkpoint commit 协议，防止旧 attempt 重复提交 optimizer
   step 或覆盖新版本。
8. 分别解释 queue wait、collective wait、straggler time 和 recovery
   replay；为什么只报告 GPU utilization 不够定位瓶颈？

## Rust 与 API 题

1. 为 `Job` 增加 tenant、priority 和 memory quota，保持资源不满足时不
   允许部分启动。
2. 为 `PlacementPolicy` 实现一个 first-fit/backfill 变体，测试它不会
   让已承诺的 FIFO 作业无限饥饿。
3. 将 `communication_cost` 抽象为 `CostModel` trait，写一个不依赖真实
   时间的 mock，测试 bytes 增加时成本单调。
4. 为 `TraceEvent` 增加 lease version 和 network domain，测试旧 attempt
   的 completion 事件不能确认新 attempt。
5. 将 failure step 改为节点故障，要求同一故障域中的 GPU 一起释放，并从
   最近的有效 checkpoint 恢复。
6. 为模拟器加入 `Result` 错误边界：非法 GPU 数、显存不足、重复 job id、
   retry limit 和 scheduler deadlock 都必须有描述性错误。
7. 阅读固定 Burn 的 `ExecutionStrategy`，实现一个只打印设备集合和策略
   的 adapter；不要把它命名为 cluster scheduler。

## 性能与系统题

1. 固定 job 数和 step 数，比较 FIFO 与 topology-aware 的 queue wait、
   makespan、cross-rack bytes 和 collective time。
2. 改变 `gradient_bytes` 与 `cross_rack_multiplier`，验证通信成本模型的
   单调性，并说明它不是实测带宽。
3. 设计一个 `p95` queue wait 报告，解释为什么平均等待时间会掩盖租户
   饥饿和队首阻塞。
4. 为每个 rank 记录 compute、collective、wait、checkpoint 和 retry，
   设计一个能定位 straggler 的聚合指标。
5. 比较 checkpoint interval 较小和较大时的写入开销与 failure replay；
   给出 time-to-recovery 的成本模型。
6. 为同一个 job 设计节点内、同 rack、跨 rack 三种 placement，列出需要
   真实实验记录的硬件、driver、通信库、rank 和同步信息。

## 源码题

1. 阅读 `burn/crates/burn-train/src/learner/supervised/strategies/base.rs`，
   区分 `MultiDevice` 与 `DistributedDataParallel` 的设备范围和优化策略。
2. 阅读 `burn/crates/burn-tensor/src/tensor/distributed.rs`，追踪
   `DistributedContext::init`、`all_reduce`、`resolve` 和
   `sync_collective` 的生命周期。
3. 阅读 `burn/crates/burn-backend/src/backend/distributed/ops.rs`，说明
   `register_sync_parameters`、`submit_gradient_sync` 和默认 collective
   实现之间的关系。
4. 阅读 `burn/crates/burn-flex/src/ops/transaction.rs`，解释为什么 Flex
   CPU 不是本章 AllReduce 的运行验证 backend。
5. 阅读 `cubecl/crates/cubecl-runtime/src/client.rs` 和
   `stream/scheduler.rs`，画出 `launch → flush → read/sync` 与本地 stream
   对齐的边界。
6. 阅读 `cubecl/crates/cubecl-cpu/src/runtime.rs`，说明 CPU runtime 的
   `SERVER_COMM_ENABLED` 与集群通信之间的差异。

## OpenMLSys v1 来源

本章逐文件参考固定 revision
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
自动重试、分布式 checkpoint 共识或统一集群遥测。

## 来源与改编说明

本章保留 OpenMLSys 的分布式训练、拓扑、集合通信、参数服务器和故障
问题，重写为“workload card → control plane → collective data plane →
device runtime”的路线。Burn 部分改为固定源码证据和限制清单；没有把
`ExecutionStrategy`、`DistributedContext` 或 `ComputeClient` 称为集群
调度器。实验和新增 Rust 代码采用 MIT OR Apache-2.0；正文采用
CC BY-NC-SA 4.0。
