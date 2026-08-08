# 第 9 章 大规模 GPU 集群管理

## 本章问题

当训练跨越大量节点后，系统如何调度昂贵的加速器、处理故障并定位性能
瓶颈？

## 学习目标

完成本章后，你应该能够：

1. 区分集群控制面、训练通信数据面和 GPU 设备运行时；
2. 用 GPU、节点、机柜、ToR（Top of Rack）和 Spine 描述通信域；
3. 为一个成组调度（gang scheduling）作业写出 GPU、显存和网络域需求；
4. 用 `alpha + beta * bytes` 模型解释集合通信和拓扑放置的关系；
5. 解释 FIFO、拓扑感知放置、配额和资源碎片之间的取舍；
6. 设计带 checkpoint、attempt、step 和幂等确认的故障恢复协议；
7. 区分 Burn/CubeCL 已核验的本机/设备接口与外部集群调度职责；
8. 在 CPU 离散事件模拟器中观察队列等待、跨机柜流量、通信成本和重试。

## 先修知识

建议先完成第 4 章的 stream、内存和执行边界，第 6 章的训练状态、数据
并行、集合通信和 checkpoint，第 7 章的部署成本。需要理解 Rust 的
`struct`、`enum`、trait、集合类型和基本的离散事件模拟；不要求 CUDA、
NCCL、RDMA 或真实 GPU 集群。

## 本章路线

```text
workload card
    │ GPU / memory / bytes / failure domain
    ▼
control plane ── queue ── admission ── placement ── recovery
    │
    ▼
training data plane ── all-reduce / all-gather / checkpoint
    │
    ▼
device runtime ── stream / memory pool / kernel / sync
```

先定义集群作业的资源和完成条件，再从硬件拓扑推导通信成本。之后讨论
队列、成组调度、拓扑放置、多租户与故障协议，最后将这些框架无关的
概念放入一个纯 Rust 模拟器。Burn 的 DDP、`DistributedContext` 和
CubeCL `ComputeClient` 只作为训练数据面和设备运行时的固定源码案例；
它们不是集群控制面实现。

## 小节

1. [集群负载、系统分层与能力边界](ch09/01-cluster-workload-and-boundary.md)
2. [GPU 节点、机柜与网络拓扑](ch09/02-gpu-node-and-network-topology.md)
3. [作业队列、资源向量与成组调度](ch09/03-job-queue-and-resource-scheduling.md)
4. [拓扑感知放置与集合通信成本](ch09/04-topology-aware-placement-and-communication.md)
5. [多租户、配额与资源碎片](ch09/05-multitenancy-and-fragmentation.md)
6. [故障、检查点与可观测性](ch09/06-faults-checkpoints-and-observability.md)
7. [实验：CPU 集群调度与故障模拟器](ch09/07-cpu-cluster-simulator-lab.md)
8. [练习、延伸阅读与来源](ch09/08-exercises-and-sources.md)

示例代码位于 `examples/ch09-cluster-simulator`，只使用 Rust 标准库和虚拟
时间。它验证调度协议、通信成本模型、checkpoint 恢复和资源归还；它不
测量真实 GPU、网络或 NCCL 性能，也不声称 Burn 固定快照提供作业队列、
多租户隔离、弹性 membership 或自动故障迁移。

## 来源与改编说明

本章改编并重组 OpenMLSys v1 `chapter_distributed_training/` 中的系统概述、
并行方法、集合通信、参数服务器和集群架构内容。新增的队列、配额、故障域、
遥测字段和 CPU 模拟器是框架无关的系统设计材料；固定 Burn/CubeCL 源码
只用于核验设备、通信、stream、内存和训练入口的边界。本章没有复用上游
硬件图片或历史性能数字。

## 证据状态

以下标签是本书的阅读证据分类，不代表 Burn 官方能力等级；完整定义见
[逐文件对照矩阵导读](crosswalk-guide.md)。

- `CPU 可运行验证`：队列、gang admission、拓扑放置、通信成本、故障
  retry、checkpoint replay 和资源归还；
- `源码核验`：Burn/CubeCL 的设备、stream、memory、collective 和
  training data-plane 入口；
- `协议/成本模型`：控制面、故障域、队列公平、链路热点和
  machine-readable trace；
- `可选平台实验`：真实 GPU 集群、NCCL/RDMA、网络拥塞、多租户 runtime
  和弹性 membership；
- `未覆盖`：把模拟器虚拟时间、放置结果或通信 penalty 当作 GPU
  benchmark。

对应 trace schema、队列指标和控制面边界见[核心主题比较卡](comparison-cards.md#第-9-章gpu-集群与控制面)。

