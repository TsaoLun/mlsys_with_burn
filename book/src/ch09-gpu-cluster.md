# 第 9 章 大规模 GPU 集群管理

当训练或推理跨越许多节点，决定「谁获得哪些 GPU、通信走哪条链路、失败
之后从哪恢复」的不再是训练循环内部的 `all_reduce`，而是集群控制面。
本章对应 OpenMLSys 分布式训练里的集群内容，并补上成组调度、多租户和
故障协议。

产业里对应 Slurm、Kubernetes 的设备插件、内部 GPU 调度器（如 Borg
系），以及 NCCL 所在的机柜网络。Burn 的 DDP 入口是数据面案例，不是
作业队列。默认实验用纯 Rust 虚拟时间模拟器观察放置与故障，不测量真
机集群。

## 本章问题

当作业跨越大量节点后，系统如何调度昂贵的加速器、处理故障并定位性能
瓶颈？控制面、训练数据面和设备运行时各管什么？

## 学习目标

完成本章后，你应该能够：

1. 区分集群控制面、训练通信数据面和 GPU 设备运行时；
2. 用 GPU、节点、机柜、ToR 和 Spine 描述通信域；
3. 为一个成组调度作业写出 GPU、显存和网络域需求；
4. 用 $\alpha + \beta \cdot \text{bytes}$ 解释集合通信，并区分同节点 /
   同机柜 / 跨机柜三档成本；
5. 解释 FIFO、拓扑感知放置、配额和资源碎片之间的取舍；
6. 设计带 checkpoint、attempt、step 和幂等确认的故障恢复协议；
7. 把第 6 章一次 AllReduce 的字节量放进机柜模型重算成本；
8. 在离散事件模拟器中观察队列等待、跨机柜流量和重试。

## 先修知识

建议先完成第 6 章的训练状态与集合通信。需要理解基本的离散事件模拟。
不要求本机已有 GPU 集群。

## 本章路线

自上而下走四层：workload card → 控制面（queue、admission、placement、
recovery）→ 训练数据面（all-reduce、checkpoint）→ 设备运行时。

![控制面负责任务与资源，训练数据面负责集合通信与设备执行；放置结果改变通信域从而进入 makespan](img/ch06-ch09-control-data-planes.svg)

相对第 6 章：那里讲清梯度同步数据面期望什么；本章把一次 AllReduce 的
字节量放进机柜/链路，看放置如何改变成本。

## 小节

1. [集群负载、系统分层与能力边界](ch09/01-cluster-workload-and-boundary.md)
2. [GPU 节点、机柜与网络拓扑](ch09/02-gpu-node-and-network-topology.md)
3. [作业队列、资源向量与成组调度](ch09/03-job-queue-and-resource-scheduling.md)
4. [拓扑感知放置与集合通信成本](ch09/04-topology-aware-placement-and-communication.md)
5. [多租户、配额与资源碎片](ch09/05-multitenancy-and-fragmentation.md)
6. [故障、检查点与可观测性](ch09/06-faults-checkpoints-and-observability.md)
7. [实验：CPU 集群调度与故障模拟器](ch09/07-cpu-cluster-simulator-lab.md)
8. [练习、延伸阅读与来源](ch09/08-exercises-and-sources.md)

读完全书主线后，可回到[综合实验](capstone.md)，把数据、训练、状态保存
和推理恢复再走一遍。

示例位于 `examples/ch09-cluster-simulator`，只使用 Rust 标准库和虚拟时间。
