# 集合通信、DDP 与能力边界

## AllReduce 的语义

集合通信（collective communication）要求一组参与者共同调用同一个操作。
对数据并行训练最常见的是 AllReduce：每个设备提供一个同形状 tensor，
所有设备得到聚合后的结果：

$$
x\_i' = \operatorname{reduce}(x\_0, x\_1, \ldots, x\_{p-1}).
$$

梯度平均时 `reduce` 通常是 `Mean`；如果每个设备先保存的是未归一化的
梯度，也可以先 `Sum`，再由应用按全局样本数除法。`Broadcast`、`Reduce`、
`AllGather` 和 `ReduceScatter` 有不同的输入/输出布局，不能因为都经过
网络就互换。

## 通信成本

OpenMLSys v1 `collective.md` 使用 $\alpha+\beta l$ 描述传输一个长度为
$l$ 的消息，其中 $\alpha$ 是延迟项，$\beta$ 反映字节传输成本。对训练
而言，AllReduce 的实际成本还受算法和拓扑影响：

$$
T\_{\text{collective}} =
T(\text{algorithm},\ \text{topology},\ p,\ \text{bytes},\ \text{dtype}).
$$

树、环、分层或硬件专用实现会在延迟、链路利用率和临时 buffer 之间取舍。
因此不能把一个本地小 tensor 的时间线性外推到跨机柜的大梯度，也不能只用
链路标称带宽代替 collective 的端到端测量。

## Burn 的 DDP 分层

固定快照中的 DDP 不是 `burn-train` 自己实现一个网络协议，而是跨越多层：

```text
ExecutionStrategy::ddp
        │
DistributedContext::init
        │
Dispatch / backend DistributedOps
        │
autodiff distributed registration
        │
backend all_reduce + sync_collective
        │
runtime / device collective implementation
```

`ExecutionStrategy::ddp(devices, DistributedConfig)` 创建
`DistributedContext`。context 的生命周期负责启动和关闭该 backend 的
通信协调器；`DistributedConfig` 在固定快照中至少包含 `Sum` 或 `Mean`
这样的 `ReduceOperation` 选择。

## DDP worker 的生命周期

固定 `burn-train` 的 DDP 代码可以按以下步骤阅读：

1. `DdpTrainingStrategy` 对每个本地 device 启动一个 `DdpWorker`；
2. train loader 用 `split_dataloader` 分给各 worker；
3. 每个 worker fork 自己的 model，并调用 `grad_sharded()` 标记需要
   同步的参数；
4. worker 执行 forward/backward；
5. autodiff 将分布式参数登记到 sync server，并在反向过程中提交梯度；
6. backend server 等到参与者和参数的要求数满足后执行 `all_reduce`；
7. `sync_collective` 形成可访问结果的同步边界；
8. 主 worker 承担 validation、event processing、checkpoint 和最终 model。

第 6 步不是“把 Rust channel 中的几个 Tensor 相加”。固定
`burn-backend/src/backend/distributed/server.rs` 会跟踪 parameter ID、
各设备登记次数和待归约 tensor；底层 backend 决定真正的 collective 实现。

## `CollectiveTensor` 的完成语义

`burn-tensor/src/tensor/distributed.rs` 中的 `all_reduce` 返回
`CollectiveTensor`，而不是一个已经可以无条件读取的普通 Tensor。调用
`resolve()` 会先 `sync_collective`，再构造可使用的 Tensor；底层源码还提供
不安全的 `assume_resolved()`，调用者必须自己保证同步已经发生。

这和第 4 章的异步执行边界相同：提交 operation、得到 handle、读取结果
是三个不同的时间点。教学代码应把 resolve/sync 放在明确位置，不用一次
`println!` 偶然触发的 read 去掩盖协议。

## DDP 的范围与参与责任

固定 `burn-train` 的 DDP README 明确说明：

- 每个本地 device 有一个线程和模型 replica；
- 每个节点都要由用户启动 DDP；
- 所有节点的 collective configuration 必须匹配；
- 第一台 device 是 main device，负责 validation 和 UI/event。

这提供了训练策略和后端 collective 的组合点，但没有从源码推出以下能力：

- 集群调度、进程发现和认证；
- 节点故障后的自动重试或 elastic membership；
- 参数服务器的 push/pull、异步版本或副本共识；
- pipeline stage 调度、micro-batch 编排和 activation recomputation；
- 跨节点 checkpoint 原子提交。

这些是系统设计问题，可以在 OpenMLSys 的框架无关叙事中学习，但不能包
装成固定 Burn API 已经提供。

## 固定后端的验证边界

`burn-flex/src/ops/transaction.rs` 的注释明确写出 Flex 不支持 collective
operations，因此 `Device::flex().autodiff()` 适合本章 CPU 训练循环，不适合
运行 DDP AllReduce。`burn-cubecl/src/ops/distributed.rs` 提供 CubeCL
backend 的 all-reduce 调用；固定 CubeCL CUDA server 使用 NCCL，但这条
路径需要 CUDA/NCCL、多个可用设备和匹配的运行时配置。

`burn-communication` 的 WebSocket/channel 和 tensor data service 也不能
直接当成 DDP 梯度同步：前者是网络传输抽象，后者是远程 tensor 数据服务；
DDP 的梯度聚合仍走 backend `DistributedOps`。这一层次区分避免把“有网络
模块”误读成“已有跨节点训练协议”。

## 何时需要参数服务器

参数服务器适合在同步 collective 的等待成本、热点参数或 straggler 成为
主要问题时讨论。它需要至少定义：

```text
worker push gradient
        │ version / quorum
parameter server update
        │
worker pull parameters
```

异步版本允许不同 worker 使用不同参数版本，换来较少等待但增加 stale
gradient 和收敛分析难度。固定 Burn `burn-train` 源码没有相应的
parameter-server strategy；本章只把它作为对照协议。

如果把一次异步更新写成：

$$
\theta\_{v+1}=U(\theta\_v,\ g(\theta\_{v-k};B),\ s\_v),
$$

其中 $v$ 是 server 当前版本、$k$ 是梯度产生时的版本差距，那么协议至少
要决定：

1. server 是否拒绝过旧的 $v-k$，还是按衰减权重接收；
2. worker pull 到的参数和 optimizer state 是否来自同一个版本；
3. worker、server 或网络失败后，未确认的 push 是否重放；
4. 热点参数是否单独分片，分片之间如何保持 step 或 epoch 语义。

同步 DDP 的 AllReduce 不回答这些问题，因为它在每个 step 形成共同的
梯度完成点。参数服务器也不能只用“异步 channel + optimizer.step”替代；
版本、确认、幂等和 checkpoint 一起构成了它的训练协议。
