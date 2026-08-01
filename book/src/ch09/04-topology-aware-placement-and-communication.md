# 9.4 拓扑感知放置与集合通信成本

## 先区分集合通信的语义

集合通信（collective communication）不是一个“网络发送”函数，而是由
一组 rank 共同发起、共同完成的协议：

- `Broadcast`：一个 rank 的值到达所有 rank；
- `Reduce`：所有 rank 聚合，结果保存到一个 rank；
- `AllReduce`：聚合结果回到所有 rank；
- `AllGather`：每个 rank 的分片最终被所有 rank 看见；
- `ReduceScatter`：聚合后只把对应分片留在各 rank。

同步数据并行通常使用 AllReduce 平均梯度。模型/流水线并行还可能需要
AllGather、ReduceScatter 或点对点 activation 传输。每种语义对字节量、
临时 buffer、完成顺序和拓扑的要求不同。

## `alpha + beta * bytes` 只是第一层模型

OpenMLSys v1 用简化的点对点模型：

$$
T\_{\text{message}} = \alpha + \beta \cdot l.
$$

`alpha` 表示启动/传播延迟，`beta` 表示单位字节传输成本，`l` 是消息
长度。对 collective，还要加上算法轮数、参与者数量和拓扑：

$$
T\_{\text{collective}} =
T(\text{algorithm},\text{topology},p,\text{bytes},\text{dtype}).
$$

同一个 `AllReduce` 在同机柜和跨机柜放置上可能有不同的 `beta` 或额外
排队项。一个教学模拟器可以把 Reduce+Bcast 近似为 `2(p-1)` 个逻辑轮，
再对跨机柜 pair 加 penalty；它不能替代 NCCL 的真实 ring/tree 算法测量。

## 为什么放置会改变训练时间

假设一个作业需要 4 个 rank：

- 放在同一机柜：梯度主要经过节点内互连和同一 ToR；
- 分布在两个机柜：梯度需要共享 ToR–Spine 上行链路；
- 分布在多个机柜：多个上行链路同时竞争，且慢 rank 会延长所有 rank
  的同步等待。

同步 step 的近似成本可以写成：

$$
T\_{\text{step}} \approx
\max\_i(T\_{\text{compute},i}+T\_{\text{load},i})
{}+ T\_{\text{collective}} + T\_{\text{wait}}.
$$

拓扑感知策略并不保证每次都最快。把作业挤在同一机柜可能减少网络成本，
却造成该机柜资源碎片或让其他租户长时间等待。实际策略需要在队列、公平、
显存、链路容量和故障域之间做多目标权衡。

## Burn 的 collective 边界

固定 `burn-backend` 的 `DistributedOps` 定义了：

- 启动和关闭通信 server；
- 注册需要同步的参数；
- 提交梯度同步；
- `all_reduce` 和 `sync_collective` 的后端入口。

`burn-tensor::all_reduce` 接收张量、归约操作和设备列表，返回尚未完成的
`CollectiveTensor`；调用 `resolve` 才建立可继续使用的张量。这个完成边界
对解释“提交”与“可读”很重要。

固定 `burn-cuda`/CubeCL CUDA 路径中可以看到 NCCL collective 的实现入口，
但这只说明该 backend 有设备通信实现。它不自动提供：

- 集群作业队列和 rank/world rendezvous；
- 跨节点成员失效、重试和 elastic join；
- 机柜拓扑发现和 placement；
- 统一的 NCCL 版本、driver、网络和 launcher 配置。

`DistributedContext` 保存的是传入的设备集合；不能把它解释为完整的
cluster membership service。

## Flex CPU 不能作为 collective 实验

固定 `burn-flex/src/ops/transaction.rs` 为 `Flex` 实现了默认的
`DistributedOps`，并明确说明 Flex 不支持 collective operations。因而
第 9 章 CPU 实验不调用 Flex DDP，也不以“API 能编译”证明 AllReduce
运行成立。模拟器中的 rank、链路和 AllReduce 只是纯 Rust 协议模型。

如果读者有两张可用 CUDA GPU，可以把本节扩展为 backend test：
记录设备、driver、CUDA/NCCL、rank 启动方式、消息大小、同步边界和错误
处理，并把结果与 CPU 模拟器分开报告。

## 本节小结

拓扑感知 collective 优化的因果链是：

```text
placement → communication domain → link contention
          → collective time → synchronous step/makespan
```

Burn 提供 collective 的数据面接口和部分 backend 实现；集群控制面仍要
负责拓扑发现、rank 配置、资源租约和失败协议。
