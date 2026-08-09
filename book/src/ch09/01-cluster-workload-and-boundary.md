# 集群负载、系统分层与能力边界

## 先把“集群训练”写成一个负载卡片

集群不是“把很多 GPU 放在一起”。一个作业至少需要声明：

- **计算**：GPU 数量、每步计算时间、step 数和是否要求成组启动；
- **内存**：每张 GPU 的显存需求、通信 buffer、activation 和 checkpoint
  的暂存空间；
- **通信**：梯度或 activation 的字节数、通信频率、可接受的跨机柜流量；
- **数据**：dataset shard、shuffle seed、输入读取速率和数据位置；
- **故障域**：允许在哪些节点、机柜或网络域上运行，以及失败后从哪里恢复；
- **目标**：time-to-train、吞吐、成本、可用性或租户公平性。

同一个模型在单节点和跨机柜集群上并不是同一个 workload。参数量只说明
存储的一部分成本；梯度同步频率、显存峰值、链路位置和 checkpoint 周期
同样决定作业是否适合某种集群。

## 三层系统边界

本章使用下面三层区分容易混淆的“调度”：

1. **控制面（control plane）**：接收作业、维护队列、检查配额、选择资源、
   做成组准入、处理抢占和故障恢复；
2. **训练数据面（training data plane）**：在已经分配的 rank/device 之间
   传输梯度、参数和 activation，执行 AllReduce、AllGather 或
   ReduceScatter；
3. **设备运行时（device runtime）**：在一张 GPU 或一个设备 backend 内
   管理 kernel、stream、buffer、编译缓存和完成同步。

```text
控制面：job → admission → placement → retry
                         │
                         ▼
数据面：rank/device ↔ collective ↔ rank/device
                         │
                         ▼
运行时：ComputeClient → stream → kernel → memory → sync
```

Fusion 的 operation queue、CubeCL 的 stream scheduler 和集群的作业队列都
可以被称为“调度”，但它们的对象和故障语义不同。前者安排已经进入一个
进程或设备的工作；后者决定哪个租户的作业可以占用哪组设备。不能因为
Burn 有 `ExecutionStrategy`，就推导出它有集群公平调度器。

## 一个最小的作业时间模型

对一个已完成准入的作业，可以先把墙钟时间拆成：

$$
T\_{\text{job}} =
T\_{\text{queue}} + T\_{\text{compute}} +
T\_{\text{collective}} + T\_{\text{checkpoint}} +
T\_{\text{recovery}}.
$$

其中 `queue` 是控制面等待，`compute` 是设备工作，`collective` 是数据面
通信，`checkpoint` 是状态提交，`recovery` 是失败后重做或恢复的代价。
如果只测 `forward` 或只测一轮 kernel，就无法解释作业为什么仍然排队、
等待最慢 rank 或在故障后变慢。

同步训练还需要区分两个时间：

- `step time`：所有参与者完成计算和 collective 后才能进入下一步；
- `makespan`：作业从实际开始到完成（包括失败和重试）的虚拟时间。

一个快但经常失败的 placement 可能有更小的单步时间，却有更大的
`makespan`。一个跨机柜通信较多的 placement 也可能不影响单卡 kernel，
但会增加 `T_collective` 和网络热点。

## 本版 Burn 能说明什么

固定 Burn 源码中，`burn-train` 的 `ExecutionStrategy` 有：

- `SingleDevice`：单设备训练；
- `MultiDevice`：单进程内的多设备数据并行，可选择主设备优化或分片优化；
- `DistributedDataParallel`：为本节点设备建立 `DistributedContext`，通过
  collective 同步梯度。

`burn-tensor` 的 `DistributedContext::init` 接收一组设备并启动通信服务；
`CollectiveTensor::resolve` 通过 `sync_collective` 等待 collective 完成。
这些接口描述的是已知设备之间的训练同步，不包含 job id、租户、队列、
rank rendezvous、资源配额或节点故障处理。

本章后续会用 Burn 源码回答“数据面有哪些入口”，再用模拟器回答“控制面
还需要哪些协议”。两者必须分开验证。

## 本节小结

集群系统的第一个抽象不是 GPU 数量，而是带有内存、通信、故障域和目标的
workload card。`queue wait`、`collective`、`checkpoint` 和 `recovery`
都应成为可观测的时间项。Burn/CubeCL 本版可以作为设备和通信数据面
的案例，但不能替代集群控制面。
