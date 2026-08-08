# 作业队列、资源向量与成组调度

## 作业生命周期

一个集群作业不应只用 `queued` 和 `running` 两个状态。最小状态机可以是：

```text
submitted → admitted → running → checkpointing → completed
                    │          │
                    │          └── failed → recovering → queued
                    └── rejected / preempted
```

每次状态转换都要有可验证的条件：

- `submitted → admitted`：资源、配额、镜像/版本和策略检查通过；
- `admitted → running`：成组资源全部分配，rank/device 映射已固定；
- `running → checkpointing`：训练 step 到达提交边界；
- `failed → recovering`：故障域、attempt 和最近 checkpoint 可定位；
- `recovering → queued`：旧资源已释放，重试不会重复提交已确认 step。

如果只在进程日志里写“开始训练”，就无法区分队列等待、资源分配、设备
初始化和真正的 compute time。

## 资源不是一个整数

调度器面对的资源向量至少包含：

```text
R_job = (gpu_count, memory_per_gpu, node_count,
         network_domain, checkpoint_storage, priority, quota)
```

GPU 数量满足但显存不满足时，作业不能启动；显存满足但必须跨越过多
机柜时，作业可能启动但通信成本不可接受；数量和显存都满足但没有同时
空闲的成组资源时，也不能部分启动。

这就是成组调度（gang scheduling）的基本约束：

> 一个需要 `p` 个同步 rank 的作业，要么同时拿到满足约束的 `p` 个设备，
> 要么继续等待。

它牺牲了一部分资源利用率，换取 collective 不会因缺失 rank 永久等待。
一个具体场景：集群有 8 张 GPU，队首作业需要 4 张，当前只有 3 张空闲。
FIFO 下队首不能启动；如果队列严格执行“不看后面”，后面的 2 卡作业也
被**队首阻塞**（head-of-line blocking），3 张 GPU 一直空闲。若允许
backfill，2 卡作业可以先跑，但只要它没结束，队首仍等不到第 4 张卡——
利用率和公平性由此成为需要同时观测的两个指标，而不是一个。

异步 Actor–Learner 系统可以采用不同的准入协议，但那是算法和数据面
共同定义的选择，不能由普通 FIFO 队列自动推导。

## FIFO、first-fit 与拓扑感知

常见的基线策略有：

- **FIFO**：按提交顺序检查队首作业；容易解释，但可能出现队首阻塞；
- **first-fit**：找到第一组满足数量和显存的设备；实现简单，常忽略拓扑；
- **topology-aware**：先尝试同节点/同机柜，再比较跨域通信成本；
- **公平队列**：按租户、优先级或累计 GPU 时间分配机会；
- **抢占式策略**：暂停低优先级作业释放资源，但依赖可恢复 checkpoint。

策略不能只看瞬时利用率。需要同时观察：

- queue wait 和 p95 queue wait；
- 作业 makespan 与 time-to-first-step；
- GPU 空闲但无法成组分配的碎片；
- 跨机柜 bytes、collective time 和链路热点；
- 低优先级租户是否长期饥饿。

队列中的“公平”也必须先定义。按 GPU 数公平、按显存公平、按作业数公平
和按完成时间公平会产生不同结果。

## Burn 入口不是作业调度器

`burn-train` 的 `MultiDevice` 和 `DistributedDataParallel` 假设调用者已经
提供了设备集合。`ExecutionStrategy::ddp` 接收本节点的 `Vec<Device>` 和
`DistributedConfig`；它不会从集群队列申请 GPU，也不会决定不同租户之间
如何共享节点。

因此外部控制面至少要负责：

1. 选择节点/GPU 并建立 rank 到设备的映射；
2. 注入训练进程所需的通信配置和数据 shard；
3. 在所有 rank 启动成功后才允许训练进入第一个 collective；
4. 记录作业版本、attempt、checkpoint 和资源归还；
5. 在进程退出或失败时释放租约，避免“幽灵 GPU”。

Burn 的 `ExecutionStrategy` 只处理第 3 步之后的训练策略的一部分。

## 本节小结

调度器的核心不在于把 GPU 放进一个数组，而在于定义资源向量、成组准入、
公平和恢复语义。模拟器用 FIFO 与拓扑感知策略对比这些协议；真实集群还
需要租户、配额、认证、镜像、节点健康和外部 launcher。
