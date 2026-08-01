# 9.5 多租户、配额与资源碎片

## 总容量不是可用容量

集群有 8 张 GPU，不代表一个需要 8 张 GPU 的作业现在就能启动。至少
存在三类“容量”：

- **总容量**：物理 GPU、显存、节点和链路的总和；
- **可分配容量**：健康、未被租约占用且满足显存约束的资源；
- **可成组容量**：在同一时间、同一故障和通信域约束下能同时分配的资源。

资源碎片（resource fragmentation）发生在总量足够、但空闲资源被切散
或显存/拓扑约束不匹配时。例如四张空闲 GPU 分布在两个机柜，每张都能
运行单卡任务，却无法满足一个要求同一机柜四卡的作业。

## 多租户的最小协议

多租户（multi-tenancy）不只是给作业加一个 `user_id`。控制面至少要
维护：

```text
tenant → quota → lease → placement → usage → release
```

配额可以按 GPU 数、GPU 时间、显存、跨机柜字节或并发作业数计算。租约
需要有过期和回收语义，否则进程崩溃后资源会永久显示为占用。

隔离也分层：

- **资源隔离**：一个租户不能占用另一个租户的 GPU、显存和共享链路预算；
- **进程隔离**：环境变量、文件、网络端点和 checkpoint 权限不能串租户；
- **性能隔离**：一个租户的跨机柜 burst 不应无上限挤占其他租户；
- **故障隔离**：一个作业的错误、OOM 或重试不能拖垮整个调度器。

固定 Burn/CubeCL 源码中的 memory pool 和 `MemoryUsage` 是设备 runtime
级别的内存管理；它们不等于租户配额。设备 backend 也不能从一个
`DeviceId` 推出租户身份、租约期限或跨作业网络预算。

## 打包、保留与抢占

调度器常见的选择包括：

- **bin packing**：尽量填满节点，减少碎片；
- **保留大块资源**：为大作业保留连续/同域 GPU，牺牲短期利用率；
- **backfilling**：大作业等待时运行不会阻塞它的短作业，但不能破坏
  已承诺的启动时间；
- **抢占**：暂停低优先级作业，要求它能安全保存和恢复 checkpoint。

这些策略都需要明确“资源何时算释放”。只发出 `kill` 不代表 GPU、显存、
stream、文件锁和网络租约已经回收。控制面应等待 worker 的确认，或通过
attempt/lease 版本防止旧进程继续提交梯度和 checkpoint。

## 一个可解释的资源记录

每次分配可以记录：

```text
allocation = {
  job_id, tenant_id, attempt,
  gpu_ids, node_ids, rack_ids,
  memory_reserved, network_domain,
  lease_version, expires_at
}
```

训练进程提交事件时必须带上 `job_id`、`attempt` 和 `lease_version`。如果
旧 attempt 在恢复后继续发送消息，控制面或数据面可以拒绝过期版本，而
不是把旧梯度混入新作业。

## 与 Burn 的连接

Burn `ModuleRecord`、`LearningCheckpointer` 和 CubeCL memory pool 解决
不同层次的状态问题：

- `ModuleRecord`：模型参数/模块状态如何序列化；
- learner checkpoint：训练状态如何保存和恢复；
- runtime memory pool：本地 buffer 如何分配、复用和清理；
- 集群租约：哪个作业在什么时间拥有哪组设备。

前几项都不能替代最后一项。把 checkpoint 文件保存成功写成“集群资源
已经安全释放”，会把 artifact 状态和控制面状态混为一谈。

## 本节小结

多租户系统要同时管理配额、租约、性能隔离、故障隔离和资源碎片。GPU
总数、Burn 的 device abstraction 和 runtime memory usage 都只是局部
事实；真正的租户调度需要外部控制面协议。
