# 故障、检查点与可观测性

## 故障域与落后者

集群故障不只有“节点挂了”：

- GPU kernel 错误、显存不足或设备 reset；
- CPU、PCIe、NIC、ToR 或 Spine 链路异常；
- 进程 OOM、通信超时或 checkpoint 存储不可用；
- 设备仍在线但速度显著下降，成为 straggler（落后者）。

同步 collective 对 straggler 很敏感：所有 rank 都必须到达完成边界，
一个慢 rank 就可能把整个 step 的等待时间拉长。调度器需要把故障域
记录在 placement 中，训练协议需要把“谁已经完成哪一步”记录在
checkpoint 和事件日志中。

## Checkpoint 不是一个文件写入动作

可恢复 checkpoint 至少包括：

```text
checkpoint = {
  job_id, attempt, step,
  model_record, optimizer_record,
  sampler/data-shard state,
  RNG state, code/config revision,
  commit version, acknowledged flag
}
```

保存过程应区分：

1. 写临时对象；
2. 校验 shape、dtype、版本和完整性；
3. 原子提交一个可见版本；
4. 由控制面确认该版本可以恢复；
5. 清理旧版本，但保留回滚窗口。

失败重试还需要幂等性：同一个 `job_id + attempt + step` 的重复提交不能
让 optimizer 更新两次。旧 attempt 的迟到消息也不能覆盖新 attempt 的
checkpoint。

如果训练在 step 5 失败、最近 checkpoint 是 step 4，那么恢复会重做
step 5；这部分叫 checkpoint replay。重做量是恢复成本的一部分，不应
被隐藏在“作业重新启动”这句话中。

checkpoint 间隔本身是一个可优化的成本权衡。设每步耗时 $t$、写一次
checkpoint 花费 $W$、平均故障间隔为 $M$：间隔 $C$ 越长，写入开销
$W/C$ 越小，但故障后的期望重做量越大。经典结果（Young 近似）给出
较优间隔 $C^* \approx \sqrt{2WM}$。代入 $W = 60\ \text{s}$、
$M = 4\ \text{h}$：
$C^* \approx \sqrt{2 \times 60 \times 14400} \approx 1315\ \text{s} \approx 22\ \text{min}$。
这个公式解释了为什么“每一步都存”和“从来只存最后”都不是好策略；它也再次
说明 checkpoint 间隔是成本模型决策，而不是配置文件的随手一项。

## 容错闭环里 Burn 站在哪一格

把“发现故障→保住进度→恢复训练”写成闭环，每个环节的归属就清楚了：

| 容错闭环的环节 | Burn 快照提供什么 | 缺口由谁补 |
|---|---|---|
| 保住进度：保存 learner 训练状态 | `LearningCheckpointer`、文件与异步 checkpointer | —— |
| 发现故障：心跳与 failure detector | 无；通信 server 生命周期只随 Rust 对象创建/销毁 | 集群控制面（本章模拟器的故障 trace 是它的协议模型） |
| 响应故障：自动重启、弹性加入、抢占、重新注入 rank | 无 | 控制面调度器 |
| 跨节点 checkpoint 提交共识 | 单机语义 | 存储协议层 |
| artifact 治理：租约、版本清理、访问控制 | 无 | 对象存储与多租户系统 |

## 观测什么，在哪里观测

建议将一个 step 拆成结构化事件：

```text
queue_wait
admission
device_init
compute
collective
collective_wait
checkpoint_write
checkpoint_commit
retry
release
```

每个事件至少带：

- `job_id`、`tenant_id`、`attempt`、`step`；
- rank、device、node、rack 和 network domain；
- start/end virtual time 或 monotonic timestamp；
- bytes、dtype、collective kind、checkpoint version；
- error kind、retry reason 和 lease version。

指标可以按四层聚合：

1. **作业层**：queue wait、makespan、成功率、重试次数；
2. **训练层**：compute、collective、straggler wait、samples/step；
3. **设备层**：显存、stream、kernel、同步和 memory cleanup；
4. **链路层**：跨机柜 bytes、链路占用、丢包/超时和热点。

CubeCL 的 logging、timestamp profiler、stream scheduler 和 memory usage
可以支持设备运行时观察；`burn-train` 的 metrics 可以记录训练事件。
它们不能自动聚合成集群级 tracing，也不能从本地 kernel 时间推导跨租户
网络瓶颈。

## 模拟器中的故障状态机

本章实验使用一个确定性故障：

```text
running(step)
    │ failure
    ▼
release resources → choose latest checkpoint
    │
    ▼
enqueue retry(attempt + 1, resume_step)
    │
    ▼
admit as a gang → replay missing steps → complete
```

该状态机验证的是协议不变量：资源归还、重试次数、checkpoint replay
和最终完成。它没有模拟真实 GPU 错误率、网络包丢失或分布式存储吞吐。

## 本节小结

可恢复集群需要故障域、版本化 checkpoint、幂等重试和跨层遥测。哪个
环节由谁提供，本节的容错闭环表已经给出；其中协议不变量（资源归还、
重试上限、checkpoint replay）是实验里可以亲手验证的部分。
