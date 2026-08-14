# 实验：CPU 集群调度与故障模拟器

## 你会学到什么

示例在 `examples/ch09-cluster-simulator`：用纯 Rust 标准库做离散事件
模拟——四张逻辑 GPU 分属两个 rack；作业需要成组 GPU、显存、计算
step、梯度字节数和 checkpoint 周期。时间是虚拟整数，不是 `sleep`
得到的墙钟时间。

你会观察到：

- FIFO 与 topology-aware placement 如何选资源；
- gang admission 不会把作业拆成部分 rank 启动；
- 放置按“同节点 → 同机柜 → 跨机柜”收紧，三档通信域成本逐级跳变；
- `alpha + beta * bytes` 的通信成本随消息变大；
- 跨机柜 pair 会推高虚拟通信成本；
- 失败后释放资源，从最近 checkpoint 恢复并记录 replay steps；
- 相同输入得到相同 trace、makespan 和指标。

模拟器建模控制面协议，不测量真实 GPU、NCCL 或网络拥塞。真机集群对应
Slurm / K8s 设备插件，见本章分层说明。

## 1. 集群与作业模型

`Gpu` 保存逻辑 id、node、rack 和显存容量；`Job` 保存 GPU 数、每步
compute、每步 gradient bytes、checkpoint interval 和可选 failure step：

```rust,ignore
{{#include ../../../examples/ch09-cluster-simulator/src/lib.rs:cluster_model}}
```

生产系统还需要 driver、NIC、链路、租约、租户和健康状态；本实验只保留
能解释 placement 与恢复的最小字段。

## 2. 模拟器接口

`simulate` 接收 cluster、job 列表和 `SimulationConfig`，返回结构化报告：

```rust,ignore
{{#include ../../../examples/ch09-cluster-simulator/src/lib.rs:simulator_api}}
```

`PlacementPolicy::Fifo` 选择按 id 排序的第一组可用 GPU；
`TopologyAware` 优先在同一 rack 内凑齐成组资源。两种策略都执行 gang
admission：队首作业无法完整放置时，不会先启动它的部分 rank。

## 3. 通信成本：三个域，不是两个

模拟器把一个同步 step 中的一次 AllReduce 的 Reduce+Bcast 近似成
`2(p-1)` 轮，并把放置结果中的每个 GPU pair 分进三个通信域：

| 域 | 判定 | 成本处理 |
|---|---|---|
| 节点内 | `(rack, node)` 相同 | 只付 `alpha + beta * bytes` 基线 |
| 同机柜跨节点 | rack 相同、node 不同 | 字节项按 `cross_node_multiplier` 放大 |
| 跨机柜 | rack 不同 | 字节项按 `cross_rack_multiplier` 放大，并计入 `cross_rack_bytes` |

节点的身份是 `(rack, node)` 二元组——node 编号在每个机柜内重复，
正如真实集群里服务器编号只在本机柜内有意义。三个域的字节量分别为：

```text
per_step_cross_node_bytes = gradient_bytes_per_step × cross_node_pairs
per_step_cross_rack_bytes = gradient_bytes_per_step × cross_rack_pairs
```

`work_time` 对每个执行 step 都加入该成本；失败前已经执行的 step、
checkpoint replay 重做的 step 也分别累计到 `collective_time_us` 与
`cross_rack_bytes`。因此增加 `steps` 或发生 replay 都会增加虚拟通信
时间。数值只用于比较策略；不能换算为实际 NIC 带宽。

## 4. 故障和 checkpoint replay

如果作业在 step 3 失败、checkpoint interval 是 2，模拟器会：

1. 释放本次 attempt 的 GPU；
2. 把最近 checkpoint 定位到 step 2；
3. 记录 `replayed_steps = 1`；
4. 把失败前已执行的 3 个 step 的 collective 成本计入该 job；
5. 以 `attempt + 1` 重新进入队列；
6. 重新成组准入，把 step 2–4 的 compute、collective 和 checkpoint 再执行
   一次。

一个失败只注入一次；`max_retries` 控制允许的恢复次数。真实系统还要
加入 checkpoint 写入失败、旧 attempt 消息、存储版本和租约过期。

## 5. 运行

在项目根目录运行：

```bash
cargo test -p ch09-cluster-simulator --locked
cargo run -p ch09-cluster-simulator --locked
```

主程序分别运行 FIFO 和 topology-aware 策略，输出类似：

```text
policy=fifo jobs=3 completed=3 makespan_ms=...
queue_wait_ms=... p95_queue_wait_ms=... cross_rack_bytes=...
collective_ms=... retries=... peak_allocated_gpus=...
job=1 attempts=2 retries=1 queue_wait_ms=...
checkpoint_replay_steps=1 placements=[[0, 1], [0, 1]]
```

具体虚拟时间取决于输入参数；教学断言应关注资源、trace、bytes、重试和
完成状态，不应把这些数字当成机器性能。

## 6. 观察与扩展

建议按以下顺序改动：

1. 把 `Fifo` 改为允许 backfill，观察队首作业和公平性的变化；
2. 用 `Cluster::uniform_interleaved(2, 2, 2, ...)` 造一个两级节点的
   集群，先预测同一作业在三类通信域上的成本顺序，再跑模拟核对；
3. 增加每个 rack 的链路容量，模拟同时运行作业的 oversubscription；
4. 为 Job 加入租户和 quota，测试资源租约释放；
5. 将单一 failure step 改为节点/机柜 failure domain；
6. 将 checkpoint report 扩展为版本、确认号和幂等提交；
7. 将虚拟 trace 转成按 job/rank/rack 聚合的 metrics。

这些扩展仍然是控制面模拟。要验证真实 Burn/CubeCL 设备路径，必须另行
记录硬件、backend、driver、通信库和启动环境。
