# 实验：CPU 集群调度与故障模拟器

## 实验目标与边界

实验位于 `examples/ch09-cluster-simulator`，使用纯 Rust 标准库建立一个
离散事件模拟器。它有四张逻辑 GPU，分属两个 rack；作业需要成组 GPU、
显存、计算 step、梯度字节数和 checkpoint 周期。模拟时间是虚拟整数，
不是 `sleep` 得到的墙钟时间。

实验验证：

- FIFO 与 topology-aware placement 的资源选择；
- gang admission 不会把作业拆成部分 rank 启动；
- `alpha + beta * bytes` 的通信成本随消息大小增加；
- 跨机柜 pair 会增加虚拟通信成本和 bytes；
- failure 后释放资源，从最近 checkpoint 恢复并记录 replay steps；
- 相同输入产生相同 trace、makespan 和指标。

实验不验证真实 GPU kernel、NCCL、RDMA、网络拥塞、节点故障率、租户安全
或 Burn DDP。它把第 9 章的控制面协议抽象成可测试的 Rust 数据结构。

## 1. 集群与作业模型

`Gpu` 保存逻辑 id、node、rack 和显存容量；`Job` 保存 GPU 数、每步
compute、gradient bytes、checkpoint interval 和可选 failure step：

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

## 3. 通信成本

模拟器把一个 AllReduce 的 Reduce+Bcast 近似成 `2(p-1)` 轮。每个跨 rack
的 GPU pair 增加：

```text
cross_rack_bytes = gradient_bytes × cross_rack_pairs
```

网络模型使用 `alpha_us`、`beta_ns_per_byte` 和
`cross_rack_multiplier`。数值只用于比较策略；不能换算为实际 NIC 带宽。

## 4. 故障和 checkpoint replay

如果作业在 step 3 失败、checkpoint interval 是 2，模拟器会：

1. 释放本次 attempt 的 GPU；
2. 把最近 checkpoint 定位到 step 2；
3. 记录 `replayed_steps = 1`；
4. 以 `attempt + 1` 重新进入队列；
5. 重新成组准入并完成剩余 step。

一个失败只注入一次；`max_retries` 控制允许的恢复次数。真实系统还要
加入 checkpoint 写入失败、旧 attempt 消息、存储版本和租约过期。

## 5. 运行

在项目根目录运行：

```bash
cargo test -p ch09-cluster-simulator
cargo run -p ch09-cluster-simulator
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
2. 增加每个 rack 的链路容量，模拟同时运行作业的 oversubscription；
3. 为 Job 加入租户和 quota，测试资源租约释放；
4. 将单一 failure step 改为节点/机柜 failure domain；
5. 将 checkpoint report 扩展为版本、确认号和幂等提交；
6. 将虚拟 trace 转成按 job/rank/rack 聚合的 metrics。

这些扩展仍然是控制面模拟。要验证真实 Burn/CubeCL 设备路径，必须另行
记录硬件、backend、driver、通信库和启动环境。
