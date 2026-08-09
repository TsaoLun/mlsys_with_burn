# 多线程加载与保序性边界

## Burn 的 worker 路径

当 `num_workers > 0` 时，固定版本构造 `MultiThreadDataLoader`。从源码可以
追踪出如下路径：

```text
Dataset
  → （可选）整体索引预 shuffle
  → PartialDataset::split / split_chunks
  → 每个 worker 一个 BatchDataLoader
  → worker 线程执行 get + batch
  → mpsc::SyncSender<Message<O>>
  → 主迭代器接收 Batch / Done / Error
```

worker pool 在第一次迭代时懒初始化，并在同一个 loader 的后续 epoch 中
复用。固定源码的消息通道容量是 100 个消息槽位；这提供了有限的背压，
但不等于“最多预取 100 个完整 batch”的跨版本 API 承诺。

如果启用了 shuffle，初始化阶段先对整个数据集的索引做一次重排，再按
worker 切分；每个 worker 内部的 `BatchDataLoader` 还会在新 iterator
创建时使用派生 RNG 重排自己的片段。这个实现细节解释了为什么多 worker
shuffle 不应被描述为一个简单的“每轮全局排列”。

## 到达顺序不是全局保序

`MultiThreadsDataloaderIterator` 从 receiver 收消息，收到 `Batch` 就
立即返回；消息中携带的是 worker index、batch 和局部 progress，没有全局
样本序号或重排缓冲。于是：

```text
worker 0: [0, 1] ──慢──────────────→
worker 1: [2, 3] ──快→ [4, 5] ────→
消费者:   [2, 3], [4, 5], [0, 1]
```

这是一个允许的到达顺序。源码测试主要用集合比较多 worker 和单 worker
是否覆盖相同 item，并没有把 worker 消息到达顺序作为契约。因而本书对
固定 Burn 快照的准确结论是：

- 多 worker 路径可以验证数据守恒和错误传播；
- 不能把 batch 的到达顺序当成全局输入顺序；
- 一次运行恰好按序，不代表调度器提供了稳定保序保证。

若应用确实需要保序，有三种方向：使用 `num_workers = 0`；在消息中附加
全局序号并在消费者侧重排；或设计带序列约束的专用调度器。第三种方案
需要处理慢 worker 造成的 head-of-line blocking，不能只加一个锁。

## 与 OpenMLSys Connector 的比较

OpenMLSys v1 用 MindSpore Connector 说明生产者/消费者编号、round-robin
分发和按序等待如何共同实现保序。这个概念很适合解释“并行不自动等于
乱序可接受”，但不能直接迁移成 Burn 能力。固定 Burn 快照的
`MultiThreadDataLoader` 使用消息接收顺序，没有对应的 Connector 序号等待
算法。

因此第 5 章把“保序性”作为系统设计问题和实验观察量，而不是把原书的
MindSpore 实现翻译成 Rust 后宣称 Burn 已经实现。

## 生命周期和错误

worker 线程在 pool 中持续存活；提前丢弃一个 iterator 不会自动杀死整个
pool，下一轮仍可继续读取。worker 发生真实读取错误时会发送 `Error`，
主 iterator 返回一个 `DatasetError`。这让所有权和线程生命周期保持在
DataLoader 内部，调用者只需要处理 iterator 的 `Result`。

但 worker 线程安全不意味着 map 函数无副作用。需要可复现的随机增强时，
应为每个 worker/epoch 明确分配 RNG 状态；不要依赖线程启动顺序或全局
可变随机源。

## 失败、重试与 epoch 边界

`DatasetError` 能把 worker 的读取错误传回主 iterator，但它不自动定义
“重试后是否会重复样本”。一个可恢复的数据协议至少需要携带：

- 数据版本和 shard/worker 标识；
- logical sample index 或 batch sequence；
- 当前 epoch、attempt 次数和 RNG 派生信息；
- 读取失败是可重试的 I/O 错误，还是不可重试的格式/校验错误。

如果按 batch 重试，已经成功交给设备但尚未写入进度的 batch 可能再次执行；
如果按 sample 重试，batch 组装和顺序可能变化。训练系统必须决定 checkpoint
记录的是“已取出”“已计算”还是“已更新”，不能用 DataLoader 的
`items_processed` 自动代替 optimizer step 进度。固定 Burn DataLoader
提供迭代和错误传播边界，但不提供上述跨阶段提交协议。
