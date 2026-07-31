# Batcher、DataLoader 与设备边界

## `Batcher` 把 item 变成模型输入

Dataset 的 item 通常是便于读取和变换的 Rust struct，而模型需要固定的
batch 表示。Burn 用 `Batcher<I, O>` 表达这个边界：

```rust
pub trait Batcher<I, O>: Send + Sync {
    fn batch(&self, items: Vec<I>, device: &Device) -> O;
}
```

`Batcher` 自己决定 padding、stack、标签编码、Tensor 构造和设备迁移。
它收到 `Device`，所以同一套 Dataset 可以在不同设备上用不同的 batch
目标执行。这个 trait 没有声明异步、预取或自动向量化；这些属于更上层
DataLoader 或用户实现的职责。

本章实验的 Batcher 为了突出顺序只保留 host 值，但仍把 `Device` 记录在
输出中：

```rust
{{#include ../../../examples/ch05-data-pipeline/src/lib.rs:batcher}}
```

固定 Burn 源码中的 `HousingBatcher`、MNIST batcher 等示例会在同一位置
构造 `Tensor`。因此“数据何时成为 Tensor”不是 Dataset trait 的要求，
而是应用选择的边界。

## DataLoader 的可观察接口

`DataLoader<O>` 提供：

- `iter()`：返回一个产生 `Result<O, DatasetError>` 的迭代器；
- `num_items()`：返回 item 数，不是 batch 数；
- `to_device()`：生成把 batch 交给另一设备的 loader；
- `slice(start, end)`：保留 batch 策略和顺序的子 loader。

迭代器还提供 `Progress { items_processed, items_total, unit }`。进度以
item 计数，即使当前输出是 batch，也不应把 `items_total` 误读成 batch
数。

## `DataLoaderBuilder`

固定快照中的 builder 主要配置以下参数：

```rust
let loader = DataLoaderBuilder::new(batcher)
    .batch_size(32)
    .shuffle(42)
    .num_workers(2)
    .set_device(Device::flex())
    .build(dataset);
```

- `batch_size` 使用 `FixBatchStrategy`；未设置时默认大小为 1；
- `shuffle(seed)` 在每次创建 iterator 时启用一次新的随机排列；
- `num_workers(None)` 或 `num_workers(0)` 使用当前线程读取；
- `num_workers(n > 0)` 使用 `MultiThreadDataLoader`；
- `set_device` 只决定 Batcher 收到的目标 Device，不改变 Dataset 的存储。

`FixBatchStrategy` 在数据量不是 batch size 的整数倍时，会在迭代末尾
强制发出一个不完整 batch。训练代码若要求固定形状，必须在 Batcher 中
padding，或明确丢弃/补齐最后一批；不能从 `batch_size` 配置本身推断
这个策略。

## 失败传播

单线程 loader 在 `get_many` 失败时返回 `Err`。多线程 worker 发生真实的
Dataset 错误时，固定源码通过消息把错误转成迭代器的 `Err`，而不是让
消费者无限等待。越界仍属于 Dataset 契约中的程序错误，不能用 `Result`
替代所有边界检查。

这也是 Rust 示例应该返回错误而不是在库代码里到处 `unwrap` 的原因：数据
文件损坏、权限问题或反序列化失败应当能带着上下文终止训练或交由调用者
决定重试。
