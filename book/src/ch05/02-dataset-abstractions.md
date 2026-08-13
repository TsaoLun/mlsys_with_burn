# Dataset 与惰性变换

## `Dataset` 是按索引读取的契约

Burn 数据模块的核心 trait 是：

```rust,ignore
pub trait Dataset<I, E = DatasetError>: Send + Sync
where
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E>;
    fn len(&self) -> usize;
}
```

实际签名还提供了默认的 `get_many`、`is_empty` 和 `iter`。这里有三个
值得注意的系统含义：

1. `I` 是用户定义的样本类型，不要求一开始就是 Tensor；
2. `Send + Sync` 让同一个 Dataset 可以安全地被多线程 DataLoader 共享；
3. `Result` 保留 I/O 或反序列化错误，越界则按 slice/Vec 风格 panic。

`get_many` 的默认实现按给定索引逐个调用 `get`，并保留请求顺序和重复
索引。具体存储后端可以覆盖它，以一次查询取得多个样本；这正是通用 trait
和存储实现之间的性能扩展点。

## 内存数据集与迭代器

`InMemDataset<I>` 内部保存 `Vec<I>`。它的 `get` 返回 item 的 clone，
`len` 返回 vector 长度；`from_csv` 和 `from_json_rows` 会把输入文件
完整读入内存。因此它适合小数据、预加载或教学实验，不等同于流式读取。

Burn 自己的 `DatasetIterator` 只保存当前索引和数据集引用，逐项调用
`get`，并把每个读取错误作为迭代器 item 返回。Rust 的借用关系保证
迭代器不会在数据集引用失效后继续运行，但不会自动给文件读取添加缓存
或预取。

## 惰性组合

`burn-dataset` 提供多种 wrapper：

- `MapperDataset<D, M, I>` 在读取时调用 `Mapper::map`，不提前物化输出；
- `SelectionDataset` 保存一组底层索引，可以重排、重复或选择子集；
- `ShuffledDataset` 是对全部索引做随机排列的 Selection wrapper；
- `PartialDataset` 表示一个连续的半开区间；
- `WindowsDataset` 把相邻样本组成重叠窗口；
- `ComposedDataset` 把多个数据集串接成一个更大的逻辑数据集。

这些类型通过组合表达流水线，而不是把每一次变换都复制成一个新文件。
代价是一次 `get` 可能穿过多层 wrapper；是否应该预计算，需要根据读取
成本、内存容量和重复 epoch 次数决定。

本章实验使用同样的设计，但把样本值保持为整数，便于测试直接检查：

```rust,ignore
{{#include ../../../examples/ch05-data-pipeline/src/lib.rs:dataset}}
```

`MapperDataset` 的 map 是纯 Rust 逐样本逻辑。它可以表达清洗或特征变换，
但本版没有把任意 `Mapper` 自动编译为 CubeCL Kernel；若变换本身是
Tensor 计算，需要由用户在 Batcher 或其他明确的 Tensor 边界中实现。

## 错误和所有权

默认的 `DatasetError` 是对线程安全错误的类型擦除包装。存储后端可以使用
更具体的错误类型，例如 SQLite Dataset 使用自己的错误枚举，再在需要时
转换为通用错误。

Dataset 的 `get` 返回 item 的所有权，方便 worker 把 item 送入 batch；
`InMemDataset` 因此要求 item 可 clone。自定义 Dataset 要同时考虑：

- item 是否能跨线程发送；
- 同一个 Dataset 是否能被并发调用；
- `get` 是否会共享可变缓存；
- 错误是否包含足够的文件、索引和 split 信息。

trait 的 `Send + Sync` 只约束并发访问的类型安全，不会替用户保证底层
数据库连接、随机数或自定义缓存的逻辑正确性。
