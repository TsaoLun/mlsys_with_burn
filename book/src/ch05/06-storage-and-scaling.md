# 存储、缓存与扩展路径

## 存储格式决定读取成本

按索引 Dataset 把“如何找到第 `i` 个样本”留给存储实现。不同实现的
成本模型不同：

- `InMemDataset` 直接从 vector clone，读取延迟低但占用完整数据集内存；
- `InMemDataset::from_csv` / `from_json_rows` 先解析全部文件，再开始
  训练，不适合把它误当作流式 reader；
- `SqliteDataset` 用 split 表和 `row_id = index + 1` 做按行读取；
- `SqliteDataset::get_many` 用一次查询取得多个请求索引，并保持请求
  顺序和重复项；
- vision、audio、NLP 和 Hugging Face source 是带 feature 的具体数据源，
  不是 `Dataset` trait 必须提供的能力。

真正的大规模管道还要考虑文件分片、索引大小、压缩解码、页缓存、远端
对象存储和数据格式演进。仅仅把 Dataset 放进线程池不会自动解决 I/O
瓶颈。

## 文件索引与随机读取的成本

一个按索引读取的 Dataset 至少需要回答“索引如何映射到物理数据”：

```text
logical index
    → shard id / row id
    → byte offset 或数据库 key
    → compressed block
    → decode / validate
    → item
```

索引表越大，初始化和内存成本越高；分片越小，随机访问定位更精确但会
增加打开文件、元数据和远端请求；分片越大，顺序读取更友好但随机样本
可能需要读取更多无关数据。压缩还会把 CPU 解码时间加入 $P$，页缓存命中
则会改变实际 $F$。因此 `Dataset::get` 的平均耗时不能单独代表文件格式
的性能。

固定 Burn 的 `InMemDataset`、CSV/JSON 构造器和 SQLite Dataset 能说明三种
边界：完整预加载、解析后驻留内存，以及按 row id 查询。它们没有统一的
跨格式分片、索引、压缩、远端重试或版本协议。生产系统需要在 Dataset
外部定义 manifest、数据版本、校验和、shard assignment 与重试策略，再
把单个 shard 包装成 `Dataset`。

## 缓存与预取的边界

固定版本的 `Dataset`/`DataLoader` 核心 API 没有一个通用的
`prefetch(n)` 或跨 Dataset 的缓存协议。多 worker 通道会产生并发生产和
有限背压，但不能据此声称存在可配置的磁盘预取、Pinned Host Memory、
零拷贝设备传输或数据变换自动融合。

可以按成本添加不同层次的缓存：

1. 在 Dataset 内缓存解码结果；
2. 预先生成更适合随机读取的本地格式；
3. 在 worker 和 Batcher 之间放置有界队列；
4. 在 Send 边界使用后端支持的异步传输。

每一层都要重新核对内存上限、失效策略、随机性和错误重试。第 4 章的
CubeCL/Fusion 优化不能直接作用于任意字符串或图像解码 map。

## CPU 扩展和异构处理

当 $P$ 是瓶颈时，可以增加 worker、减少昂贵的重复变换、预计算特征，
或者把明确的 Tensor 变换移到设备。但异构路径需要回答设备拷贝、队列
容量、算子覆盖率和同步边界；固定 Burn 数据模块源码并没有一个通用的
“把 Dataset map 自动卸载到 GPU”的接口。

当单机无法满足生产率时，常见的横向扩展是离线预处理后共享结果、按
worker/节点切分输入，或使用专门的分布式数据系统。它们还会引入全局
shuffle、数据版本、重复样本、故障恢复和跨框架序列化问题。本章只建立
这些成本模型；Burn 的多设备 `split_dataloader` 不应被写成完整的跨节点
分布式数据处理方案。

## 性能报告的最低证据

一个可复查的吞吐报告至少应包含：

```text
dataset size / sample size / storage format
batch size / num_workers / shuffle seed
warm-up 与计时轮数
wall-clock 边界 / CPU 与设备信息
是否包含解码、batch、拷贝和同步
```

本章实验的 `measure_throughput` 只用内存整数样本，并在计时前 warm-up
一次。它适合比较同一进程中配置的相对差异，不足以证明文件 I/O、GPU
传输或生产训练任务的性能结论。
