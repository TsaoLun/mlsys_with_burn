# 练习、延伸阅读与来源

## 小结

数据系统把存储、索引顺序、样本变换、batch 组装和设备发送连接起来。
`Dataset` 是按索引取得 Rust item 的 `Send + Sync` trait；wrapper 可以把
map、selection、shuffle、partial 和 window 组合成惰性逻辑。`Batcher`
在明确的 Device 边界把 item 变成模型可消费的输出。

`DataLoaderBuilder` 可以配置 batch size、seed、worker 数和设备。固定
快照的多 worker 实现能并行读取、批处理、传播错误并复用 worker pool，
但 receiver 依据消息到达返回 batch，没有全局序号重排。因此必须分别测试
样本覆盖率、变换正确性、进度和顺序要求。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或固定源码路径），不提供完整答案；挑战题常涉及
`可选平台实验` 或开放设计，不在默认 CPU CI 中验证。


## 概念题

1. 【基础】为什么 `Dataset::get_many` 的“请求顺序”与多 worker loader 的“到达
   顺序”是两个不同的性质？

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

2. 【基础】用 $F$、$P$、$G$ 分别表示读取、变换和设备消费速率。一个有界队列
   能隐藏什么问题，不能解决什么问题？

<details>
<summary>提示</summary>

见第 5 章数据路径与背压模型节。

</details>

3. 【基础】`MapperDataset`、`SelectionDataset` 和 `SamplerDataset` 在数据复制、
   `len()` 和重复样本方面有什么不同？

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【基础】为什么固定 seed 可以复现第一轮单 worker shuffle，却不能单独证明
   多 worker 到达顺序？

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

5. 【进阶】最后一批不完整时，padding、丢弃和保留三种策略各有什么代价？

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【进阶】`split_dataloader` 的最后一个设备为什么可能拿到余数？这和跨节点
   分布式 sampler 还差哪些协议？

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>


## Rust 与实验题

1. 【基础】把 `PrepareSample` 改成带 `Result` 的自定义 Dataset，验证 worker
   能把真实读取错误传播给主 iterator。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

2. 【基础】增加一个 `SelectionDataset`，使用乱序且含重复的索引，检查
   `get_many` 的输出顺序。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【进阶】把 batch size 改成 5，断言最后一批大小为 2；再实现显式 padding
   Batcher，比较输出类型。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【进阶】使用 `WindowsDataset` 构造长度为 3 的窗口，说明窗口重叠带来的样本
   复用和有效长度变化。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】给每个样本增加可控的 sleep，观察多 worker 的到达顺序；测试只断言
   数据守恒，不要用一次调度结果建立永久契约。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

6. 【进阶】把 host Batcher 替换成 `Tensor` batch，明确读取、构造、传输和同步
   的计时边界。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

7. 【进阶】为一个分片数据集设计 logical index → shard/offset → decode 的索引
   协议，比较大/小分片对随机读取、元数据和缓存的影响。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

8. 【进阶】让一个 worker 注入可重试和不可重试错误，定义 batch retry 后的重复
   语义，并说明 checkpoint 应记录哪个提交边界。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>


## 源码题

1. 【进阶】阅读 `burn-dataset/src/dataset/base.rs`，找出越界和真实读取错误的
   不同契约。

<details>
<summary>提示</summary>

在固定 revision 源码中按章节末“源码入口”定位，勿跟 online main。

</details>

2. 【进阶】比较 `InMemDataset::get`、`MapperDataset::get` 和
   `SqliteDataset::get_many` 的复制/查询边界。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【进阶】沿 `DataLoaderBuilder::build` 追踪 `num_workers = 0` 和大于零时的
   两条实现路径。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

4. 【进阶】阅读 `MultiThreadDataLoader::initialize`，说明整体预 shuffle、分片
   和派生 RNG 的顺序。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

5. 【进阶】阅读 `MultiThreadsDataloaderIterator::next`，找出为什么没有全局序号
   重排。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

6. 【进阶】对照 `split_dataloader` 与 `PartialDataset::split_chunks`，解释“按
   设备切分”和“按 batch chunk 切分”的不同。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>


## 性能与系统题

1. 【进阶】用固定大小的文件样本替换内存样本，分别测冷 cache 和 warm cache。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【挑战】增大 map 计算量，比较单 worker 与多 worker；报告何时线程开销被
   计算覆盖。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

3. 【挑战】设计有界队列的生产/消费实验，记录队列深度、设备空闲时间和峰值内存。

<details>
<summary>提示</summary>

见第 5 章数据路径与背压模型节。

</details>

4. 【挑战】设计一个带全局序号的 reorder buffer，分析慢 worker 造成的
   head-of-line blocking。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

5. 【挑战】设计多设备数据划分的 epoch 协议，明确 seed、shard、drop-last、
   checkpoint 和故障恢复字段。

<details>
<summary>提示</summary>

回看第 5 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


## 延伸阅读

固定上游中的权威入口：

- `burn/crates/burn-dataset/src/dataset/base.rs`
- `burn/crates/burn-dataset/src/dataset/in_memory.rs`
- `burn/crates/burn-dataset/src/dataset/sqlite.rs`
- `burn/crates/burn-dataset/src/transform/mapper.rs`
- `burn/crates/burn-dataset/src/transform/selection.rs`
- `burn/crates/burn-dataset/src/transform/sampler.rs`
- `burn/crates/burn-core/src/data/dataloader/builder.rs`
- `burn/crates/burn-core/src/data/dataloader/batch.rs`
- `burn/crates/burn-core/src/data/dataloader/multithread.rs`
- `burn/crates/burn-core/src/data/dataloader/split.rs`
- `burn/examples/simple-regression/src/dataset.rs`
- `burn/examples/simple-regression/src/training.rs`

PyTorch DataLoader、TensorFlow `tf.data`、MindSpore MindData、DALI、Ray
Dataset 和 Arrow/Parquet 可以作为系统对照。比较时要记录版本、顺序
语义、进程/线程模型和是否包含设备传输，不能把它们的特性自动外推到
固定 Burn 快照。

## 来源与改编说明

本章改编并重组 OpenMLSys v1 的
`chapter_data_processing/`：

- `index.md`：保留数据模块的问题定义和学习目标，改为本书的 Rust/Burn
  阅读路线；
- `requirements.md`：保留 Load、Shuffle、Map、Batch、Send 与易用性、
  高效性、保序性三维框架；
- `program_model.md`：保留 Dataset 变换和自定义算子抽象，删除
  MindData、Spark 和长 Python 代码，改写为 `Dataset`、`Mapper`、
  `Batcher`；
- `performance.md`：保留 $F/P/G$ 成本模型、随机访问、异步生产消费和
  流水线/算子并行对照，删除 MindRecord/Unirecord 和厂商性能结论；
- `data_order.md`：保留保序问题与 Connector 的设计动机，明确其只是
  Burn 的对照概念，固定快照没有对应的序号等待实现；
- `extension.md`：保留 CPU 瓶颈、异构和分布式扩展的系统问题，改为
  边界与未来工作，不宣称 Burn 已提供通用异构数据预处理；
- `summary.md`：重写为本章的 Dataset/DataLoader 结论与验证边界。

OpenMLSys v2 固定快照的第 5 章仍是 TODO；本章依据 v1 中文文件。原章
引用的框架专用图片在固定 clone 中没有可复用的图像资源，本章没有复制
图片或 MindSpore/PyTorch/C++ 示例，结构关系使用原创文本图。

完整逐文件映射、固定 Burn 源码定位和不作出的能力承诺见
[`planning/chapter-sources/ch05.md`](https://github.com/TsaoLun/mlsys_with_burn/blob/main/planning/chapter-sources/ch05.md)。OpenMLSys 原作和本章改编正文采用
CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT OR Apache-2.0。
