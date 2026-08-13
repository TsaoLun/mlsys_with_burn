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
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么 `Dataset::get_many` 的“请求顺序”与多 worker loader 的“到达
   顺序”是两个不同的性质？

<details>
<summary>提示</summary>

「请求顺序」是 trait 契约：
[「Dataset 与惰性变换」](02-dataset-abstractions.md)写明 `get_many`
默认实现按给定索引逐个 `get`，保留顺序与重复项。「到达顺序」是
调度结果：[「多线程加载与保序性边界」](05-multithread-and-order.md)
里 receiver 收到 `Batch` 消息就立即返回。想清楚两者分别由谁保证、
在哪一层失效。

</details>

2. 【基础】用 $F$、$P$、$G$ 分别表示读取、变换和设备消费速率。一个有界队列
   能隐藏什么问题，不能解决什么问题？

<details>
<summary>提示</summary>

[「数据路径、语义与成本模型」](01-data-pipeline-and-cost.md)的
「吞吐率与队列」算过一笔账：长期缺口不受队列影响，容量 $Q$ 只把
设备空转推迟 $Q / (G - F)$；偶发抖动则可以被完全吸收。把「短期
抖动」与「长期瓶颈」分别代入这笔账，答案自然分成两半。

</details>

3. 【基础】`MapperDataset`、`SelectionDataset` 和 `SamplerDataset` 在数据复制、
   `len()` 和重复样本方面有什么不同？

<details>
<summary>提示</summary>

机制描述在[「Dataset 与惰性变换」](02-dataset-abstractions.md)的
「惰性组合」：读取时才 map、保存一组底层索引、按分布抽样；
[「Shuffle、采样与数据划分」](04-shuffle-sampling-split.md)还专门
对比了 Sampler 与 shuffle 的 `len()` 和放回语义。按「复制什么／
`len()` 是什么／会不会重复」三列各填一行，差异就清楚了。

</details>

4. 【基础】为什么固定 seed 可以复现第一轮单 worker shuffle，却不能单独证明
   多 worker 到达顺序？

<details>
<summary>提示</summary>

对照 `examples/ch05-data-pipeline` 的两个测试：
`fixed_seed_reproduces_single_worker_epoch` 敢断言完整 ID 序列，
`multi_worker_loader_conserves_items_and_receives_device` 却只在
排序后比较集合。原因见
[「多线程加载与保序性边界」](05-multithread-and-order.md)：seed 只
决定语义排列，到达顺序还叠加了线程调度这个自由变量。

</details>

5. 【进阶】最后一批不完整时，padding、丢弃和保留三种策略各有什么代价？

<details>
<summary>提示</summary>

[「Batcher、DataLoader 与设备边界」](03-batching-and-device.md)用
$1000 = 31 \times 32 + 8$ 说明 `FixBatchStrategy` 总会发出尾批。
从三个维度比较策略：样本是否被丢弃或重复计权、batch 形状是否稳定
（影响下游编译与缓存）、padding 需要的 mask 逻辑由谁承担。

</details>

6. 【进阶】`split_dataloader` 的最后一个设备为什么可能拿到余数？这和跨节点
   分布式 sampler 还差哪些协议？

<details>
<summary>提示</summary>

[「Shuffle、采样与数据划分」](04-shuffle-sampling-split.md)的
「多设备和 worker 的划分」写明：按连续范围切分，最后一个分片接收
余数；`burn/crates/burn-core/src/data/dataloader/split.rs` 可核对
算式。再对照[「存储、缓存与扩展路径」](06-storage-and-scaling.md)：
跨节点还缺全局 shuffle、epoch barrier、数据版本与故障恢复协议，
这些都不在本地切分的职责内。

</details>


## Rust 与实验题

1. 【基础】把 `PrepareSample` 改成带 `Result` 的自定义 Dataset，验证 worker
   能把真实读取错误传播给主 iterator。

<details>
<summary>提示</summary>

错误应发生在 `Dataset::get` 这一层——
[「Dataset 与惰性变换」](02-dataset-abstractions.md)写明 `get` 用
`Result` 携带读取错误。让自定义 Dataset 在某个索引返回 `Err`，仿照
`examples/ch05-data-pipeline` 的 `run_epoch` 用多 worker 收集，按
[「Batcher、DataLoader 与设备边界」](03-batching-and-device.md)的
「失败传播」契约断言主迭代器得到 `Err` 而不是无限等待。

</details>

2. 【基础】增加一个 `SelectionDataset`，使用乱序且含重复的索引，检查
   `get_many` 的输出顺序。

<details>
<summary>提示</summary>

[「Dataset 与惰性变换」](02-dataset-abstractions.md)给出两条可组合
的事实：`SelectionDataset` 保存一组可重排、可重复的底层索引，
`get_many` 默认实现保留请求顺序与重复索引；实现可对照
`burn/crates/burn-dataset/src/transform/selection.rs`。断言时除了
输出顺序，别忘了 `len()` 与重复项的 value 是否各自正确。

</details>

3. 【进阶】把 batch size 改成 5，断言最后一批大小为 2；再实现显式 padding
   Batcher，比较输出类型。

<details>
<summary>提示</summary>

12 个样本按 $12 = 2 \times 5 + 2$ 拆分：把
`examples/ch05-data-pipeline` 的 `run_epoch(5, 0, None)` 结果断言成
`batch_sizes == [5, 5, 2]` 即复现尾批，机制见
[「Batcher、DataLoader 与设备边界」](03-batching-and-device.md)。
padding 版 Batcher 的输出类型需要额外携带什么（有效长度或 mask），
正是要比较的差异。

</details>

4. 【进阶】使用 `WindowsDataset` 构造长度为 3 的窗口，说明窗口重叠带来的样本
   复用和有效长度变化。

<details>
<summary>提示</summary>

`WindowsDataset` 的定位见
[「Dataset 与惰性变换」](02-dataset-abstractions.md)的「惰性组合」。
把它套在 `examples/ch05-data-pipeline` 的 `prepared_dataset()`
（12 个样本）外，数一数 `len()` 变成多少、中间样本出现在几个窗口
里，再解释这对「一个 epoch 处理多少 item」意味着什么。

</details>

5. 【进阶】给每个样本增加可控的 sleep，观察多 worker 的到达顺序；测试只断言
   数据守恒，不要用一次调度结果建立永久契约。

<details>
<summary>提示</summary>

断言写法仿照 `examples/ch05-data-pipeline` 的
`multi_worker_loader_conserves_items_and_receives_device`：排序后
比较 ID 与 map 值，不断言展平顺序。sleep 的作用是放大
[「多线程加载与保序性边界」](05-multithread-and-order.md)里快慢
worker 交错的时间线：让某个 worker 显著变慢，观察它的批被后到的
批超越。

</details>

6. 【进阶】把 host Batcher 替换成 `Tensor` batch，明确读取、构造、传输和同步
   的计时边界。

<details>
<summary>提示</summary>

替换点是 `examples/ch05-data-pipeline` 的 `SampleBatcher::batch`；
[「实验：可复现数据管道」](07-reproducible-pipeline-lab.md)第 2 节
说明真实模型正是在此处构造 Tensor。计时边界按
[「存储、缓存与扩展路径」](06-storage-and-scaling.md)的「性能报告
的最低证据」逐项声明：读取、构造、传输、同步各自是否计入测量，
缺一项报告就不可比。

</details>

7. 【进阶】为一个分片数据集设计 logical index → shard/offset → decode 的索引
   协议，比较大/小分片对随机读取、元数据和缓存的影响。

<details>
<summary>提示</summary>

`examples/ch05-data-pipeline` 的 `deterministic_shards` 已给出
logical index → 半开区间的最小划分（见测试
`protocol_card_preserves_shards_and_epoch_commit`），把它扩展成
shard/offset/decode 三层。取舍分析沿
[「存储、缓存与扩展路径」](06-storage-and-scaling.md)的「文件索引
与随机读取的成本」：定位精度、打开文件与元数据开销、顺序读友好度
往往互相冲突。

</details>

8. 【进阶】让一个 worker 注入可重试和不可重试错误，定义 batch retry 后的重复
   语义，并说明 checkpoint 应记录哪个提交边界。

<details>
<summary>提示</summary>

[「多线程加载与保序性边界」](05-multithread-and-order.md)的
「失败、重试与 epoch 边界」列出协议字段，并区分「已取出／已计算／
已更新」三个提交点。可从 `examples/ch05-data-pipeline` 的
`PipelineError` 与 `epoch_commit` 出发：先给错误分类（可重试的
I/O 还是不可重试的格式错误），再回答重试后哪些样本会重复、由哪个
提交点负责去重。

</details>


## 源码题

1. 【进阶】阅读 `burn-dataset/src/dataset/base.rs`，找出越界和真实读取错误的
   不同契约。

<details>
<summary>提示</summary>

完整路径是 `burn/crates/burn-dataset/src/dataset/base.rs`。对照
[「Dataset 与惰性变换」](02-dataset-abstractions.md)的契约描述：
`Result` 保留 I/O 或反序列化错误，越界按 slice/Vec 风格 panic。
在源码里找出这两条约定各自的落点（签名、默认方法与注释），并想想
为什么把越界也塞进 `Result` 反而会掩盖程序错误。

</details>

2. 【进阶】比较 `InMemDataset::get`、`MapperDataset::get` 和
   `SqliteDataset::get_many` 的复制/查询边界。

<details>
<summary>提示</summary>

在 `burn/crates/burn-dataset/src/` 下读 `dataset/in_memory.rs`、
`transform/mapper.rs` 与 `dataset/sqlite.rs`。正文结论可先记住：
[「Dataset 与惰性变换」](02-dataset-abstractions.md)说
`InMemDataset` 的 `get` 返回 clone、Mapper 读取时才计算；
[「存储、缓存与扩展路径」](06-storage-and-scaling.md)说
`SqliteDataset::get_many` 一次查询并保持请求顺序。归纳每层复制了
什么、何时才真正访问存储。

</details>

3. 【进阶】沿 `DataLoaderBuilder::build` 追踪 `num_workers = 0` 和大于零时的
   两条实现路径。

<details>
<summary>提示</summary>

入口是 `burn/crates/burn-core/src/data/dataloader/builder.rs`；
[「Batcher、DataLoader 与设备边界」](03-batching-and-device.md)已给
结论：0 或未设置用当前线程，大于零构造 `MultiThreadDataLoader`。
追踪时留意 shuffle 的 RNG 与 `set_device` 的 Device 各自如何传进
两条路径——差异不止线程数一项。

</details>

4. 【进阶】阅读 `MultiThreadDataLoader::initialize`，说明整体预 shuffle、分片
   和派生 RNG 的顺序。

<details>
<summary>提示</summary>

文件是 `burn/crates/burn-core/src/data/dataloader/multithread.rs`；
把[「多线程加载与保序性边界」](05-multithread-and-order.md)开头的
路径图当核对清单：整体索引预 shuffle → `PartialDataset` 切分 →
每个 worker 一个 `BatchDataLoader`，其新迭代器再用派生 RNG 重排
自己的片段。关键是分清哪些发生在初始化、哪些发生在每轮迭代器
创建时。

</details>

5. 【进阶】阅读 `MultiThreadsDataloaderIterator::next`，找出为什么没有全局序号
   重排。

<details>
<summary>提示</summary>

文件同为 `burn/crates/burn-core/src/data/dataloader/multithread.rs`，
这次盯住消息类型：按
[「多线程加载与保序性边界」](05-multithread-and-order.md)的分析，
消息只携带 worker index、batch 与局部 progress。列出「若要重排还
缺什么」（全局样本序号、重排缓冲），缺口本身就是答案。

</details>

6. 【进阶】对照 `split_dataloader` 与 `PartialDataset::split_chunks`，解释“按
   设备切分”和“按 batch chunk 切分”的不同。

<details>
<summary>提示</summary>

读 `burn/crates/burn-core/src/data/dataloader/split.rs`，对照
[「Shuffle、采样与数据划分」](04-shuffle-sampling-split.md)的
「多设备和 worker 的划分」：前者切已建好的 loader（连续范围加
`to_device`，余数归最后一个设备），后者切 Dataset 索引（已知
batch size 时尽量按完整 batch 对齐）。从输入对象、余数落点和调用
时机三处找不同。

</details>


## 性能与系统题

1. 【进阶】用固定大小的文件样本替换内存样本，分别测冷 cache 和 warm cache。

<details>
<summary>提示</summary>

起点是 `examples/ch05-data-pipeline` 的 `measure_throughput`
（warm-up 一轮后才计时）。
[「存储、缓存与扩展路径」](06-storage-and-scaling.md)指出页缓存
命中会改变实际读取速率 $F$，所以先定义「冷」如何构造：换新文件、
清页缓存或把首轮单独计时，并在报告里写明缓存状态与样本大小。

</details>

2. 【挑战】增大 map 计算量，比较单 worker 与多 worker；报告何时线程开销被
   计算覆盖。

<details>
<summary>提示</summary>

把 `examples/ch05-data-pipeline` 的 `PrepareSample::map` 换成可控
计算量，用 `measure_throughput` 在同一 batch size 下对比
`num_workers` 为 0 与 2。判断框架在
[「数据路径、语义与成本模型」](01-data-pipeline-and-cost.md)：
worker 自带线程通信与调度开销，只有 map 服务时间足够大时并行才
净赚。找交叉点要给多组测量，不能只报单点。

</details>

3. 【挑战】设计有界队列的生产/消费实验，记录队列深度、设备空闲时间和峰值内存。

<details>
<summary>提示</summary>

[「数据路径、语义与成本模型」](01-data-pipeline-and-cost.md)的
背压模型给出可检验的预测：长期缺口下设备约在 $Q / (G - F)$ 后开始
空转。记录项可仿照 `examples/ch05-data-pipeline` 的
`FlowInvariant`（fetched/produced/consumed 加 queue_peak），把队列
深度、设备空闲和峰值内存对齐到同一条时间轴上验证这笔账。

</details>

4. 【挑战】设计一个带全局序号的 reorder buffer，分析慢 worker 造成的
   head-of-line blocking。

<details>
<summary>提示</summary>

[「多线程加载与保序性边界」](05-multithread-and-order.md)把「带
序列约束的专用调度器」列为保序的第三种方向，并点名它必须处理
head-of-line blocking。设计时回答三件事：全局序号由谁分配；某个
序号未到达时后续 batch 最多积压多少（缓冲内存上界）；慢 worker
的延迟如何直接变成消费者等待。数据守恒测试对这些量完全无感，
需要另设指标。

</details>

5. 【挑战】设计多设备数据划分的 epoch 协议，明确 seed、shard、drop-last、
   checkpoint 和故障恢复字段。

<details>
<summary>提示</summary>

字段骨架在[「多线程加载与保序性边界」](05-multithread-and-order.md)
的「失败、重试与 epoch 边界」：数据版本、shard 标识、epoch、
attempt、RNG 派生与错误分类；`examples/ch05-data-pipeline` 的
`deterministic_shards` 与 `epoch_commit` 演示了 shard 划分和提交
判定的最小形态。逐字段说明由谁写入、恢复时如何避免重复或遗漏，
并注意 drop-last 与余数分片的交互。

</details>


## 延伸阅读

数据管道系统的论文见附录[参考文献](../references.md#第-5-章-数据处理系统)。
本书固定版本源码中的权威入口：

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

## 本章系统结论

1. 数据管道要同时满足正确语义、吞吐与可复现，而不是“越快越好”。
2. Dataset/Mapper/Batcher/DataLoader 分层：惰性变换、组 batch、设备投放各有边界。
3. CPU 上你验证了数据守恒、固定 seed 与多 worker 下的顺序/进度语义。
4. GPU 阅读线索：`to_device` 之后的 HtoD、pinned memory、与训练 step 的流水线重叠。
5. 不能把一次 CPU loader 测量当成存储系统或跨节点 sampler 的吞吐结论。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-5-章)。
