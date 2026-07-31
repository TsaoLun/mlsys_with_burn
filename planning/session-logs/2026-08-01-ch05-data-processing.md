# 2026-08-01：第 5 章数据处理系统

## 会话目标

继续 M3 系统篇，先审计项目和固定上游快照，再完成第 5 章
OpenMLSys v1 与 Burn `burn-dataset`/DataLoader 的映射、正文、CPU 实验
和交接状态。

## 开始状态

- 分支：`main`，开始时工作区干净；
- 当前目标：`planning/STATUS.md` 中的第 5 章数据处理映射；
- 固定快照：Burn `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`，
  OpenMLSys `9c289782ccbb165ac8ad7c960ecffc12942a5560`；
- OpenMLSys v2 第 5 章在固定快照中仍为 TODO，正文依据 v1 中文章节。

## 源码核验

### OpenMLSys

逐文件审查 v1 `zh_chapters/chapter_data_processing/`：

- `index.md`：数据模块目标与易用性/高效性/保序性三维框架；
- `requirements.md`：Load、Shuffle、Map、Batch、Send 组件；
- `program_model.md`：Dataset 变换、LINQ/RDD 对照和自定义算子；
- `performance.md`：随机读取、$F/P/G$ 速率模型、异步生产消费、
  流水线/算子并行和数据图优化；
- `data_order.md`：MindSpore Connector 的保序编号和等待约束；
- `extension.md`：异构与分布式预处理扩展；
- `summary.md`：章节总结和阅读入口。

保留框架无关的系统问题，删除或改写 MindData、MindRecord、Ascend、DALI、
Ray、Python/C++ 长代码和固定厂商性能结论。固定 clone 没有原 Markdown
引用图片的可复用图像文件，本章使用原创文本图。

### Burn

核验固定 Burn 源码：

- `burn-dataset::Dataset` 要求 `Send + Sync`，`get`/`get_many`/`len`/
  `iter` 分开表达索引读取、批量读取和遍历；
- `InMemDataset` 保存 `Vec`，CSV/JSON 构造会把输入载入内存；
- `MapperDataset`、`SelectionDataset`、`ShuffledDataset`、
  `SamplerDataset`、`PartialDataset` 和 `WindowsDataset` 是组合式 wrapper；
- `Batcher<I,O>` 在 `Device` 边界把 item 变为 batch；
- `DataLoaderBuilder` 的 batch size、shuffle seed、worker 数和 Device
  决定 loader 构造；
- 0 worker 走 `BatchDataLoader`，大于 0 走
  `MultiThreadDataLoader`；
- 多 worker 通过 `PartialDataset::split_chunks`、后台线程和 bounded
  mpsc 消息返回 batch；消息包含 worker index，但没有全局样本序号重排；
- `split_dataloader` 是本地连续分片和设备分派，不是完整跨节点分布式
  sampler；
- SQLite Dataset 使用连接池按 `row_id` 读取，`get_many` 保持请求顺序和
  重复。

上述定位与完整不承诺清单记录于 `planning/chapter-sources/ch05.md`；
多 worker 的范围决定记录为 `planning/DECISIONS.md` 的 D008。

## 实现

新增和修改：

- `examples/ch05-data-pipeline/Cargo.toml`
- `examples/ch05-data-pipeline/src/lib.rs`
- `examples/ch05-data-pipeline/src/main.rs`
- 根 `Cargo.toml` workspace member
- `Cargo.lock`（增加固定 Burn revision 下的 `burn-dataset` 解析）
- `book/src/ch05-data-processing.md`
- `book/src/ch05/01-data-pipeline-and-cost.md`
- `book/src/ch05/02-dataset-abstractions.md`
- `book/src/ch05/03-batching-and-device.md`
- `book/src/ch05/04-shuffle-sampling-split.md`
- `book/src/ch05/05-multithread-and-order.md`
- `book/src/ch05/06-storage-and-scaling.md`
- `book/src/ch05/07-reproducible-pipeline-lab.md`
- `book/src/ch05/08-exercises-and-sources.md`
- `book/src/SUMMARY.md`
- `planning/chapter-sources/ch05.md`
- `planning/DECISIONS.md` D008
- `planning/STATUS.md`

实验使用 12 个内存整数样本：

```text
RawSample → MapperDataset(2 * value + 1) → SampleBatcher
```

覆盖单 worker batching、固定 seed 第一轮复现、同一 loader epoch RNG
前进、多 worker 样本守恒/变换值/进度/Device 传递，以及 warm-up 后的
粗粒度 items/s 观察。测试不对线程到达顺序作脆弱断言；主程序打印该顺序
供读者观察。

## 验证

通过：

```text
cargo test -p ch05-data-pipeline
cargo clippy -p ch05-data-pipeline --all-targets -- -D warnings
cargo run -p ch05-data-pipeline
python3 tools/check_upstreams.py
python3 tools/check_upstreams.py --check-local
mdbook build book
cargo fmt --all --check
make check
make check-local-sources
git diff --check
```

`ch05-data-pipeline` 测试 5 项全部通过；完整 `make check` 的 workspace
测试和 Clippy 也通过。Cargo 仍提示用户目录同时存在 `~/.cargo/config`
和 `~/.cargo/config.toml`，未修改用户级配置。

## 偏差与边界

- 没有复制 OpenMLSys 的 MindData Connector 实现；Burn 固定快照的多 worker
  接收顺序不作为全局保序承诺；
- 没有把内存整数实验的 items/s 外推到磁盘、GPU 或分布式训练；
- 没有增加本地 path dependency、`[patch]`、生成 mdBook 输出或上游修改；
- 本次本机 Intel macOS 完整检查通过；`tracel-llvm` bundler 在干净平台
  环境的资产差异仍保留为已知风险。

## 交接

第 5 章已具备正文、来源映射、练习、实验和验证记录。下一次从
OpenMLSys v1 `chapter_distributed_training/` 开始第 6 章映射，先核验
`burn-train`、optimizer、`split_dataloader` 和固定快照可验证的多设备/
通信边界。
