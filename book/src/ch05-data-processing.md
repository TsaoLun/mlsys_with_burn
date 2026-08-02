# 第 5 章 数据处理系统

第 4 章讨论的是已经形成 Tensor 操作之后的 IR、融合和设备执行。本章把
视线移到设备之前：样本如何从存储到达 CPU，怎样经过变换和 batching，
又怎样安全、可复现地交给模型。数据管道不是“训练循环外的一段脚本”；
它和模型执行通过吞吐率、缓冲和顺序约束耦合在一起。

## 本章问题

如何持续向加速器提供数据，而不让读取、变换、组 batch 或线程通信成为
训练瓶颈？当多线程提高生产率时，怎样区分“样本没有丢失”和“样本仍按
指定顺序到达”？

## 学习目标

完成本章后，你应该能够：

1. 用 Load、Shuffle、Map、Batch 和 Send 描述数据处理路径；
2. 用生产速率、变换速率和消费速率定位数据管道的瓶颈；
3. 阅读 Burn 固定快照中的 `Dataset`、`MapperDataset`、`Batcher` 和
   `DataLoader` 边界；
4. 区分 `InMemDataset` 的全内存模型、惰性变换和基于 SQLite 的按索引读取；
5. 解释固定 seed、每个 epoch 的 shuffle，以及 `SamplerDataset` 的替换语义；
6. 说明 `num_workers`、batch 分片、设备分派和多线程消息到达顺序之间的关系；
7. 设计同时检查数据守恒、变换值、batch 形状和进度的实验；
8. 诚实地分析何时需要自定义顺序重排、缓存或更大范围的分布式数据系统。

## 先修知识

建议先完成第 2 章的 Tensor/Device 和第 4 章的执行边界。需要理解 Rust
trait、`Iterator`、`Send`/`Sync` 和基本线程通信；不要求先学习 SQLite。

## 本章路线

我们先用框架无关的 ETL 和速率模型定义问题，再逐层进入 Burn：

```text
存储 / 内存
  → Dataset.get / get_many
  → map / selection / shuffle
  → DataLoader 的 batch strategy
  → Batcher(I, O) + Device
  → 模型训练或推理
```

第 4 章的 Fusion 计划优化的是 Tensor 操作；本章的数据变换仍由 Dataset
和 Batcher 的 Rust 代码执行。固定快照没有把一般 Dataset map 自动 lower
成 CubeCL Kernel，因此不能把两种“流水”混成同一套编译图。

## 小节

1. [数据路径、语义与成本模型](ch05/01-data-pipeline-and-cost.md)
2. [Dataset 与惰性变换](ch05/02-dataset-abstractions.md)
3. [Batcher、DataLoader 与设备边界](ch05/03-batching-and-device.md)
4. [Shuffle、采样与数据划分](ch05/04-shuffle-sampling-split.md)
5. [多线程加载与保序性边界](ch05/05-multithread-and-order.md)
6. [存储、缓存与扩展路径](ch05/06-storage-and-scaling.md)
7. [实验：可复现数据管道](ch05/07-reproducible-pipeline-lab.md)
8. [练习、延伸阅读与来源](ch05/08-exercises-and-sources.md)

## 证据状态

- `CPU 可运行验证`：Dataset、Mapper、Batcher、DataLoader、固定 seed
  和多 worker 数据守恒；
- `固定源码核验`：内存 Dataset、SQLite、采样、worker 和 Device 边界；
- `框架无关模型/协议模拟`：文件索引、背压、retry、epoch commit 和
  reorder buffer；
- `需要 CUDA/NCCL/网络/旧 revision 的可选扩展`：真实存储吞吐、pinned
  memory、跨节点 sampler 和设备数据通道；
- `明确未覆盖`：把数据守恒或一次 CPU 测量描述成全局保序/真实吞吐。

对应 `F/P/G`、分片和提交协议见[核心主题比较卡](comparison-cards.md#第-5-章数据处理)。

