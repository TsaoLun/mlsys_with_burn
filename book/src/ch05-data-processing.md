# 第 5 章 数据处理系统

第 4 章讨论的是已经形成张量操作之后的 IR 与设备执行。本章把视线移到
设备之前：样本如何从存储到达 CPU，怎样经过变换和组 batch，又怎样交给
模型。数据管道不是训练循环外的一段脚本；它通过吞吐、缓冲和顺序约束与
模型执行耦合。

这是 OpenMLSys「数据处理系统」一章的对应物。产业里对应 PyTorch
DataLoader、`tf.data`、NVIDIA DALI、以及对象存储上的分片读取。GPU 农场
最常见的故障之一，不是算子不够快，而是设备在等数据。

## 本章问题

如何持续向加速器提供数据，而不让读取、变换、组 batch 或线程通信成为
瓶颈？多线程提高生产率时，怎样区分「样本没有丢失」和「样本仍按指定
顺序到达」？

## 学习目标

完成本章后，你应该能够：

1. 用 Load、Shuffle、Map、Batch 和 Send 描述数据处理路径；
2. 用生产、变换和消费三类速率定位瓶颈；
3. 阅读 `Dataset`、`MapperDataset`、`Batcher` 和 `DataLoader` 的职责边界；
4. 区分全内存数据集、惰性变换和按索引读取；
5. 解释固定 seed、每个 epoch 的 shuffle，以及有放回采样的语义；
6. 说明 `num_workers`、batch 分片、设备分派和消息到达顺序之间的关系；
7. 设计同时检查数据守恒、变换值、batch 形状和进度的实验；
8. 判断何时需要自定义重排、缓存或跨节点采样。

Burn 的多 worker DataLoader **不保证**全局样本顺序——这与某些框架的
Connector 保序语义不同，写分布式训练时必须分开处理。

## 先修知识

建议先完成第 2 章的 Tensor/Device。需要理解 Rust trait、`Iterator` 和
基本线程通信。

## 本章路线

先用 ETL 和速率模型定义问题，再进入实现：

```text
存储 / 内存
  → Dataset.get / get_many
  → map / selection / shuffle
  → DataLoader 的 batch strategy
  → Batcher(I, O) + Device
  → 模型训练或推理
```

第 4 章的 Fusion 优化的是张量操作；本章的 map 仍由普通 Rust 代码执行，
两套「流水」不要混成同一张编译图。

## 小节

1. [数据路径、语义与成本模型](ch05/01-data-pipeline-and-cost.md)
2. [Dataset 与惰性变换](ch05/02-dataset-abstractions.md)
3. [Batcher、DataLoader 与设备边界](ch05/03-batching-and-device.md)
4. [Shuffle、采样与数据划分](ch05/04-shuffle-sampling-split.md)
5. [多线程加载与保序性边界](ch05/05-multithread-and-order.md)
6. [存储、缓存与扩展路径](ch05/06-storage-and-scaling.md)
7. [实验：可复现数据管道](ch05/07-reproducible-pipeline-lab.md)
8. [练习、延伸阅读与来源](ch05/08-exercises-and-sources.md)

下一章把已经到达的 batch 放入训练循环，讨论优化器状态、检查点与跨设备
同步。
