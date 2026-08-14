# 第 6 章 训练系统

第 5 章解决样本怎样到达模型。本章继续向后：一个 batch 到达之后，系统
怎样组织前向、反向、参数更新、验证、检查点和跨设备同步。训练系统不是
把 `backward` 和 `optimizer.step` 放进更大的 `for` 循环就结束；它还要
管理学习率、梯度累积、指标、状态恢复，以及不同设备之间的等待关系。

这是 OpenMLSys「分布式训练」在**数据面**上的对应物（控制面见第 9 章）。
产业里你会碰到 PyTorch DDP、Horovod、FSDP / ZeRO、Megatron 的张量并行
与流水线并行。本章先把单设备状态机讲清，再用通信成本和切分不变量解释
这些策略；默认实验在单设备上跑通一次真正的 SGD。

## 本章问题

一个训练系统如何组织前向、反向、优化、检查点和跨设备通信，并在规模
增长时保持正确的状态与可接受的效率？数据并行、张量并行、流水线并行
和参数切分各自切开的是什么？

## 学习目标

完成本章后，你应该能够：

1. 用 loss、gradient、optimizer state 和同步边界描述一次训练迭代；
2. 阅读 `TrainStep`、`Learner`、优化器和检查点之间的职责分工；
3. 解释 Rust 中 module ownership 如何让「参数更新后得到新 module」成为
   显式数据流；
4. 区分学习率调度、梯度累积、检查点和 sampler 状态；
5. 区分本机多设备、DDP 的梯度 collective，以及 TP / PP / ZeRO 切的对象；
6. 用 $\alpha+\beta$ 模型估计 AllReduce 和 1F1B 空泡的数量级；
7. 通过实验确认一次训练循环确实降低 loss 并改变参数；
8. 指出作业启动、弹性成员和跨节点 checkpoint 属于第 9 章控制面。

## 先修知识

建议先完成第 2 章的 Tensor / Module / autodiff，以及第 5 章的 DataLoader。
不要求本机已有 CUDA 或 NCCL。

## 本章路线

先从训练状态和成本模型开始，再进入 API：

![训练闭环：Dataset/DataLoader 供给 batch，经 forward/loss 与 backward/autodiff tape 得到 GradientsParams，再进入 optimizer、checkpoint 与 validation](img/ch06-training-loop.svg)

与第 9 章的分工见控制面/数据面对照图
（[`ch06-ch09-control-data-planes.svg`](img/ch06-ch09-control-data-planes.svg)）：
本章落在训练数据面。

单设备是最小闭环；本机多设备在同一进程内分摊；DDP 要求后端 collective。
张量并行切开的是隐藏维，流水线切开的是层，ZeRO 切开的是优化器状态——
不要把它们都叫做「并行」。

## 小节

1. [训练状态、迭代与成本模型](ch06/01-training-state-and-cost.md)
2. [前向、反向与自定义训练循环](ch06/02-forward-backward-loop.md)
3. [burn-train 的 Learner 与训练装配](ch06/03-burn-train-orchestration.md)
4. [优化器、学习率与检查点](ch06/04-optimizer-and-checkpoint.md)
5. [本机多设备与数据并行](ch06/05-local-data-parallel.md)
6. [集合通信、DDP 与并行策略](ch06/06-collective-and-ddp.md)
7. [实验：CPU 线性回归训练循环](ch06/07-training-loop-lab.md)
8. [练习、延伸阅读与来源](ch06/08-exercises-and-sources.md)

训练完成后，第 7 章讨论如何保存并服务这些参数；第 9 章把集合通信放进
机柜与作业队列。

示例位于 `examples/ch06-training-loop`。它使用 Flex CPU，不下载数据。
