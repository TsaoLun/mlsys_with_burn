# 练习、延伸阅读与来源

## 小结

训练系统把 batch、forward、backward、optimizer、metric、checkpoint 和
设备策略连接为一条有状态的执行路径。`TrainStep` 定义应用模型如何从
输入产生梯度，`Learner` 组合 model、optimizer 和 scheduler，
`SupervisedTraining` 再装配数据、事件、指标、检查点和执行策略。

固定 Burn 快照提供单设备、本机多设备和 DDP 的不同入口。`MultiDevice`
在一个进程内切分 loader 并聚合多设备梯度；DDP 为每个本地设备启动 worker，
通过 backend `DistributedOps` 进行梯度 collective，并要求用户在每个节点
启动且保持配置一致。Flex CPU 没有 collective 实现，所以本章实验只验证
CPU 单设备训练循环。

参数服务器、流水线并行、跨节点故障恢复、集群调度和网络性能仍是系统
设计主题，而不是本固定快照中已经由 `burn-train` 实现并被本章实验验证
的功能。

## 概念题

1. 为什么保存 model parameters 不能等价于保存完整 optimizer state？
2. 局部 batch 大小不相等时，为什么简单平均各设备梯度可能不再等价于
   全局样本平均？
3. 梯度累积和增大 DataLoader batch size 分别改变了哪些内存、更新频率和
   metric 语义？
4. 同步 AllReduce 中一个 straggler 为什么会影响所有设备的 step 时间？
5. `CollectiveTensor::resolve` 与异步 operation handle 的完成边界有什么
   关系？
6. 为什么参数服务器的异步更新需要处理 stale gradient 和版本协议？

## Rust 与 API 题

1. 把实验中的 `Linear` 替换为包含两个 `Linear` 的自定义 `Module`，检查
   `GradientsParams` 是否能为两个参数建立正确映射。
2. 为训练函数增加验证数据和 `model.valid()` 路径，比较训练/验证的
   autodiff 和 metric 输入。
3. 把 `SgdConfig` 换成 `AdamConfig`，保存 optimizer record，再从 record
   恢复并比较后续一步的参数结果。
4. 用 `ModuleLrScheduler` 为两个 `ParamGroup` 设置不同学习率，并记录每
   个 iteration 的有效学习率。
5. 为 `TrainStep::Output` 添加 lazy metric 字段，说明为什么输出类型需要
   满足 `ItemLazy + 'static`。
6. 为本机多设备策略构造一个带不同 shard 大小的数据集，验证局部梯度应按
   样本数加权，而不是无条件做设备平均。

## 源码题

1. 阅读 `burn-train/src/learner/train_val.rs`，追踪 `TrainStep::step`、
   `TrainOutput::new` 和默认 `optimize` 的所有权流。
2. 阅读 `burn-train/src/learner/supervised/strategies/single/epoch.rs`，
   记录 `lr_step`、backward、optimizer step 和 event 的实际调用顺序。
3. 阅读 `burn-train/src/learner/supervised/paradigm.rs`，比较
   `SingleDevice`、`MultiDevice` 和 DDP 的构造分支。
4. 阅读 `burn-train/src/learner/supervised/strategies/multi/epoch.rs`，
   对比 `OptimMainDevice` 与 `OptimSharded` 的 gradient 容器。
5. 阅读 `burn-train/src/learner/supervised/strategies/ddp/worker.rs` 和
   `burn-autodiff/src/distributed.rs`，画出 worker 到 gradient sync server
   的消息顺序。
6. 阅读 `burn-flex/src/ops/transaction.rs` 与
   `burn-cubecl/src/ops/distributed.rs`，说明为什么“有 DDP API”不等于
   “所有后端都能执行 DDP”。

## 性能与系统题

1. 在固定输入上分别测 forward、backward、optimizer 和 `into_scalar`
   readback；报告同步边界、设备、形状和 dtype。
2. 增加一个可控计算量的 map/batcher，比较数据生产时间和训练计算时间；
   不要只报告总墙钟时间。
3. 设计单设备 reference 与本机多设备梯度聚合的数值对照，明确 Mean、
   Sum 和样本数权重。
4. 用 $\alpha+\beta l$ 模型比较 ring、tree 和分层 AllReduce 的适用
   条件，说明拓扑和链路 oversubscription 的影响。
5. 设计一个可恢复 epoch 协议，列出 model、optimizer、scheduler、seed、
   sampler position、shard assignment 和 code revision。
6. 设计参数服务器异步训练的版本/丢失 worker/热点参数测试，并说明哪些
   结果不能从同步 DDP 实验外推。

## 延伸阅读与固定源码入口

Burn 固定快照：

- `burn/crates/burn-train/src/learner/train_val.rs`
- `burn/crates/burn-train/src/learner/base.rs`
- `burn/crates/burn-train/src/learner/supervised/paradigm.rs`
- `burn/crates/burn-train/src/learner/supervised/strategies/single/epoch.rs`
- `burn/crates/burn-train/src/learner/supervised/strategies/multi/epoch.rs`
- `burn/crates/burn-train/src/learner/supervised/strategies/ddp/README.md`
- `burn/crates/burn-train/src/learner/supervised/strategies/ddp/worker.rs`
- `burn/crates/burn-optim/src/optim/module/module_optimizer.rs`
- `burn/crates/burn-optim/src/lr_scheduler/module_lr_scheduler.rs`
- `burn/crates/burn-tensor/src/tensor/distributed.rs`
- `burn/crates/burn-backend/src/backend/distributed/server.rs`
- `burn/crates/burn-cubecl/src/ops/distributed.rs`
- `burn/crates/burn-flex/src/ops/transaction.rs`
- `burn/examples/custom-training-loop/src/lib.rs`
- `burn/examples/text-classification/examples/ag-news-train.rs`

OpenMLSys v1：

- `openmlsys/v1/zh_chapters/chapter_distributed_training/overview.md`
- `openmlsys/v1/zh_chapters/chapter_distributed_training/methods.md`
- `openmlsys/v1/zh_chapters/chapter_distributed_training/collective.md`
- `openmlsys/v1/zh_chapters/chapter_distributed_training/parameter_servers.md`
- `openmlsys/v1/zh_chapters/chapter_distributed_training/cluster.md`

可以把 PyTorch DDP、Horovod、NCCL、gPipe、ZeRO 和参数服务器论文作为
对照阅读，但比较时必须记录版本、通信后端、进程模型、梯度归一化和
checkpoint 协议；它们的 API 或性能不能自动转化为 Burn 固定快照事实。

## 来源与改编说明

本章改编并重组 OpenMLSys v1 的
`chapter_distributed_training/`：

- `index.md`：保留动机和并行方法地图，改为本书第 6 章的训练状态路线；
- `overview.md`：保留算力、内存、分而治之和 time-to-accuracy 问题，删除
  固定历史硬件数字与原图编号；
- `methods.md`：保留数据/模型/混合/流水线并行的语义，具体 Burn 只实现
  已核验的本机多设备和 DDP API；
- `collective.md`：保留集合通信算子、$\alpha+\beta l$ 成本和梯度平均，
  以 `DistributedContext`、backend `all_reduce` 和同步边界重写；
- `parameter_servers.md`：保留同步/异步、straggler、热点和副本一致性，
  明确固定 Burn 没有对应的 `burn-train` strategy；
- `cluster.md`：保留通信层次和带宽瓶颈，将调度、遥测、容错后移第 9 章；
- `summary.md`：重写为经过固定源码核验的能力清单。

没有复制 OpenMLSys 的 MindSpore、TensorFlow、PyTorch、Gloo、NCCL 代码或
章节图片；跨系统代码只在解释接口边界时以文字提及。完整逐文件核验、
Burn 版本定位和未承诺能力见
`planning/chapter-sources/ch06.md`。OpenMLSys 改编正文采用
CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT OR Apache-2.0。
