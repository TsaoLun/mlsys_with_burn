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
设计主题，而不是本版中已经由 `burn-train` 实现并被本章实验验证
的功能。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么保存 model parameters 不能等价于保存完整 optimizer state？

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【基础】局部 batch 大小不相等时，为什么简单平均各设备梯度可能不再等价于
   全局样本平均？

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

3. 【基础】梯度累积和增大 DataLoader batch size 分别改变了哪些内存、更新频率和
   metric 语义？

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

4. 【基础】同步 AllReduce 中一个 straggler 为什么会影响所有设备的 step 时间？

<details>
<summary>提示</summary>

见第 6 章集合通信节与 Flex CPU 无 collective 的边界。

</details>

5. 【进阶】`CollectiveTensor::resolve` 与异步 operation handle 的完成边界有什么
   关系？

<details>
<summary>提示</summary>

见第 6 章集合通信节与 Flex CPU 无 collective 的边界。

</details>

6. 【进阶】为什么参数服务器的异步更新需要处理 stale gradient 和版本协议？

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

7. 【进阶】为三个 pipeline stage 画出 1F1B micro-batch 时间线，计算 warm-up/
   cool-down bubble，并比较增加 micro-batch 与 activation recomputation
   对内存和算力的影响。

<details>
<summary>提示</summary>

见第 6 章 1F1B 配图与空泡占比公式。

</details>

8. 【进阶】用 $\theta\_{v+1}=U(\theta\_v,g(\theta\_{v-k}),s\_v)$ 设计 stale gradient
   拒绝、衰减和重放三种策略，列出各自的 checkpoint/幂等要求。

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


## Rust 与 API 题

1. 【基础】把实验中的 `Linear` 替换为包含两个 `Linear` 的自定义 `Module`，检查
   `GradientsParams` 是否能为两个参数建立正确映射。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【基础】为训练函数增加验证数据和 `model.valid()` 路径，比较训练/验证的
   autodiff 和 metric 输入。

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【进阶】把 `SgdConfig` 换成 `AdamConfig`，保存 optimizer record，再从 record
   恢复并比较后续一步的参数结果。

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【进阶】用 `ModuleLrScheduler` 为两个 `ParamGroup` 设置不同学习率，并记录每
   个 iteration 的有效学习率。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

5. 【进阶】为 `TrainStep::Output` 添加 lazy metric 字段，说明为什么输出类型需要
   满足 `ItemLazy + 'static`。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

6. 【进阶】为本机多设备策略构造一个带不同 shard 大小的数据集，验证局部梯度应按
   样本数加权，而不是无条件做设备平均。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>


## 源码题

1. 【进阶】阅读 `burn-train/src/learner/train_val.rs`，追踪 `TrainStep::step`、
   `TrainOutput::new` 和默认 `optimize` 的所有权流。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

2. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/single/epoch.rs`，
   记录 `lr_step`、backward、optimizer step 和 event 的实际调用顺序。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

3. 【进阶】阅读 `burn-train/src/learner/supervised/paradigm.rs`，比较
   `SingleDevice`、`MultiDevice` 和 DDP 的构造分支。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

4. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/multi/epoch.rs`，
   对比 `OptimMainDevice` 与 `OptimSharded` 的 gradient 容器。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

5. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/ddp/worker.rs` 和
   `burn-autodiff/src/distributed.rs`，画出 worker 到 gradient sync server
   的消息顺序。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

6. 【进阶】阅读 `burn-flex/src/ops/transaction.rs` 与
   `burn-cubecl/src/ops/distributed.rs`，说明为什么“有 DDP API”不等于
   “所有后端都能执行 DDP”。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>


## 性能与系统题

1. 【进阶】在固定输入上分别测 forward、backward、optimizer 和 `into_scalar`
   readback；报告同步边界、设备、形状和 dtype。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

2. 【挑战】增加一个可控计算量的 map/batcher，比较数据生产时间和训练计算时间；
   不要只报告总墙钟时间。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

3. 【挑战】设计单设备 reference 与本机多设备梯度聚合的数值对照，明确 Mean、
   Sum 和样本数权重。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

4. 【挑战】用 $\alpha+\beta l$ 模型比较 ring、tree 和分层 AllReduce 的适用
   条件，说明拓扑和链路 oversubscription 的影响。

<details>
<summary>提示</summary>

见第 6 章集合通信节与 Flex CPU 无 collective 的边界。

</details>

5. 【挑战】设计一个可恢复 epoch 协议，列出 model、optimizer、scheduler、seed、
   sampler position、shard assignment 和 code revision。

<details>
<summary>提示</summary>

回看第 6 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【挑战】设计参数服务器异步训练的版本/丢失 worker/热点参数测试，并说明哪些
   结果不能从同步 DDP 实验外推。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>


## 延伸阅读与固定源码入口

本书所用的 Burn 版本：

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
checkpoint 协议；它们的 API 或性能不能自动转化为 本书所用的 Burn 版本事实。

## 本章系统结论

1. 训练系统管理的是可恢复状态：参数、优化器、采样器、步数与检查点。
2. 数据并行的关键成本在梯度同步；可用 $\alpha+\beta$ 与 bubble/staleness 做数量级估计。
3. Burn 在源码中提供 `DistributedContext`、`all_reduce` 与 DDP strategy 入口；Flex CPU 没有 collective 实现。
4. CPU 上你观察到单设备 SGD 使 loss 下降并改变参数。
5. GPU 阅读时应对照：多 `Device`、collective 后端与拓扑带来的字节×延迟项。
6. 不能把单机 CPU 训练 loop 当成 NCCL/跨节点 DDP 已经验证。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-6-章)。
