# 练习、延伸阅读与来源

## 小结

训练系统把 batch、forward、backward、optimizer、metric、checkpoint 和
设备策略连接为一条有状态的执行路径。`TrainStep` 定义应用模型如何从
输入产生梯度，`Learner` 组合 model、optimizer 和 scheduler，
`SupervisedTraining` 再装配数据、事件、指标、检查点和执行策略。

固定 Burn 快照提供单设备、本机多设备和 DDP 的不同入口。`MultiDevice`
在一个进程内切分 loader 并聚合多设备梯度；DDP 为每个本地设备启动 worker，
通过 backend `DistributedOps` 进行梯度 collective，并要求用户在每个节点
启动且保持配置一致。DDP 入口之外还差哪几层才是完整的分布式训练系统，见
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)中
「DDP 的范围与参与责任」的分工表。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么保存 model parameters 不能等价于保存完整 optimizer state？

<details>
<summary>提示</summary>

Adam 的一阶/二阶矩也是需要恢复的训练状态。见
[「优化器、学习率与检查点」](04-optimizer-and-checkpoint.md)，
想一想只恢复参数后，优化器第一步会与原轨迹差在哪里。

</details>

2. 【基础】局部 batch 大小不相等时，为什么简单平均各设备梯度可能不再等价于
   全局样本平均？

<details>
<summary>提示</summary>

把每个设备的梯度写成「本地样本损失和 ÷ 本地样本数」，再对设备做无权
平均，比较它与全局样本平均的差别。见
[「本机多设备与数据并行」](05-local-data-parallel.md)的加权讨论。

</details>

3. 【基础】梯度累积和增大 DataLoader batch size 分别改变了哪些内存、更新频率和
   metric 语义？

<details>
<summary>提示</summary>

梯度累积不增加单步激活内存但降低 optimizer step 频率；增大 batch 则
相反。对照[「训练状态、迭代与成本模型」](01-training-state-and-cost.md)
的内存预算，再想 metric 按 step 还是按样本归一。

</details>

4. 【基础】同步 AllReduce 中一个 straggler 为什么会影响所有设备的 step 时间？

<details>
<summary>提示</summary>

集合通信要求全体参与者共同到达；step 时间近似
「最慢设备的计算时间 + 通信时间」。见
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的
AllReduce 语义段。

</details>

5. 【进阶】`CollectiveTensor::resolve` 与异步 operation handle 的完成边界有什么
   关系？

<details>
<summary>提示</summary>

两者都把「提交操作」与「结果可读」拆成不同时间点。对照
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的完成语义段
与第 4 章[「内存、Stream 与异步执行」](../ch04/06-memory-streams-execution.md)。

</details>

6. 【进阶】为什么参数服务器的异步更新需要处理 stale gradient 和版本协议？

<details>
<summary>提示</summary>

异步 push/pull 意味着梯度可能基于旧参数版本计算。见
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)末尾的版本
协议四问。

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

三种策略分别要求 server 记住什么才能在重启后不重复应用同一梯度？从
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的版本协议
四问逐条推。

</details>


## Rust 与 API 题

1. 【基础】把实验中的 `Linear` 替换为包含两个 `Linear` 的自定义 `Module`，检查
   `GradientsParams` 是否能为两个参数建立正确映射。

<details>
<summary>提示</summary>

在 `examples/ch06-training-loop` 里改模型定义；`#[derive(Module)]` 的
参数注册规则见第 2 章
[「Module、参数与模型状态」](../ch02/03-module-and-state.md)。

</details>

2. 【基础】为训练函数增加验证数据和 `model.valid()` 路径，比较训练/验证的
   autodiff 和 metric 输入。

<details>
<summary>提示</summary>

`valid()` 返回不带 autodiff 的 inner model。见
[「burn-train 的 Learner 与训练装配」](03-burn-train-orchestration.md)
中训练/验证路径的区分。

</details>

3. 【进阶】把 `SgdConfig` 换成 `AdamConfig`，保存 optimizer record，再从 record
   恢复并比较后续一步的参数结果。

<details>
<summary>提示</summary>

optimizer record 的保存/恢复模式见
[「优化器、学习率与检查点」](04-optimizer-and-checkpoint.md)；
record 往返的代码样板可参考 `examples/ch07-record-roundtrip`。

</details>

4. 【进阶】用 `ModuleLrScheduler` 为两个 `ParamGroup` 设置不同学习率，并记录每
   个 iteration 的有效学习率。

<details>
<summary>提示</summary>

入口在本章源码清单的
`burn-optim/src/lr_scheduler/module_lr_scheduler.rs`；scheduler 语义见
[「优化器、学习率与检查点」](04-optimizer-and-checkpoint.md)。

</details>

5. 【进阶】为 `TrainStep::Output` 添加 lazy metric 字段，说明为什么输出类型需要
   满足 `ItemLazy + 'static`。

<details>
<summary>提示</summary>

metric 在事件处理线程上被惰性物化，输出因此要能跨线程存活。见
[「burn-train 的 Learner 与训练装配」](03-burn-train-orchestration.md)
的事件与 metric 段。

</details>

6. 【进阶】为本机多设备策略构造一个带不同 shard 大小的数据集，验证局部梯度应按
   样本数加权，而不是无条件做设备平均。

<details>
<summary>提示</summary>

与概念题 2 是同一个问题的实现版。见
[「本机多设备与数据并行」](05-local-data-parallel.md)的 Mean/Sum
与样本数讨论。

</details>


## 源码题

1. 【进阶】阅读 `burn-train/src/learner/train_val.rs`，追踪 `TrainStep::step`、
   `TrainOutput::new` 和默认 `optimize` 的所有权流。

<details>
<summary>提示</summary>

关注 model 是被移动还是被借用、在哪一步被替换成新 model。所有权与
训练循环的关系见
[「前向、反向与自定义训练循环」](02-forward-backward-loop.md)。

</details>

2. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/single/epoch.rs`，
   记录 `lr_step`、backward、optimizer step 和 event 的实际调用顺序。

<details>
<summary>提示</summary>

先猜一个顺序再对照源码验证；scheduler 先于还是后于 optimizer step
会改变有效学习率的含义，见
[「优化器、学习率与检查点」](04-optimizer-and-checkpoint.md)。

</details>

3. 【进阶】阅读 `burn-train/src/learner/supervised/paradigm.rs`，比较
   `SingleDevice`、`MultiDevice` 和 DDP 的构造分支。

<details>
<summary>提示</summary>

三个分支的差异集中在 loader 如何切分、梯度在哪里聚合。对照
[「本机多设备与数据并行」](05-local-data-parallel.md)与
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)。

</details>

4. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/multi/epoch.rs`，
   对比 `OptimMainDevice` 与 `OptimSharded` 的 gradient 容器。

<details>
<summary>提示</summary>

两种模式决定梯度在主设备集中更新还是分片更新。背景见
[「本机多设备与数据并行」](05-local-data-parallel.md)。

</details>

5. 【进阶】阅读 `burn-train/src/learner/supervised/strategies/ddp/worker.rs` 和
   `burn-autodiff/src/distributed.rs`，画出 worker 到 gradient sync server
   的消息顺序。

<details>
<summary>提示</summary>

按[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的
DDP 分层图逐层对应：参数注册、梯度提交、`all_reduce`、
`sync_collective` 各发生在哪个文件。

</details>

6. 【进阶】阅读 `burn-flex/src/ops/transaction.rs` 与
   `burn-cubecl/src/ops/distributed.rs`，说明为什么“有 DDP API”不等于
   “所有后端都能执行 DDP”。

<details>
<summary>提示</summary>

Flex 的注释明确写出不支持 collective。对照
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的
「各后端的实现现状」一节，区分「API 层存在」与「backend 有实现」。

</details>


## 性能与系统题

1. 【进阶】在固定输入上分别测 forward、backward、optimizer 和 `into_scalar`
   readback；报告同步边界、设备、形状和 dtype。

<details>
<summary>提示</summary>

`into_scalar` 是同步读回点，计时器夹住的段落决定你测到什么。异步
提交与读取的区分见第 4 章
[「内存、Stream 与异步执行」](../ch04/06-memory-streams-execution.md)。

</details>

2. 【挑战】增加一个可控计算量的 map/batcher，比较数据生产时间和训练计算时间；
   不要只报告总墙钟时间。

<details>
<summary>提示</summary>

在 `examples/ch05-data-pipeline` 的 map 里加入可调延迟，分别计时
生产与消费两侧；生产/消费预算模型见第 5 章
[「数据路径、语义与成本模型」](../ch05/01-data-pipeline-and-cost.md)。

</details>

3. 【挑战】设计单设备 reference 与本机多设备梯度聚合的数值对照，明确 Mean、
   Sum 和样本数权重。

<details>
<summary>提示</summary>

用相同 seed 与确定性初始化，让单设备跑完整 batch、多设备各跑一半，
比较聚合梯度。加权语义见
[「本机多设备与数据并行」](05-local-data-parallel.md)。

</details>

4. 【挑战】用 $\alpha+\beta l$ 模型比较 ring、tree 和分层 AllReduce 的适用
   条件，说明拓扑和链路 oversubscription 的影响。

<details>
<summary>提示</summary>

ring 的每设备流量近似 $2S$ 但延迟项按 $2(p-1)$ 步增长；小消息、多
设备时树形占优。推导见
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)的通信成本段。

</details>

5. 【挑战】设计一个可恢复 epoch 协议，列出 model、optimizer、scheduler、seed、
   sampler position、shard assignment 和 code revision。

<details>
<summary>提示</summary>

对每一项都问同一个问题：丢了它，恢复后的哪一步会与原轨迹不一致？
检查点状态清单见
[「优化器、学习率与检查点」](04-optimizer-and-checkpoint.md)。

</details>

6. 【挑战】设计参数服务器异步训练的版本/丢失 worker/热点参数测试，并说明哪些
   结果不能从同步 DDP 实验外推。

<details>
<summary>提示</summary>

同步 DDP 每步形成共同完成点，没有版本差；异步测试要围绕
[「集合通信、DDP 与能力边界」](06-collective-and-ddp.md)版本协议
四问设计断言。

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

Horovod、GPipe、ZeRO、参数服务器等系统的论文集中在附录
[参考文献](../references.md#第-6-章-训练系统)。对照阅读时记录版本、
通信后端、进程模型、梯度归一化和 checkpoint 协议，比较才有意义。

## 本章系统结论

1. 训练系统管理的是可恢复状态：参数、优化器、采样器、步数与检查点。
2. 数据并行的关键成本在梯度同步；可用 $\alpha+\beta$ 与 bubble/staleness 做数量级估计。
3. Burn 在源码中提供 `DistributedContext`、`all_reduce` 与 DDP strategy 入口；Flex CPU 没有 collective 实现。
4. CPU 上你观察到单设备 SGD 使 loss 下降并改变参数。
5. GPU 阅读时应对照：多 `Device`、collective 后端与拓扑带来的字节×延迟项。
6. 不能把单机 CPU 训练 loop 当成 NCCL/跨节点 DDP 已经验证。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-6-章)。
