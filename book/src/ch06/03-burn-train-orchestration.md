# `burn-train` 的 Learner 与训练装配

## 从 step 到训练环境

手写循环适合验证数学和 ownership 边界，但实际训练还需要：

- 训练/验证 DataLoader；
- epoch、iteration 和 progress；
- loss、accuracy、learning rate 等 metric；
- 日志、renderer 和 experiment directory；
- early stopping；
- module、optimizer、scheduler 的 checkpoint；
- 单设备、本机多设备或 DDP 的 execution strategy。

固定 Burn 快照把这些职责放进 `burn-train`，而不是让 `Module` 自己负责
文件、线程和 UI。

## `Learner` 的组合

`burn-train/src/learner/base.rs` 中的 `Learner<M>` 保存四类核心对象：

```text
Learner<M>
├── model: M
├── ModuleOptimizer
├── ModuleLrScheduler
└── current ModuleLearningRate
```

它提供的操作很窄但很关键：

- `train_step(item)` 调用 `TrainStep::step`；
- `lr_step()` 前进 scheduler；
- `optimizer_step(grads)` 执行单设备更新；
- `optimizer_step_multi(grads)` 执行多设备更新；
- `fork(device)` 把 model 的参数迁移/复制到目标设备；
- `load_model`、`load_optim`、`load_scheduler` 应用各自的 record。

`Learner` 不负责决定 batch 如何读取；输入通过
`Arc<dyn DataLoader<TrainingModelInput<M>>>` 注入 supervised runner。这样
数据系统和训练策略可以分别替换，也符合第 5 章的数据边界。

## `SupervisedTraining` 负责装配

固定 `burn-train/src/learner/supervised/paradigm.rs` 的
`SupervisedTraining::new` 接受训练和验证 loader，然后以 builder 方法配置：

```text
SupervisedTraining
├── metrics / event processor
├── renderer / application logger
├── checkpointing strategy
├── gradient accumulation / checkpointing
├── early stopping / interrupter
└── ExecutionStrategy
```

`launch(learner)` 时，它会建立 event processor，按需要建立三个
checkpointer，并把训练交给 `SingleDeviceTrainingStrategy`、
`MultiDeviceLearningStrategy`、DDP 或 custom learning paradigm。

默认策略根据 model 的第一个 device 选择单设备路径。显式配置策略时，不能
只看 `devices.len()`：`MultiDevice` 和 `DistributedDataParallel` 的同步
实现不同，数据 loader 切法和参数梯度处理也不同。

## 单设备 epoch 的顺序

固定 `single/epoch.rs` 的循环可以压缩成：

```text
for item in train_loader:
    lr_step()
    output = learner.train_step(item)
    optionally accumulate output.grads
    optimizer_step(accumulated_or_current_grads)
    emit ProcessedItem(metric/event)
```

验证阶段使用 `learner.model().valid()` 和 `InferenceStep`，不执行 optimizer
更新。epoch 结束后，runner 可以写 checkpoint，并根据 event store 判断
early stopping。

注意学习率前进的位置：固定单设备策略在每个训练 item 前调用 `lr_step`。
如果自定义策略改变调用次数，scheduler 曲线也会改变；这不是一个只影响
日志的字段。

## 指标和日志不是训练数学

metric processor 观察 `TrainingModelOutput` 和 `InferenceModelOutput`，把
结果放进 event store。它可以报告 loss、accuracy、速度或设备指标，但
不能替代对参数更新的测试。一个吞吐量很高却不更新参数的循环仍然是错误
的训练系统。

同样，renderer 只负责展示；`experiment.log` 和 metric 文件是诊断材料，
不是自动的可恢复 checkpoint。要恢复训练，必须显式配置 checkpointer，
并确保数据 epoch/shard 协议与恢复点一致。

## 为什么实验不直接用 `burn-train`

本章的实验只做线性回归和 loss 下降验证，故意使用 `burn::optim` 手写
循环。这样测试失败时可以把问题缩小到：

```text
Tensor / autodiff / GradientsParams / SGD
```

如果一开始就引入 renderer、metric、文件 checkpoint 和多个 worker，出错时
很难分清是装配问题还是梯度问题。完成实验后，可以把相同模型接到
`TrainStep` 再交给 `Learner`；那是 API 装配练习，不是另一套优化数学。
