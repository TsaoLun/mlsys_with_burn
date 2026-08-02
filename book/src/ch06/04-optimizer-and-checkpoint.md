# 优化器、学习率与检查点

## 优化器不只是一个公式

对 SGD 而言，最简单的更新是：

$$
\theta\_{t+1} = \theta\_t - \eta\_t g\_t.
$$

加入 momentum、weight decay 或 gradient clipping 后，更新还依赖额外状态
和策略。对 Adam，更新需要一阶矩、二阶矩和步数。训练框架必须同时保证：

1. 梯度按正确的 `ParamId` 到达参数；
2. optimizer state 与对应参数形状、设备和参数组匹配；
3. 当前 learning-rate policy 使用正确的 step；
4. state 能在 checkpoint 后恢复。

## `ModuleOptimizer` 的参数映射

固定 `burn-optim/src/optim/module/module_optimizer.rs` 的
`ModuleOptimizer` 用动态 optimizer 保存每个参数的上下文。参数在
`ModuleMapper` 遍历时以 `ParamId` 和路径匹配 optimizer group，然后：

```text
ParamId + path
      │
      ├── choose optimizer / clipping
      ├── load previous state
      ├── move parameter and gradient to one Device
      ├── apply optimizer.step
      └── store updated state
```

它提供：

- `step`：消费一个 `GradientsParams`；
- `step_multi`：消费多个 `(GradientsParams, Device)`；
- `with_group`：按参数组使用不同 optimizer；
- `with_grad_clipping`：在更新前处理梯度；
- `to_record` / `load_record`：保存和恢复 optimizer state。

因此 optimizer 不应被写成“遍历一个 Vec 权重”的教学捷径。module 的结构、
参数 ID 和状态映射正是大模型、冻结层和多 optimizer 训练能否安全组合的
关键。

## 学习率调度

`ModuleLearningRate` 把当前学习率映射到参数组；`ModuleLrScheduler` 负责
推进一个或多个 scheduler。固定源码允许默认组和特定 `ParamGroup` 使用
不同 schedule，并把 scheduler state 保存为 `LrSchedulerRecord`。

训练系统至少要记录：

```text
global step / epoch
    → scheduler.step()
    → effective learning rate per group
    → optimizer update
```

“学习率写在配置文件里”不等于恢复时学习率相同。若 scheduler 已经走了
若干步却从头初始化，后半段的更新将改变。

## 检查点包含哪些 record

固定 `burn-train` 的默认 checkpointer 会分开保存：

```text
checkpoint/
├── model-<epoch>.bpk
├── optim-<epoch>.bpk
└── scheduler-<epoch>.bpk
```

`ModuleRecord`、`OptimizerRecord` 和 `LrSchedulerRecord` 是 device-free 的
burnpack record。加载 model 时，参数保留已有 module 的设备；加载 optimizer
后，state 在下一次更新时迁移到参数/梯度的设备。这种设计避免把一个特定
设备的句柄硬编码到 checkpoint 文件中。

但 device-free 不代表“什么都可以恢复”。固定 `LearningCheckpointer`
只组合上述三类训练状态，并由 epoch 和 checkpointing strategy 决定保存/
删除。数据 loader 的文件偏移、sampler RNG、当前 shard、外部数据版本和
集群成员列表需要应用或作业层另行记录。

## 恢复协议

一个可审计的恢复点应至少有：

```text
model record
+ optimizer record
+ scheduler record
+ completed epoch / iteration
+ dataset version and sampler seed/state
+ device or shard assignment
+ configuration / code revision
```

如果只追求“从某个 epoch 的模型继续跑”，可以接受重新生成下一轮 shuffle；
如果要复现实验轨迹，则需要定义 sampler 和随机状态的精确恢复语义。第 5
章的固定 seed 测试只验证了可复现的起点，不等于完整 checkpoint 协议。

## 梯度累积与 checkpointing

`SupervisedTraining::grads_accumulation(n)` 让多次 backward 的梯度先进入
`GradientsAccumulator`，到第 `n` 次才 optimizer step。它改变有效 batch、
更新频率和日志解释，不能只把 `n` 当作性能开关。

`gradient_checkpointing()` 则是 autodiff 的内存/计算 trade-off：对适合
重算的操作少保存激活，反向时重新计算。它减少峰值内存但可能增加计算；
与 optimizer checkpoint 是不同层次的机制。

## 实验中的可观察状态

本章实验不保存文件 checkpoint，而是把 `initial_loss`、`final_loss`、
每一步 `losses` 和参数变化量作为最小状态观测。生产训练应在此基础上再
测试三个 record 的往返保存与恢复，以及恢复后 scheduler 和 optimizer state
是否与未中断轨迹一致。
