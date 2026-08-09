# TD 更新、off-policy 与训练编排

## 从 transition 到 TD target

以离散动作的 Q-learning 为例，单步 TD target 为：

$$
y\_t =
\begin{cases}
r\_t, & d\_t=1,\\
r\_t+\gamma\max\_{a'}Q\_{\bar\theta}(s\_{t+1},a'), & d\_t=0.
\end{cases}
$$

当前 Q 值 $Q\_\theta(s\_t,a\_t)$ 再沿误差方向更新：

$$
\theta \leftarrow \theta -
\eta\nabla\_\theta
\mathcal{L}\left(Q\_\theta(s\_t,a\_t),y\_t\right).
$$

这里的 $\bar\theta$ 可以是 target network，也可以是延迟的参数快照。
关键系统事实是：target 是否 bootstrap、哪个 policy 产生数据、reward
如何缩放，都必须和 transition 的 flags/metadata 一致。一个 shape 正确
的 MSE 并不能证明 RL 语义正确。

本章实验的表格更新使用同一个 target 公式，但不建立神经网络和 autodiff
tape。这样读者可以先观察 `done` 如何切断 bootstrap，再进入 Burn
module、gradient 和 optimizer；这是有意的分层，而不是声称 Burn 的 RL
训练只需要表格。

## 探索、行为策略与数据分布

训练时执行动作的行为策略 $\mu$ 不一定等于用于评估或更新的目标策略
$\pi$。最简单的离散探索是 epsilon-greedy：

$$
a\_t =
\begin{cases}
\text{随机动作}, & \text{概率 }\varepsilon,\\
\arg\max\_a Q(s\_t,a), & \text{概率 }1-\varepsilon.
\end{cases}
$$

epsilon 的初值、衰减步数和恢复位置会改变 replay 的数据分布。on-policy
更新通常要求 batch 仍来自当前 $\pi$；off-policy 更新允许来自较旧的
$\mu$，但可能需要 importance sampling、target network、行为策略版本或
其他稳定化措施。仅仅把数据放进 `TransitionBuffer` 不会完成这些校正。

因此 rollout 与 learner 之间至少应能记录：

```text
transition → behavior_policy_version
           → exploration state / log-probability (if needed)
           → target/learner policy version
```

## `PolicyLearner` 的职责

固定 `burn-rl::PolicyLearner` 定义：

- `train(LearnerTransitionBatch)`：消费一个批量 transition，返回
  `RLTrainOutput`；
- `policy()`：给 rollout/evaluation 使用的 policy；
- `update_policy()`：把更新后的 policy 送回 runner；
- `record()`/`load_record()`：保存和恢复 learner state；
- `device()`：声明 learner 使用的设备。

它没有规定 loss、gradient、optimizer、target network 或 exploration。
一个 DQN learner 可以将这些字段都放在自己的 struct 中；一个 policy
gradient learner 也可以返回 log probability、entropy 或 value loss。
Rust 的关联类型将具体 transition、training output 和 record 绑定起来，
减少了运行时类型标签，但实现者要承担更多 trait 约束。

## 固定 DQN example 的完整边界

固定 `burn/examples/dqn-agent` 是理解这条边界的最好源码入口。它由应用
自己实现：

1. CartPole wrapper 把 gym observation/action 转为 `Environment` 的
   `State`/`Action`；
2. `DQN` 实现 `Policy`，用 logits 选择离散动作；
3. epsilon-greedy wrapper 记录 exploration context；
4. `DqnLearningAgent::train` 计算 action value、target network 的
   next-state max、MSE、autodiff backward 和 optimizer step；
5. target model 通过 soft update 跟随 policy model；
6. `DqnLearningRecord` 把 policy、target 和 optimizer records 一起保存；
7. `RLTraining::OffPolicyStrategy` 负责 rollout、replay、evaluation 和
   checkpoint。

这段代码证明的是“本版可以组合出一个 DQN example”，不是
`burn-rl` 自带 DQN。该 example 还放在独立工程里，因为 gym-rs 会引入
native SDL2 等环境依赖；所以本书默认实验不直接复制它的外部 simulator。

## Checkpoint 不只保存 policy

RL 恢复比普通 inference 更敏感。至少有三种状态可能影响下一步：

```text
policy parameters
target / value parameters
optimizer moments and scheduler
exploration step / replay / RNG
```

固定 `RLCheckpointer` 将 policy record 与 learning-agent record 分开；
learning agent 的自定义 record 可以再把 model、target 和 optimizer
打包。它没有自动保存 replay 内容、外部环境状态或全局 RNG。若从
checkpoint 继续训练而重置了 epsilon step 或 replay 分布，新的训练可能
仍能运行，但不再是原实验的严格续跑。

第 7 章介绍的 `ModuleRecord` 只解决 module 参数 record 的保存/恢复；
第 8 章要进一步问：哪些策略/learner 状态属于算法不变量？哪些只是可
重新初始化的 cache？这正是 RL checkpoint schema 必须由应用定义的原因。

## 评估与训练的隔离

固定 `OffPolicyStrategy` 在 evaluation interval 触发 validation runner，
把 learner 的 policy state 更新到评估 runner，并用 deterministic 配置
运行 episode。评估还会发出 episode length、cumulative reward 等 metric
事件。

评估结果必须标记：

- 使用的是哪一个 policy version；
- 是否 deterministic；
- episode 数和最大长度；
- 环境 seed/version；
- reward 是 episode sum 还是 step mean。

否则“reward 上升”可能只是 episode 更长、探索减少或环境版本不同。

## 本节小结

RL 算法实现位于 `PolicyLearner` 与应用 model 之间，训练编排位于
`RLTraining`/`OffPolicyStrategy`。固定 Burn 为两者提供可组合的边界，但
不替用户选择 bootstrap、探索、target state、loss、optimizer 和恢复
协议。先用表格 TD 实验验证数学，再用固定 DQN example 对照神经网络
实现，是较安全的学习路径。
