# 第 8 章 强化学习系统

本章对应 OpenMLSys 的强化学习系统，是全书的**可选轨**。若你的目标是
训练、推理服务和 GPU 集群，读完第 7 章可以直接进入第 9 章。若你需要
理解「训练数据由策略和环境共同产生」时系统多出哪些边界，再读本章。

强化学习（reinforcement learning，RL）不是把监督学习的 loss 换一个名字。
智能体观察环境、选择动作，环境返回下一状态与奖励；采样器再把交互交给
学习器。环境速度、策略版本、回放容量和终止语义，都会改变算法实际看到
的数据。

产业里对应 Gym / EnvPool、replay buffer、Impala / Actor–Learner，以及
大模型对齐里的 RLHF 采样—训练分离。`burn-rl` 提供环境、策略和 buffer
的组合抽象；具体 DQN / PPO 由应用实现。

## 本章问题

如何把环境交互、轨迹、经验回放、策略更新和评估组织成一个可测试的系统？
哪些是组合抽象，哪些必须由算法和应用补上？

## 学习目标

完成本章后，你应该能够：

1. 用 MDP 区分 state、observation、action、reward、`done` 和 `truncated`；
2. 解释环境 step、策略 inference、transition 和 episode 的所有权边界；
3. 区分 on-policy、off-policy、trajectory 与 replay batch；
4. 使用 `Environment` 和 `TransitionBuffer` 构造 rollout；
5. 解释 `Policy`、`Batchable` 和 `PolicyLearner` 如何组合；
6. 用采样吞吐、inference batching 和设备拷贝建立成本模型；
7. 说明多环境 off-policy 编排、评估和 checkpoint 的边界；
8. 识别多智能体、Actor–Learner 和跨节点通信仍需额外协议的部分。

## 先修知识

建议先完成第 2 章和第 6 章。不要求游戏模拟器或集群。

## 本章路线

![强化学习数据环路：Environment 与 Policy 交互产生 rollout/trajectory，再经 on-policy batch 或 replay sample 进入 learner/evaluation](img/ch08-rl-loop.svg)

实验使用一个确定性、无外部依赖的小环境，验证 transition 形状、episode
边界、循环 buffer，以及在线 TD 与回放驱动 TD 的数据分布差异。

## 小节

1. [MDP、环境与轨迹边界](ch08/01-mdp-environment-and-trajectory.md)
2. [Policy、观察转换与动作批处理](ch08/02-policy-and-batching.md)
3. [Transition、回放与采样](ch08/03-replay-and-sampling.md)
4. [Rollout 吞吐、异步环境与推理队列](ch08/04-rollout-throughput.md)
5. [TD 更新、off-policy 与训练编排](ch08/05-learning-and-off-policy.md)
6. [多智能体与分布式系统边界](ch08/06-multi-agent-boundary.md)
7. [实验：CPU 确定性 rollout 与 replay](ch08/07-rollout-lab.md)
8. [练习、延伸阅读与来源](ch08/08-exercises-and-sources.md)

第 9 章把多个训练或采样进程放回共享集群，讨论队列、拓扑、故障与遥测。

示例位于 `examples/ch08-rl-rollout`。
