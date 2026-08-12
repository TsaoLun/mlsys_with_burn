# 练习、延伸阅读与来源

## 小结

强化学习系统把环境、policy、transition、replay 和 learner 连接成一条
有状态的数据路径。MDP 提供 state/action/reward 的数学语义；Rust
`Environment` 和 `Policy` traits 把环境与模型的所有权边界写出来；
`TransitionBuffer` 提供有限容量的 tensor replay；`AsyncPolicy` 和
`burn-train` runner 提供单进程多环境、自动 batching、off-policy
编排、评估和 checkpoint 入口。

本版没有因此自动提供 DQN、PPO、SAC、prioritized replay、分布式
Actor–Learner 或 MARL league。具体 loss、optimizer、target network、
探索策略、版本协议和故障恢复仍由应用实现。第 8 章实验用确定性 CPU
环境和表格 TD 更新验证这些边界；固定的 DQN example 可作为进一步阅读，
但它依赖独立的 gym/native 环境。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么 observation 不一定等于 environment state？部分可观测时，策略
   可能需要保存哪些历史信息？

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【基础】`done` 和 `truncated` 在 bootstrap 中为什么可能有不同语义？固定
   `burn-train` 的 transition 路径在哪里合并了它们？

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

3. 【基础】经验回放为什么会改变数据分布？什么时候短期 replay 仍可被称为
   on-policy batch？

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

4. 【进阶】同一环境、同一动作序列下，为什么在线 TD 与回放驱动 TD 会学到
   不同的 Q 值？`capacity = 1` 时 `initial_right_q` 为什么精确等于 0？

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照实验的第 4 节；先回答 learner 在
两条路径下分别能看到哪些 transition，再检查 TD 公式本身是否改变。

</details>

5. 【基础】增大 `num_envs`、`autobatch_size` 和 `train_steps` 分别会怎样影响
   queue wait、device utilization、policy staleness 和样本吞吐？

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【进阶】为什么 `MultiAgentEnvLoop` 的“agent”不能直接说明 Burn 已支持
   多智能体强化学习？

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

7. 【进阶】policy parameters、target network、optimizer state、exploration step
   和 replay 哪些必须进入严格恢复协议？请给出理由。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

8. 【进阶】对同一条 episode 分别计算 Monte Carlo return、TD(0) target 和
   Q-learning target，比较 bias/variance、终止和截断 step 的处理。

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

9. 【进阶】实现 epsilon-greedy 的可恢复衰减状态，记录 behavior policy version；
   解释为什么仅保存 Q 网络参数不足以重现 replay 分布。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>


## Rust 与 API 题

1. 【基础】给 `CounterEnv` 实现一个 `ToObservation<Tensor<2>>` adapter，并测试
   position/step 的 shape 和 device。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【基础】为一个结构化 action 实现 `ToAction`，拒绝超出 action space 的值，
   不要把错误动作静默截断。

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【进阶】为一个自定义 observation 实现 `Batchable` 和 `SliceAccess`，测试
   `batch → unbatch`、`push → sample` 的第 0 维语义。

<details>
<summary>提示</summary>

回看第 8 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【进阶】写一个最小 `Policy`，让 deterministic 模式选择 greedy action，
   training 模式使用显式 RNG；把 exploration context 记录到 metric。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

5. 【进阶】把实验的表格 Q update 替换为单层 Burn module，比较 target、loss、
   backward 和 optimizer step 的边界。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

6. 【进阶】为 policy、target 和 optimizer 定义一个组合 `Checkpoint` record，测试
   缺少一段 frame、版本不匹配和恢复后的下一步数值。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

7. 【进阶】使用 `AsyncPolicy` 的 mock policy 发出多个并发 action 请求，记录
   batch size 和 queue wait；说明为什么测试不能依赖 `sleep` 的偶然时序。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>


## 源码题

1. 【进阶】阅读 `burn/crates/burn-rl/src/environment/base.rs`，指出
   `MAX_STEPS`、`done` 和 `truncated` 的职责。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

2. 【进阶】阅读 `burn/crates/burn-rl/src/policy/base.rs`，画出
   `Policy`、`PolicyState`、`Batchable`、`ToObservation` 和
   `ToAction` 的类型关系。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

3. 【进阶】阅读 `burn/crates/burn-rl/src/transition_buffer/base.rs` 和
   `slice_access.rs`，追踪第一次 `push` 的惰性分配、环形覆盖和
   `sample` 的共同 indices。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

4. 【进阶】阅读 `burn/crates/burn-rl/src/policy/async_policy.rs`，说明
   `num_agents`、`max_autobatch_size`、flush 和 update 顺序如何影响
   请求延迟。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

5. 【进阶】阅读 `burn/crates/burn-train/src/learner/rl/env_runner/base.rs` 和
   `async_runner.rs`，比较同步 runner、单环境线程和多环境 runner 的
   transition/trajectory 生命周期。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

6. 【进阶】阅读 `burn/crates/burn-train/src/learner/rl/off_policy.rs`，
   记录 collect、replay push、sample、learner train、evaluation 和
   checkpoint 的实际调用顺序。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

7. 【进阶】阅读 `burn/examples/dqn-agent/src/agent.rs` 与 `training.rs`，标出
   Policy、TD target、optimizer、target soft update、record 和
   `RLTraining` 的边界。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>


## 性能与系统题

1. 【进阶】在固定环境数和 episode 长度下分别测 environment、policy、queue、
   replay 和 learner 时间；报告 steps/s 与 p50/p95/p99。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

2. 【挑战】改变 `autobatch_size`，画出 batch size、queue wait 和 policy throughput
   的关系；不要把单次 forward latency 当成 rollout throughput。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

3. 【挑战】比较 CPU environment + GPU policy 与同设备 environment/policy，记录
   observation/action copy 和同步边界。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

4. 【挑战】实现一个带序号的多 worker trajectory reorder layer，验证 worker 返回
   顺序变化时 episode 内 step 不乱序。

<details>
<summary>提示</summary>

从 `examples/ch05-data-pipeline` 与第 5 章对应小节观察。

</details>

5. 【挑战】为 replay 加入 n-step return 或 prioritized index，列出它改变的
   memory、sampling bias 和 checkpoint 字段。

<details>
<summary>提示</summary>

运行 `examples/ch08-rl-rollout` 并对照第 8 章对应抽象。

</details>

6. 【挑战】设计 Actor–Learner 的 policy version/trajectory schema、queue 上限、
   retry、duplicate detection 和 checkpoint 恢复协议。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>

7. 【挑战】为两名 agent 的剪刀-石头-布实现联合 action 和 reward vector，说明
   单独复制单智能体 `Policy` 为什么不能解决 non-stationarity。

<details>
<summary>提示</summary>

对照第 8 章「多智能体与分布式系统边界」：联合 transition 需要同时表达
所有 agent 的动作与奖励；非平稳性来自其他 agent 的策略变化，不是网络拓扑
或调度顺序问题。

</details>


## 延伸阅读与固定源码入口

本书所用的 Burn 版本：

- `burn/crates/burn-rl/src/environment/base.rs`
- `burn/crates/burn-rl/src/policy/base.rs`
- `burn/crates/burn-rl/src/policy/async_policy.rs`
- `burn/crates/burn-rl/src/transition_buffer/base.rs`
- `burn/crates/burn-rl/src/transition_buffer/slice_access.rs`
- `burn/crates/burn-train/src/learner/rl/components.rs`
- `burn/crates/burn-train/src/learner/rl/env_runner/base.rs`
- `burn/crates/burn-train/src/learner/rl/env_runner/async_runner.rs`
- `burn/crates/burn-train/src/learner/rl/off_policy.rs`
- `burn/crates/burn-train/src/learner/rl/paradigm.rs`
- `burn/crates/burn-train/src/learner/rl/checkpointer.rs`
- `burn/examples/dqn-agent/src/env.rs`
- `burn/examples/dqn-agent/src/agent.rs`
- `burn/examples/dqn-agent/src/training.rs`

OpenMLSys v1：

- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/index.md`
- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/rl_introduction.md`
- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/single_node_rl.md`
- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/marl.md`
- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/marl_sys.md`
- `openmlsys/v1/zh_chapters/chapter_reinforcement_learning/summary.md`

## 本章系统结论

1. RL 系统把“环境交互”接进与监督学习不同的数据与状态边界（done/truncated、轨迹）。
2. replay、policy 版本与 off-policy 元数据决定样本能否安全用于更新；容量与采样分布本身就是 learner 的数据边界。
3. CPU 上你观察到确定性 rollout、circular replay、在线与回放驱动两条 TD 路径在同一环境序列上的不同结果。
4. GPU 阅读线索：大批量 replay 与策略网络 forward 的设备放置、以及 Actor–Learner 间的版本延迟。
5. 不能把组合 API 或小型 TD 实验当成完整 DQN/PPO/MARL runtime。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-8-章)。
