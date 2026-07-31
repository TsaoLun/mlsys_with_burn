# 8.8 练习、延伸阅读与来源

## 小结

强化学习系统把环境、policy、transition、replay 和 learner 连接成一条
有状态的数据路径。MDP 提供 state/action/reward 的数学语义；Rust
`Environment` 和 `Policy` traits 把环境与模型的所有权边界写出来；
`TransitionBuffer` 提供有限容量的 tensor replay；`AsyncPolicy` 和
`burn-train` runner 提供单进程多环境、自动 batching、off-policy
编排、评估和 checkpoint 入口。

固定快照没有因此自动提供 DQN、PPO、SAC、prioritized replay、分布式
Actor–Learner 或 MARL league。具体 loss、optimizer、target network、
探索策略、版本协议和故障恢复仍由应用实现。第 8 章实验用确定性 CPU
环境和表格 TD 更新验证这些边界；固定的 DQN example 可作为进一步阅读，
但它依赖独立的 gym/native 环境。

## 概念题

1. 为什么 observation 不一定等于 environment state？部分可观测时，策略
   可能需要保存哪些历史信息？
2. `done` 和 `truncated` 在 bootstrap 中为什么可能有不同语义？固定
   `burn-train` 的 transition 路径在哪里合并了它们？
3. 经验回放为什么会改变数据分布？什么时候短期 replay 仍可被称为
   on-policy batch？
4. 增大 `num_envs`、`autobatch_size` 和 `train_steps` 分别会怎样影响
   queue wait、device utilization、policy staleness 和样本吞吐？
5. 为什么 `MultiAgentEnvLoop` 的“agent”不能直接说明 Burn 已支持
   多智能体强化学习？
6. policy parameters、target network、optimizer state、exploration step
   和 replay 哪些必须进入严格恢复协议？请给出理由。

## Rust 与 API 题

1. 给 `CounterEnv` 实现一个 `ToObservation<Tensor<2>>` adapter，并测试
   position/step 的 shape 和 device。
2. 为一个结构化 action 实现 `ToAction`，拒绝超出 action space 的值，
   不要把错误动作静默截断。
3. 为一个自定义 observation 实现 `Batchable` 和 `SliceAccess`，测试
   `batch → unbatch`、`push → sample` 的第 0 维语义。
4. 写一个最小 `Policy`，让 deterministic 模式选择 greedy action，
   training 模式使用显式 RNG；把 exploration context 记录到 metric。
5. 把实验的表格 Q update 替换为单层 Burn module，比较 target、loss、
   backward 和 optimizer step 的边界。
6. 为 policy、target 和 optimizer 定义一个组合 `Checkpoint` record，测试
   缺少一段 frame、版本不匹配和恢复后的下一步数值。
7. 使用 `AsyncPolicy` 的 mock policy 发出多个并发 action 请求，记录
   batch size 和 queue wait；说明为什么测试不能依赖 `sleep` 的偶然时序。

## 源码题

1. 阅读 `burn/crates/burn-rl/src/environment/base.rs`，指出
   `MAX_STEPS`、`done` 和 `truncated` 的职责。
2. 阅读 `burn/crates/burn-rl/src/policy/base.rs`，画出
   `Policy`、`PolicyState`、`Batchable`、`ToObservation` 和
   `ToAction` 的类型关系。
3. 阅读 `burn/crates/burn-rl/src/transition_buffer/base.rs` 和
   `slice_access.rs`，追踪第一次 `push` 的惰性分配、环形覆盖和
   `sample` 的共同 indices。
4. 阅读 `burn/crates/burn-rl/src/policy/async_policy.rs`，说明
   `num_agents`、`max_autobatch_size`、flush 和 update 顺序如何影响
   请求延迟。
5. 阅读 `burn/crates/burn-train/src/learner/rl/env_runner/base.rs` 和
   `async_runner.rs`，比较同步 runner、单环境线程和多环境 runner 的
   transition/trajectory 生命周期。
6. 阅读 `burn/crates/burn-train/src/learner/rl/off_policy.rs`，
   记录 collect、replay push、sample、learner train、evaluation 和
   checkpoint 的实际调用顺序。
7. 阅读 `burn/examples/dqn-agent/src/agent.rs` 与 `training.rs`，标出
   Policy、TD target、optimizer、target soft update、record 和
   `RLTraining` 的边界。

## 性能与系统题

1. 在固定环境数和 episode 长度下分别测 environment、policy、queue、
   replay 和 learner 时间；报告 steps/s 与 p50/p95/p99。
2. 改变 `autobatch_size`，画出 batch size、queue wait 和 policy throughput
   的关系；不要把单次 forward latency 当成 rollout throughput。
3. 比较 CPU environment + GPU policy 与同设备 environment/policy，记录
   observation/action copy 和同步边界。
4. 实现一个带序号的多 worker trajectory reorder layer，验证 worker 返回
   顺序变化时 episode 内 step 不乱序。
5. 为 replay 加入 n-step return 或 prioritized index，列出它改变的
   memory、sampling bias 和 checkpoint 字段。
6. 设计 Actor–Learner 的 policy version/trajectory schema、queue 上限、
   retry、duplicate detection 和 checkpoint 恢复协议。
7. 为两名 agent 的剪刀-石头-布实现联合 action 和 reward vector，说明
   单独复制单智能体 `Policy` 为什么不能解决 non-stationarity。

## 延伸阅读与固定源码入口

Burn 固定快照：

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

## 来源与改编说明

本章改编并重组 OpenMLSys v1 的
`chapter_reinforcement_learning/`：

- `index.md`：保留基础、单节点/分布式和多智能体的学习地图，改成
  本章的环境→采样→更新→系统边界路线；
- `rl_introduction.md`：保留 Agent/Environment、state/observation、
  action/reward、MDP、Markov property 和 discounted return，改用 Rust
  `Environment`/`StepResult` 与 `done`/`truncated` 解释；
- `single_node_rl.md`：保留 policy/value、adapter、learner、replay、
  online/offline 和多环境采样，改为 `burn-rl` traits、`TransitionBuffer`
  和 `burn-train` off-policy pipeline；
- `marl.md`：保留联合动作、奖励向量、合作/竞争/self-play 与非平稳性，
  明确固定 Burn 当前没有通用 MARL API；
- `marl_sys.md`：保留 Actor/Learner、league、模型评估/选择和 inference
  server 的系统问题，改为能力边界与未来协议，而不是现成 Burn runtime；
- `summary.md`：重写为采样吞吐、设备协同、checkpoint 和可复现性的核验
  清单。

没有复制 OpenMLSys 的图、框架专用代码、外部 simulator 或硬件性能数字。
`planning/chapter-sources/ch08.md` 记录逐文件来源、固定 Burn 路径和
实验范围；D011 记录确定性环境 + replay/TD 实验与完整 DQN/MARL 的隔离。
OpenMLSys 改编正文采用 CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT
OR Apache-2.0。
