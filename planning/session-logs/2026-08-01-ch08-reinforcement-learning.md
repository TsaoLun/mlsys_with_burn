# 2026-08-01：第 8 章强化学习系统

## 会话目标

承接第 7 章交接，审查固定 OpenMLSys v1
`chapter_reinforcement_learning/` 和 Burn 0.22.0-pre.1 的 `burn-rl`/
`burn-train` RL 源码，完成第 8 章来源映射、CPU 可验证 rollout 实验、
正文和交接状态。

## 源码核验

### OpenMLSys

逐文件审查：

- `index.md`：强化学习基础、单节点/分布式和多智能体地图；
- `rl_introduction.md`：Agent、Environment、state/observation、action、
  reward、MDP、Markov property 和折扣回报；
- `single_node_rl.md`：policy/value、adapter、learner、经验回放、
  online/offline、多环境采样和 CPU/GPU simulator 边界；
- `marl.md`：联合动作、奖励向量、合作/竞争/self-play 和非平稳性；
- `marl_sys.md`：Actor/Learner、league、模型评估/选择和 inference server；
- `summary.md`：采样/训练平衡、设备协同和分布式挑战。

没有复制 OpenMLSys 图片、外部框架代码、simulator 性能数字或
`chapter_rl_sys/` 的机器人/ROS 内容。

### Burn

固定 Burn revision 为 `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`。核验：

- `burn-rl/src/environment/base.rs`：`Environment`、`StepResult` 和
  `EnvironmentInit`；
- `burn-rl/src/policy/base.rs`：`Policy`、`PolicyState`、`Batchable`、
  `ToObservation`/`ToAction` 和 `PolicyLearner`；
- `burn-rl/src/transition_buffer/base.rs`/`slice_access.rs`：惰性分配、
  circular overwrite、random select 和 shape contract；
- `burn-rl/src/policy/async_policy.rs`：native thread、mpsc、
  `num_agents`、autobatch、flush 和未实现的 wrapper `load_record`；
- `burn-train/src/learner/rl/env_runner/`：同步、单环境异步、多环境
  rollout、trajectory 和 policy update；
- `burn-train/src/learner/rl/off_policy.rs`：collect→replay→sample→
  `PolicyLearner::train`→evaluation/checkpoint；
- `burn-train/src/learner/rl/paradigm.rs`/`components.rs`/`checkpointer.rs`：
  RL 组件关联类型、metrics、inference device 和双 record checkpoint；
- `burn/examples/dqn-agent/`：完整 DQN 组合，以及 gym-rs/native SDL2
  独立 workspace 依赖。

`burn-rl` 是组合抽象，不是内置 DQN/PPO/SAC；`burn-train` 是训练编排，
具体 learner、loss、optimizer、target network 和探索策略仍由应用实现。
多环境 runner 不自动等于多智能体 runtime。

## 实现

新增：

- `examples/ch08-rl-rollout/Cargo.toml`
- `examples/ch08-rl-rollout/src/lib.rs`
- `examples/ch08-rl-rollout/src/main.rs`
- `book/src/ch08/01-mdp-environment-and-trajectory.md`
- `book/src/ch08/02-policy-and-batching.md`
- `book/src/ch08/03-replay-and-sampling.md`
- `book/src/ch08/04-rollout-throughput.md`
- `book/src/ch08/05-learning-and-off-policy.md`
- `book/src/ch08/06-multi-agent-boundary.md`
- `book/src/ch08/07-rollout-lab.md`
- `book/src/ch08/08-exercises-and-sources.md`
- `planning/chapter-sources/ch08.md`

修改：

- 根 `Cargo.toml`/`Cargo.lock`：加入第 8 章 workspace example 和固定
  Burn `rl` 依赖闭包；
- `book/src/ch08-rl-systems.md`：入口、目标、路线和能力边界；
- `book/src/SUMMARY.md`：加入八个小节导航；
- `planning/DECISIONS.md`：增加 D011；
- `planning/STATUS.md`：完成第 8 章交接并指向第 9 章；
- 本日志和 `planning/session-logs/README.md`。

实验路径：

```text
deterministic CounterEnv
  → Environment::step
  → state/action Tensor<2>
  → TransitionBuffer(capacity)
  → random replay batch
  → tabular TD update
```

实验用 `done || truncated` 作为 replay 的 terminal flag，测试自然终止/
时间截断、buffer capacity、sample shapes、TD bootstrap 和参数错误。

## 验证

已通过：

- `cargo fmt --all`
- `cargo test -p ch08-rl-rollout`（4 tests passed）
- `cargo clippy -p ch08-rl-rollout --all-targets -- -D warnings`
- `cargo run -p ch08-rl-rollout`，输出
  `transitions=6 buffer_len=4 terminal_transitions=1 state_shape=[2, 2]`
  `action_shape=[2, 1] reward_shape=[2, 1] done_shape=[2, 1]`
  `initial_right_q=0.9762`
- `mdbook build book`
- `make check`（pin、mdBook、fmt、workspace clippy 和 workspace tests）
- `make check-local-sources`
- `git diff --check`
- IDE lint diagnostics：无错误。

Cargo 仍提示用户目录同时存在 `~/.cargo/config` 和
`~/.cargo/config.toml`，未修改用户级配置。

## 决策与边界

- D011：基础实验采用确定性环境 + `TransitionBuffer` + 表格 TD，隔离
  `burn-rl` 抽象与具体神经网络 RL 算法；完整 DQN/MARL 不进入根 CI。
- 没有加入 gym-rs、SDL2、GPU simulator、网络通信或外部随机环境。
- 没有把 `MultiAgentEnvLoop` 宣称为 MARL，也没有声称固定快照提供
  prioritized replay、n-step return、跨节点 Actor–Learner、league 或
  通用 policy version 协议。
- 没有添加本地 path dependency、`[patch]`、生成 mdBook 输出或上游修改。

## 交接

第 8 章正文、逐文件来源映射、CPU rollout/replay/TD 实验和全量检查已
完成。下一步是审查 OpenMLSys v1 分布式训练中的 cluster/systems 内容，
核验固定 CubeCL/CUDA/NCCL/调度边界，开始第 9 章“大规模 GPU 集群管理”。
