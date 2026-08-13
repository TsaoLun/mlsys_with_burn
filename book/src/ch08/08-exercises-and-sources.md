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

[「MDP、环境与轨迹边界」](01-mdp-environment-and-trajectory.md)的
「从决策问题开始」区分了环境内部状态与策略收到的表示；
`examples/ch08-rl-rollout` 的 `CounterState` 把 position 与 step 全部
暴露，属于完全可观测的特例。思考方向：若把 `step` 字段藏起来，策略
要判断“还差几步被截断”，需要自己补回什么历史信息。

</details>

2. 【基础】`done` 和 `truncated` 在 bootstrap 中为什么可能有不同语义？固定
   `burn-train` 的 transition 路径在哪里合并了它们？

<details>
<summary>提示</summary>

两种结束原因的 bootstrap 差异见
[「MDP、环境与轨迹边界」](01-mdp-environment-and-trajectory.md)的
「一步、episode 与轨迹」。查合并位置时，先在
`burn/crates/burn-rl/src/transition_buffer/base.rs` 确认 `push` 只收
一个 done 布尔，再沿 `burn/crates/burn-train/src/learner/rl/` 下的
runner 源码找是谁把两个标志折叠后传进来。
`examples/ch08-rl-rollout` 的单容量测试给出可观察后果：截断的
transition 存入 replay 后，done 标志已经是 1。

</details>

3. 【基础】经验回放为什么会改变数据分布？什么时候短期 replay 仍可被称为
   on-policy batch？

<details>
<summary>提示</summary>

判断标准在[「Transition、回放与采样」](03-replay-and-sampling.md)的
「Online 与 off-policy」：分类取决于数据与更新 policy 的关系，而不是
有没有一个 buffer。可用 `examples/ch08-rl-rollout` 的两阶段对照具体
化：阶段 A 逐条即时更新，阶段 B 从容量窗口随机抽样、可能重复命中
同一条 transition——想想哪一侧的 batch 还能算“来自当前策略”。

</details>

4. 【进阶】同一环境、同一动作序列下，为什么在线 TD 与回放驱动 TD 会学到
   不同的 Q 值？`capacity = 1` 时 `initial_right_q` 为什么精确等于 0？

<details>
<summary>提示</summary>

对照[「实验：CPU 确定性 rollout 与 replay」](07-rollout-lab.md)的
「用 replay batch 驱动更新」：TD 公式在两条路径中完全相同，变的只是
learner 能看到的 transition 集合。检查 `capacity = 1` 时环形 buffer
里幸存的是哪一条 transition、它的 target 由什么组成，再推初始状态的
Q 值为什么一次也轮不到更新。

</details>

5. 【基础】增大 `num_envs`、`autobatch_size` 和 `train_steps` 分别会怎样影响
   queue wait、device utilization、policy staleness 和样本吞吐？

<details>
<summary>提示</summary>

[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)开头
的数字推演演示了瓶颈如何从环境移到 policy 合批；
「OffPolicyStrategy 的执行顺序」逐条讨论这几个配置字段的耦合，
「采样—更新平衡与策略陈旧」把 staleness 与采样/消费速率联系起来。
回答时固定其余参数，说明每个增量把等待搬到了队列、设备还是
learner 一侧，避免笼统的“变快/变慢”。

</details>

6. 【进阶】为什么 `MultiAgentEnvLoop` 的“agent”不能直接说明 Burn 已支持
   多智能体强化学习？

<details>
<summary>提示</summary>

[「多智能体与分布式系统边界」](06-multi-agent-boundary.md)的
「固定 Burn 可以组合什么」列出了这个名字实际提供的能力：多个环境
实例共享同一个 `AsyncPolicy` 的合批推理。从固定 `Environment` 只有
一个 `Action` 关联类型入手，检查联合动作、每个 agent 独立的 reward
与 credit assignment 各缺少什么支撑。

</details>

7. 【进阶】policy parameters、target network、optimizer state、exploration step
   和 replay 哪些必须进入严格恢复协议？请给出理由。

<details>
<summary>提示</summary>

[「TD 更新、off-policy 与训练编排」](05-learning-and-off-policy.md)的
「Checkpoint 不只保存 policy」给出判断框架：区分算法不变量与可以
重建的 cache。对照
`burn/crates/burn-train/src/learner/rl/checkpointer.rs`，看固定实现把
哪些 record 分开保存、哪些完全没有替你保存；对每一项论证遗漏它会
破坏什么——下一步动作、更新方向还是数据分布。

</details>

8. 【进阶】对同一条 episode 分别计算 Monte Carlo return、TD(0) target 和
   Q-learning target，比较 bias/variance、终止和截断 step 的处理。

<details>
<summary>提示</summary>

[「MDP、环境与轨迹边界」](01-mdp-environment-and-trajectory.md)的
「从决策问题开始」对比了 Monte Carlo 与 TD 的更新时机和方差来源；
`examples/ch08-rl-rollout` 的 `td_target` 函数及其终止测试展示了
done 如何切断 bootstrap。留给你的核心问题：截断 step 对三种 target
分别应当按终止处理还是继续 bootstrap，理由是什么。

</details>

9. 【进阶】实现 epsilon-greedy 的可恢复衰减状态，记录 behavior policy version；
   解释为什么仅保存 Q 网络参数不足以重现 replay 分布。

<details>
<summary>提示</summary>

[「TD 更新、off-policy 与训练编排」](05-learning-and-off-policy.md)的
「探索、行为策略与数据分布」解释了 epsilon 衰减位置如何改变 replay
分布；`examples/ch08-rl-rollout` 的 `PolicySampleMetadata` 与
`policy_is_fresh` 测试是行为/目标版本差的最小记法。思考方向：恢复
训练时若 epsilon 从头再衰减一遍，新写入 replay 的数据分布与中断前
有什么系统性差别。

</details>


## Rust 与 API 题

1. 【基础】给 `CounterEnv` 实现一个 `ToObservation<Tensor<2>>` adapter，并测试
   position/step 的 shape 和 device。

<details>
<summary>提示</summary>

转换接口的语义见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「三种表示之间的转换」：`to_observation(&self, device)` 应在转换时把
tensor 放到指定 Device。`examples/ch08-rl-rollout` 的 `state_tensor`
已把 position 与 step 编成 `[1, 2]` tensor，可作为 adapter 的起点；
tensor 构造与 Device 的基础见第 2 章
[「Tensor、Device 与运行时后端」](../ch02/02-tensor-device-backend.md)。
测试除了 shape，还要断言结果确实落在传入的 device 上。

</details>

2. 【基础】为一个结构化 action 实现 `ToAction`，拒绝超出 action space 的值，
   不要把错误动作静默截断。

<details>
<summary>提示</summary>

接口方向见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「三种表示之间的转换」；动手前先到
`burn/crates/burn-rl/src/policy/base.rs` 核对 `to_action` 的签名，
确认它有没有给错误留位置，再决定校验放在哪一层。拒绝非法值的风格
可参考 `examples/ch08-rl-rollout` 用 `RolloutError` 把无效配置变成
可断言返回值的做法，而不是 clamp 成合法动作。

</details>

3. 【进阶】为一个自定义 observation 实现 `Batchable` 和 `SliceAccess`，测试
   `batch → unbatch`、`push → sample` 的第 0 维语义。

<details>
<summary>提示</summary>

两个 trait 的分工见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「Batchable 与 batch 的语义」和
[「Transition、回放与采样」](03-replay-and-sampling.md)的
「SliceAccess 与 Rust 泛型」：`zeros_like`、`select` 和
`slice_assign_inplace` 都围绕第 0 维展开。按正文的提醒设计测试——
把“第 0 维是样本维”写成显式断言，而不是依赖一次 `cat` 恰好成功。

</details>

4. 【进阶】写一个最小 `Policy`，让 deterministic 模式选择 greedy action，
   training 模式使用显式 RNG；把 exploration context 记录到 metric。

<details>
<summary>提示</summary>

[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「Policy 不是算法」拆开了 `forward` 与 `action` 的分工：
`deterministic` 是显式调用参数，探索随机性应通过显式 RNG 与
`ActionContext` 旁路输出，而不是藏进全局状态。签名细节到
`burn/crates/burn-rl/src/policy/base.rs` 核对。自查：换一个 seed
重跑，deterministic 模式的输出必须完全不变。

</details>

5. 【进阶】把实验的表格 Q update 替换为单层 Burn module，比较 target、loss、
   backward 和 optimizer step 的边界。

<details>
<summary>提示</summary>

改造清单在[「实验：CPU 确定性 rollout 与 replay」](07-rollout-lab.md)
「观察在线 TD target」的末尾：把 `q_values` 换成 `Module`、用
`gather` 选 action value、算 loss、backward、optimizer step，并另外
维护 target network。完整对照可读
`burn/examples/dqn-agent/src/agent.rs`；autodiff 与 optimizer 的基础
路径见第 6 章
[「前向、反向与自定义训练循环」](../ch06/02-forward-backward-loop.md)。

</details>

6. 【进阶】为 policy、target 和 optimizer 定义一个组合 `Checkpoint` record，测试
   缺少一段 frame、版本不匹配和恢复后的下一步数值。

<details>
<summary>提示</summary>

[「TD 更新、off-policy 与训练编排」](05-learning-and-off-policy.md)的
「Checkpoint 不只保存 policy」区分了 policy record 与 learning-agent
record；组合打包的写法可对照 `burn/examples/dqn-agent/src/agent.rs`
与 `training.rs` 里的 learning record。测“恢复后的下一步数值”时，
参考第 7 章
[「实验：CPU 模型状态往返保存与恢复」](../ch07/07-record-roundtrip-lab.md)
的 round-trip 断言法：恢复前后各走一步，比较输出是否一致。

</details>

7. 【进阶】使用 `AsyncPolicy` 的 mock policy 发出多个并发 action 请求，记录
   batch size 和 queue wait；说明为什么测试不能依赖 `sleep` 的偶然时序。

<details>
<summary>提示</summary>

进程内合批与 flush 的机制见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「Batchable 与 batch 的语义」，实现按
`burn/crates/burn-rl/src/policy/async_policy.rs` 阅读。batch 的组成
取决于请求到达顺序与 server 线程调度，这些都不受 `sleep` 控制；
测试应断言与时序无关的性质，例如请求数守恒、每个请求都收到答复、
batch size 不超过上限。

</details>


## 源码题

1. 【进阶】阅读 `burn/crates/burn-rl/src/environment/base.rs`，指出
   `MAX_STEPS`、`done` 和 `truncated` 的职责。

<details>
<summary>提示</summary>

带着[「MDP、环境与轨迹边界」](01-mdp-environment-and-trajectory.md)
里 done/truncated 的 bootstrap 差异去读这份 trait 定义；
`examples/ch08-rl-rollout` 的 `CounterEnv` 是它的最小实现，测试分别
构造了“连续右移自然终止”与“四步截断”两条路径，可边读边跑对照。
注意 `MAX_STEPS` 是关联常量而不是运行时配置，想想这对环境实现者
意味着什么。

</details>

2. 【进阶】阅读 `burn/crates/burn-rl/src/policy/base.rs`，画出
   `Policy`、`PolicyState`、`Batchable`、`ToObservation` 和
   `ToAction` 的类型关系。

<details>
<summary>提示</summary>

先用[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)
给出的职责拆解（forward/action/update 加上两个转换 trait）当作图的
骨架，再到源码里核对关联类型的连线：哪些类型由 `Policy` 自己声明、
哪些经 `Batchable` 或转换 trait 的泛型参数进入。画完之后，用“一个
observation 从环境到分布再变回动作”走一遍图，检验有没有断边。

</details>

3. 【进阶】阅读 `burn/crates/burn-rl/src/transition_buffer/base.rs` 和
   `slice_access.rs`，追踪第一次 `push` 的惰性分配、环形覆盖和
   `sample` 的共同 indices。

<details>
<summary>提示</summary>

[「Transition、回放与采样」](03-replay-and-sampling.md)的
「Circular replay buffer」把固定实现的行为列成了五步，可当作阅读
提纲逐条到源码里找对应；`examples/ch08-rl-rollout` 的
`unit_capacity_retains_only_the_latest_aligned_transition` 测试展示
覆盖之后所有字段仍指向同一条 transition——读 `sample` 时留意这种
对齐由哪一行代码保证。

</details>

4. 【进阶】阅读 `burn/crates/burn-rl/src/policy/async_policy.rs`，说明
   `num_agents`、`max_autobatch_size`、flush 和 update 顺序如何影响
   请求延迟。

<details>
<summary>提示</summary>

请求延迟的组成见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)中
「Batchable 与 batch 的语义」末尾的吞吐/延迟交换讨论。读源码时跟踪
一条请求从 mpsc 入队到答复返回经过的分支，重点回答两问：凑不满
batch 时由谁触发 flush；update 消息与在途 action 请求的先后顺序
如何影响最早到达请求的等待。

</details>

5. 【进阶】阅读 `burn/crates/burn-train/src/learner/rl/env_runner/base.rs` 和
   `async_runner.rs`，比较同步 runner、单环境线程和多环境 runner 的
   transition/trajectory 生命周期。

<details>
<summary>提示</summary>

[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)的
「固定 Burn 的三种 runner」概括了三层结构与 double batching 的
“领先一步”设计。阅读时为每种 runner 画一条 transition 从
`env.step` 到交给调用方的路径，标出它跨越哪些 channel、在哪一步被
组装成 `Trajectory`，再比较三条路径对 reset 边界的处理差异。

</details>

6. 【进阶】阅读 `burn/crates/burn-train/src/learner/rl/off_policy.rs`，
   记录 collect、replay push、sample、learner train、evaluation 和
   checkpoint 的实际调用顺序。

<details>
<summary>提示</summary>

对照[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)
「OffPolicyStrategy 的执行顺序」的流程图逐段读源码，确认
`warmup_steps`、`train_interval`、`train_steps` 各自的判断位置；再
记下 evaluation 与 checkpoint 挂在循环的哪个环节、与流程图是否
一致。发现顺序与预期不同时，先想它对 replay 新鲜度意味着什么。

</details>

7. 【进阶】阅读 `burn/examples/dqn-agent/src/agent.rs` 与 `training.rs`，标出
   Policy、TD target、optimizer、target soft update、record 和
   `RLTraining` 的边界。

<details>
<summary>提示</summary>

[「TD 更新、off-policy 与训练编排」](05-learning-and-off-policy.md)的
「固定 DQN example 的完整边界」已把应用侧要自己实现的七件事列成
清单，阅读时给每一项找到对应代码位置即可。特别注意区分哪些类型来自
burn-rl/burn-train 的 trait、哪些是这个 example 自带的实现——这条
边界正是“能组合出 DQN”与“自带 DQN”的差别。

</details>


## 性能与系统题

1. 【进阶】在固定环境数和 episode 长度下分别测 environment、policy、queue、
   replay 和 learner 时间；报告 steps/s 与 p50/p95/p99。

<details>
<summary>提示</summary>

[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)的
「吞吐与延迟的测量」列出了至少要分开记录的六类时间，并解释了为何
平均值不够、要报长尾分位数。`examples/ch08-rl-rollout` 的确定性环境
适合当计时脚手架：先在单线程下验证分段时间之和接近总时长，再引入
异步与队列，避免一开始就把测量误差和排队效应混在一起。

</details>

2. 【挑战】改变 `autobatch_size`，画出 batch size、queue wait 和 policy throughput
   的关系；不要把单次 forward latency 当成 rollout throughput。

<details>
<summary>提示</summary>

[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)开头
的数字推演给出上限估算法：环境侧产能与合批推理上限分开算，谁小谁是
瓶颈；等待与合批的交换见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)的
「Batchable 与 batch 的语义」。对每个 `autobatch_size` 记录实际
batch 的分布而不只均值：活跃请求凑不满而触发 flush 时，名义与实际
batch 会分离。

</details>

3. 【挑战】比较 CPU environment + GPU policy 与同设备 environment/policy，记录
   observation/action copy 和同步边界。

<details>
<summary>提示</summary>

[「MDP、环境与轨迹边界」](01-mdp-environment-and-trajectory.md)的
「成本模型」解释了小 observation 的往返搬运可能吃掉 kernel 收益；
转换发生的位置见
[「Policy、观察转换与动作批处理」](02-policy-and-batching.md)中
`to_observation` 的 device 参数。测量时把 host/device copy 单独
计时，并固定 observation shape 与 batch 大小，才能把差异归因到传输
而不是模型本身。

</details>

4. 【挑战】实现一个带序号的多 worker trajectory reorder layer，验证 worker 返回
   顺序变化时 episode 内 step 不乱序。

<details>
<summary>提示</summary>

要守住的性质列在
[「Rollout 吞吐、异步环境与推理队列」](04-rollout-throughput.md)
「固定 Burn 的三种 runner」的异步验证清单里：每个环境的 transition
按自身顺序出现、`env_id` 能把结果路由回正确环境。重排机制可借鉴
第 5 章[「多线程加载与保序性边界」](../ch05/05-multithread-and-order.md)
的“附加全局序号、在消费者侧重排”方案；测试时故意打乱 worker 的
返回顺序，断言每个 episode 内 step 序号单调。

</details>

5. 【挑战】为 replay 加入 n-step return 或 prioritized index，列出它改变的
   memory、sampling bias 和 checkpoint 字段。

<details>
<summary>提示</summary>

[「Transition、回放与采样」](03-replay-and-sampling.md)给出两个
支点：内存估算式把容量与每字段字节数联系起来；「本节小结」明确
优先级采样与 n-step 不在固定 `TransitionBuffer` 的能力内，这是一道
自行扩展设计题。从“`sample` 的所有字段共用同一 indices”这个
不变量出发，推 n-step 需要额外读哪些相邻行、优先级需要在 push、
sample 和 checkpoint 各加什么状态。

</details>

6. 【挑战】设计 Actor–Learner 的 policy version/trajectory schema、queue 上限、
   retry、duplicate detection 和 checkpoint 恢复协议。

<details>
<summary>提示</summary>

[「多智能体与分布式系统边界」](06-multi-agent-boundary.md)的
「与分布式训练的关系」列出了六条应当先写下的可验证条件，可以直接
当设计文档的提纲；`examples/ch08-rl-rollout` 的 `policy_is_fresh`
测试是版本滞后判断的最小原型。记住固定 Burn 只提供单进程编排，
跨节点部分全由你的协议承担——包括队列满时阻塞、丢弃还是覆盖的
选择。

</details>

7. 【挑战】为两名 agent 的剪刀-石头-布实现联合 action 和 reward vector，说明
   单独复制单智能体 `Policy` 为什么不能解决 non-stationarity。

<details>
<summary>提示</summary>

对照[「多智能体与分布式系统边界」](06-multi-agent-boundary.md)：联合
transition 需要同时表达所有 agent 的动作与奖励；非平稳性来自其他
agent 的策略变化，不是网络拓扑或调度顺序问题。
`examples/ch08-rl-rollout` 的 `joint_transition` 给出了联合动作/奖励
向量的最小表示，可从它出发设计 schema，再论证对手策略变化时，
单智能体 replay 打破了哪条假设。

</details>


## 延伸阅读与固定源码入口

教材（Sutton & Barto）与 DQN、PPO、Ape-X、IMPALA、Ray 等论文见附录
[参考文献](../references.md#第-8-章-强化学习系统)。
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
