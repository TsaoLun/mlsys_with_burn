# 实验：CPU 确定性 rollout 与 replay

## 你会学到什么

示例在 `examples/ch08-rl-rollout`：用没有外部 simulator 的一维
`CounterEnv`，先连续右移触发自然终止，再在后续回合走到时间截断。实验
分成两个阶段，回答两个不同的问题：

```text
阶段 A（在线）
  CounterEnv ──step──► transition ──► 在线 TD 更新（逐条立即学习）
                          │
                          ▼
                  TransitionBuffer ──sample──► shape 检查

阶段 B（回放驱动）
  CounterEnv ──step──► TransitionBuffer ──random sample──► TD 更新
                       （先只收集，learner 再从回放批学习）
```

阶段 A 问“环境边界与表格 TD 是否可观察”；阶段 B 问“当 learner 的数据
来自回放而不是环境当下时，同一个算法会看到什么”。你会观察到：`done`
与 `truncated` 分别出现、circular buffer 容量、replay batch 的 shape、
终止转移不做 next-state bootstrap，以及同一个 Q 值在两条数据路径下走向
不同结果。本实验刻意不做神经网络 forward、autodiff、DQN 收敛、gym、
GPU 仿真、多环境异步或多智能体通信——先把机制看清楚，再放回更大系统。

## 1. 实现环境边界

环境通过`burn-rl::Environment` trait 表达状态、动作和 step 结果：

```rust,ignore
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:environment}}
```

`CounterEnv` 在位置达到 2 时返回 `done = true`；如果四步内没有到达，
第四步返回 `truncated = true`。主 rollout 在任一标志出现后调用 `reset`。
测试分别走 `Left × 4` 观察截断、走 `Right × 2` 观察自然终止；主程序使用
确定性动作序列，因此一次运行能同时报告两类边界。

注意写入 replay 时的语义合并：`TransitionBuffer::push` 只收一个
`done: bool`，示例传入的是 `result.done || result.truncated`。因此回放
数据里看不出两种结束的区别，截断的 transition 在 replay 学习中也会被
当作终止而停止 bootstrap。需要区分两者的算法必须在写入前自行保留原始
标志——这也是
[TD 更新、off-policy 与训练编排](05-learning-and-off-policy.md)
讨论编排边界时要求你核对的内容。

## 2. 把 transition 放入 replay

示例把 state 编成 `[position, step]`，把 action 编成一列 `-1/1`。这个
编码只是实验协议，不是通用 observation 设计。`TransitionBuffer` 的
state/action 类型是 `Tensor<2>`，因此 Burn 已经有对应的
`SliceAccess` 实现。

调用 replay 前先验证应用配置：

- `steps > 0`；
- `capacity > 0`；
- `sample_size > 0`；
- `sample_size <= min(steps, capacity)`。

这样不会让底层 `sample` 的 panic 成为用户输入错误的唯一反馈。

## 3. 观察在线 TD target

实验使用的目标函数是：

```rust,ignore
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:td_target}}
```

`done` 时 target 只有 reward；非 terminal 时才加入
`gamma * next_max_q`。主循环用三个位置、两个动作的表格保存 Q 值，在每条
transition 产生时在线应用：

$$
Q(s,a)\leftarrow Q(s,a)+
\alpha\left(y-Q(s,a)\right).
$$

它不是 Burn optimizer 的替代品，而是一个可以在不引入网络的情况下观察
bootstrap 的对照实现。若把它改成 DQN，需要将 `q_values` 替换为
`Module`，用 `gather` 选择 action value，计算 loss，调用 backward 和
optimizer，并另外维护 target network；上游 DQN example 展示了这条扩展。

## 4. 用 replay batch 驱动更新

阶段 A 里 replay 只是被写入和抽查，learner 的数据仍来自环境当下。把
sample 得到的 batch 真正接回更新，才构成 off-policy 的数据路径：learner
学习的不再是“刚刚发生”的 transition，而是 capacity 窗口内的一个随机
子集。

TD 公式不变，变的是数据分布：

- 在线路径按环境产生顺序逐条使用 transition，每条最多用一次；
- 回放路径把最近 `capacity` 条 transition 作为采样池，顺序被打乱，
  旧数据会被新数据覆盖，同一条 transition 可能被重复学习。

示例的第二阶段先只收集、不更新，再对 buffer 做若干轮“随机采样 → 更新”：

```rust,ignore
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:replay_update}}
```

为了让“容量决定 learner 能看到什么”成为精确观察而不是比喻，主程序用
`capacity = 1` 跑了一个极端对照：环形 buffer 只保留最新一条
transition，只有一个元素时 sample 完全确定。此时 8 轮更新全部落在最后
一条 transition（state `[1, 3]` 上的 `Left`）上，初始状态的
`initial_right_q` 精确保持为 0——这不是 TD 公式失效，而是 learner 的
数据分布被容量截断了。同一个环境序列，在线路径却学出了 1.2125。

把容量放宽到 6（保留全部 6 条）后，随机采样能让初始状态被学到，
但具体数值随抽样顺序变化。回放既去除了相邻 transition 的相关性，也
引入了采样方差——这两个效应在这个玩具表格上已经能直接看到。

## 5. 运行完整实验

在项目根目录运行：

```bash
cargo test -p ch08-rl-rollout --locked
cargo run -p ch08-rl-rollout --locked
```

主程序会打印类似：

```text
phase=online transitions=6 buffer_len=4 done_transitions=1 truncated_transitions=1 \
state_shape=[2, 2] action_shape=[2, 1] reward_shape=[2, 1] done_shape=[2, 1] \
initial_right_q=1.2125
phase=replay capacity=1 sample=1 rounds=8 updates=8 buffer_len=1 initial_right_q=0.0000
phase=replay capacity=6 sample=2 rounds=10 updates=20 buffer_len=6 initial_right_q=...（随机采样，数值会变）
```

前两行在固定源码版本下是确定的：在线路径的 `initial_right_q=1.2125`
大于 1，是因为初始 `Right` 先在自然终止前得到 reward 1，reset 后同一
状态/动作又经历了一次含 bootstrap 的非终止更新——同一条物理转移在不同
episode 位置可以产生不同 target。回放路径在 `capacity = 1` 时精确为 0，
原因见上一节。第三行使用随机采样，只应断言它有限且非负，不应写成固定
对照值；改变 gamma、learning rate、环境或 action schedule 时，也应同时
更新前两条的对照值和解释。

## 6. 扩展路径

可以按以下顺序扩展：

1. 把交替 action 换成一个显式 policy，实现 `ToObservation`、`ToAction`
   和 `Policy`；
2. 为 observation 增加 batch/unbatch 测试，接入 `AsyncPolicy` 的 mock；
3. 将表格 Q 替换为 Burn `Module`，使用 `AutodiffModule` 和 optimizer；
4. 保存 policy、target、optimizer、exploration step 的组合 record；
5. 用 `OffPolicyConfig` 的字段模拟 collect/train/evaluate cadence；
6. 增加多个 deterministic environment，测量 queue wait 和 batch size；
7. 最后再接入 gym 或 GPU simulator，并单独记录平台前提和 benchmark 方法。

每一步都应保持 environment correctness、replay correctness 和 learner
correctness 的测试分层。一个高 reward 不能替代对 reset、terminal、
policy version 和 checkpoint 的验证。
