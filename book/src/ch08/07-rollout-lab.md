# 8.7 实验：CPU 确定性 rollout 与 replay

## 实验目标与边界

实验位于 `examples/ch08-rl-rollout`。它实现一个没有外部 simulator 的
一维 `CounterEnv`，交替执行右移/左移动作，将环境 step 编码为二维
observation/action tensor，写入固定 Burn 的 `TransitionBuffer`，再运行
一个小型表格 TD 更新。

```text
CounterEnv
   │ Environment::step
   ▼
state/action tensor
   │
   ▼
TransitionBuffer(capacity = 4)
   │ random sample(batch = 2)
   ▼
shape assertions + tabular TD update
```

实验验证：

- `done` 与 `truncated` 的 episode 边界；
- rollout step 数与 circular buffer 的容量上限；
- replay batch 的 state/action/reward/done shape；
- terminal transition 不进行 next-state bootstrap；
- 一个可观察的 Q 值确实被 TD 更新改变。

实验不验证神经网络 forward、autodiff、DQN 收敛、gym 环境、GPU 仿真、
多环境异步吞吐或多智能体通信。它的目的与第 6 章 CPU 训练实验相同：
先隔离一个可测的机制，再把机制放回更大的系统。

## 1. 实现环境边界

环境通过固定 `burn-rl::Environment` trait 表达状态、动作和 step 结果：

```rust
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:environment}}
```

`CounterEnv` 在位置达到 2 时返回 `done = true`；如果四步内没有到达，
第四步返回 `truncated = true`。主 rollout 在任一标志出现后调用 `reset`。
这让测试可以同时观察自然终止和时间截断，而不是只测一个布尔字段。

## 2. 把 transition 放入 replay

示例把 state 编成 `[position, step]`，把 action 编成一列 `-1/1`。这个
编码只是实验协议，不是通用 observation 设计。`TransitionBuffer` 的
state/action 类型是 `Tensor<2>`，因此固定快照已经有对应的
`SliceAccess` 实现。

调用 replay 前先验证应用配置：

- `steps > 0`；
- `capacity > 0`；
- `sample_size > 0`；
- `sample_size <= min(steps, capacity)`。

这样不会让底层 `sample` 的 panic 成为用户输入错误的唯一反馈。

## 3. 观察 TD target

实验使用的目标函数是：

```rust
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:td_target}}
```

`done` 时 target 只有 reward；非 terminal 时才加入
`gamma * next_max_q`。主循环用三个位置、两个动作的表格保存 Q 值，再
应用：

$$
Q(s,a)\leftarrow Q(s,a)+
\alpha\left(y-Q(s,a)\right).
$$

它不是 Burn optimizer 的替代品，而是一个可以在不引入网络的情况下观察
bootstrap 的 reference。若把它改成 DQN，需要将 `q_values` 替换为
`Module`，用 `gather` 选择 action value，计算 loss，调用 backward 和
optimizer，并另外维护 target network；固定 DQN example 展示了这条扩展。

## 4. 运行完整实验

在项目根目录运行：

```bash
cargo test -p ch08-rl-rollout
cargo run -p ch08-rl-rollout
```

主程序会打印类似：

```text
transitions=6 buffer_len=4 terminal_transitions=1 \
state_shape=[2, 2] action_shape=[2, 1] reward_shape=[2, 1] done_shape=[2, 1] \
initial_right_q=0.9762
```

sample 的 index 是随机的，所以不要断言某一行一定被抽到；本实验断言
的是 batch shape 和 buffer 长度。环境、action sequence 和 TD update
本身是确定的，因此 `initial_right_q` 在固定源码下可作为回归观察值；
如果改变 gamma、learning rate、环境或 action schedule，应同时更新
reference 和解释。

## 5. 扩展路径

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
