# 8.1 MDP、环境与轨迹边界

## 从决策问题开始

强化学习的最小闭环不是“输入和标签”，而是一个反复发生的决策过程：

```text
s_t ── policy ──► a_t ── environment ──► s_{t+1}, r_t
 ▲                                             │
 └─────────────────────────────────────────────┘
```

常用的马尔可夫决策过程（Markov Decision Process，MDP）写成
$(\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \gamma)$：

- $\mathcal{S}$ 是环境状态空间；
- $\mathcal{A}$ 是动作空间；
- $\mathcal{T}(s'|s,a)$ 是状态转移分布；
- $R(s,a)$ 是奖励；
- $\gamma \in [0,1]$ 是折扣因子。

策略 $\pi(a|s)$ 产生动作，目标通常是最大化折扣累计回报：

$$
J(\pi)=\mathbb{E}_{\pi,\mathcal{T}}
  \left[\sum_{t=0}^{H-1}\gamma^t r_t\right].
$$

价值函数把“策略长期有多好”与一次 step 的 reward 区分开：

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s],\qquad
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a],
$$

其中 return $G_t=\sum_{k=0}^{H-1}\gamma^k r_{t+k}$。这带来两类常见
更新路径：

- **Monte Carlo** 等 episode 结束后，用完整 $G_t$ 估计价值；方差较大，
  但不需要 bootstrap；
- **Temporal Difference (TD)** 在 episode 中途用下一状态估计未来：
  $r_t+\gamma V(s_{t+1})$ 或
  $r_t+\gamma\max_a Q(s_{t+1},a)$；更新及时，但会引入估计偏差。

`done` 通常令 TD target 截止；`truncated` 是否截止取决于环境和算法
协议。这个选择应在 transition schema 中保留，不能只由 buffer 的
`dones` shape 推断。

这里的“状态”是环境为了决定下一步而维护的内部信息，而“观察量”
（observation）是策略实际收到的表示。完全可观测环境中观察量可能足够
恢复状态；部分可观测环境中，策略还需要历史、记忆或 belief state。
不能因为 Rust 结构叫 `State`，就默认它已经是策略的完整输入。

## 一步、episode 与轨迹

一次环境 step 至少有五个语义字段：

```text
(state, action) → (next_state, reward, terminal)
```

连续 step 组成 episode；一个按时间顺序排列的 episode 片段是轨迹
（trajectory）。系统必须明确 episode 结束的原因：

- `done`：任务本身已经结束，例如到达目标或失败；
- `truncated`：达到时间上限、采样预算或外部截断，但任务不一定自然结束。

二者在 bootstrap 时可能有不同含义。自然 terminal 通常不再使用
$\max_a Q(s',a)$；时间截断是否 bootstrap 则取决于算法和环境协议。如果
系统把两个标志过早合并，就失去了之后修正这个选择的机会。固定 Burn
训练 runner 在写入 replay transition 时将它们合并为 `done` tensor，
所以环境 adapter 仍应保留原始字段并在文档中写清楚这一损失。

## Burn 的环境 trait

固定 `burn-rl` 用关联类型把环境的状态和动作写进实现：

```rust
{{#include ../../../examples/ch08-rl-rollout/src/lib.rs:environment}}
```

`Environment` 本身不要求状态是 tensor，也不要求环境是随机的。它只定义
`state`、`step`、`reset` 和 `MAX_STEPS`。这带来两个重要的工程边界：

1. 环境可以用 Rust 值表达物理状态、模拟器句柄或压缩 observation；
2. policy 如何消费该状态，要通过 `ToObservation<O>` 显式转换，而不是
   在环境 trait 内偷偷绑定某个神经网络。

`EnvironmentInit<E>` 负责创建环境实例。固定实现允许一个 `Fn() -> E`
直接充当 initializer，因此多个 rollout worker 可以各自拥有一个可变
环境；它们不需要共享同一个 `&mut E`。这是 Rust 所有权在系统设计上的
直接收益：每个 worker 的环境状态有明确 owner，跨线程传递的是消息或
可 clone 的配置，而不是未受保护的共享可变状态。

## 成本模型

一次 rollout step 的墙钟时间可以拆成：

$$
T_{\text{step}} =
T_{\text{env}}+
T_{\text{to-observation}}+
T_{\text{policy}}+
T_{\text{to-action}}+
T_{\text{record}}+
T_{\text{queue}}.
$$

其中 `T_env` 可能来自 CPU 仿真，`T_policy` 可能来自 GPU inference；
中间的 conversion、host/device copy 和 queue wait 在小模型上反而可能
占主导。总训练吞吐不等于 policy 的单次 forward 吞吐：

$$
\text{steps/s} =
\frac{\text{完成的 environment steps}}
{\text{environment、inference、传输和等待的总时间}}.
$$

因此“把模型搬到 GPU”并不自动加速 RL。如果环境仍在 CPU，且每一步都把
一个小 observation 往返搬运，`T_copy + T_queue` 可能抵消 kernel 的
收益。后面的异步环境和 batching 只是在这些项之间重新安排工作，不改变
它们的存在。

## 本节小结

MDP 给出决策问题的数学边界，`Environment` 给出一条可组合的 Rust
接口；轨迹则是将多个 step 按顺序保存的系统数据结构。写 rollout 时，
先确定 state/observation、done/truncated 和 reset 协议，再讨论模型或
并行度，才能避免“能调用 forward”与“数据语义正确”之间的混淆。
