# 8.3 Transition、回放与采样

## Transition 是训练数据协议

一个最小 transition 写作：

$$
\tau_t=(s_t,a_t,r_t,s_{t+1},d_t).
$$

其中 $d_t$ 表示该 step 是否应切断后续 bootstrap。它不是“日志里的几列
数值”这么简单，因为 learner 还需要知道：

- `s_t` 与 `s_{t+1}` 是否属于同一个 observation schema；
- action 是离散索引、连续向量还是结构化 Rust 值；
- reward 的 dtype、缩放和累计范围；
- `d_t` 是自然 terminal、时间截断，还是二者已合并；
- 这条 transition 由哪个行为 policy 版本产生。

缺少这些元数据时，replay 仍然可以返回 shape 正确的 batch，却可能训练
出语义错误的 target。工程上应把环境版本、policy version、seed 和
normalization 规则看作数据协议的一部分。

固定 `burn-rl` 的 `Transition<S, A>` 保存 state、next_state、action 和
一维 reward/done tensors；`TransitionBatch<SB, AB>` 则将 observation/
action 作为批量类型，reward/done 变为二维 tensor。它没有要求 state 或
action 必须是 tensor，因此环境原始 transition 可以先使用结构化 Rust
类型，再转换成 learner 表示。

## Circular replay buffer

`TransitionBuffer<SB, AB>` 是一个有限容量、tensor-backed 的环形 buffer：

```text
write_head
    │
    ▼
[ t_4 ][ t_5 ][ t_2 ][ t_3 ]   capacity = 4
                    ▲
                oldest data
```

固定实现的行为是：

1. 构造时只保存 capacity、device 和空 storage；
2. 第一次 `push` 根据 `SliceAccess::zeros_like` 分配 state/action/reward/
   done 存储；
3. 每条新 transition 写入 `write_head % capacity`；
4. 满后覆盖最旧内容；
5. `sample(batch_size)` 用 tensor random 生成第 0 维 indices，再对所有
   字段使用相同 indices。

因此 replay 的容量是状态内存预算，不是“最多训练多少次”的预算。粗略的
内存估算为：

$$
M_{\text{replay}}\approx C\cdot
(\operatorname{bytes}(s)+\operatorname{bytes}(s')
+\operatorname{bytes}(a)+\operatorname{bytes}(r)+\operatorname{bytes}(d)).
$$

真实值还要加上 backend storage、alignment、autodiff 或 device copy 的
开销。增大 capacity 可能改善样本多样性，却会增加内存和旧 policy 数据
的比例；它不是无条件的算法改进。

## `SliceAccess` 与 Rust 泛型

`TransitionBuffer` 不把 observation 写死成 `Tensor<2>`。它只要求
`SB`/`AB` 实现 `SliceAccess`：

- `zeros_like`：按一个样本的 shape 创建 capacity 行；
- `select`：按 indices 选 batch；
- `slice_assign_inplace`：把一个样本写入第 0 维的某一行。

固定快照为 `Tensor<2>` 提供了实现；上游 DQN example 则为自己的
`ObservationTensor<2>` 和 `DiscreteActionTensor<2>` 实现同一接口，并在
写入时处理 autodiff/device 的 inner tensor。这展示了 Rust 泛型的价值：
replay 只知道“可批量切片”，而不需要知道具体网络的字段。

同时，这个接口也暴露了实现者的责任。不同维度、不同 device 或
autodiff/non-autodiff tensor 之间如何转换，不会被 `SliceAccess` 自动
解决。应用必须在 `zeros_like`、`slice_assign_inplace` 和 sample 后的
forward 上分别测试。

## Online 与 off-policy

如果用于更新的 policy 与刚刚采样的行为 policy 相同，通常称为 on-policy
路径；如果 replay 里混有较早 policy 产生的数据，通常称为 off-policy
路径。这个分类不是由“有没有一个 buffer”单独决定的：on-policy 算法也
可能保留一个短暂 batch，关键在数据与更新 policy 的关系。

replay 会引入分布偏移：

$$
\mathcal{D}_{\text{replay}}\ne
\mathcal{D}_{\text{current policy}}.
$$

Q-learning 可以通过 bootstrap 使用这类数据，但 policy gradient 通常还
需要 importance ratio、近端约束或其他校正。Burn 的 buffer 只负责存取，
不会替应用判断算法是否允许这样的数据分布。

固定实现还有两个必须写进配置校验的边界：

- `sample` 的 batch size 不能大于当前 `len`，否则会 panic；
- capacity 为零时第一次 push 会在取模处失败，不是有效配置。

本章实验在调用前返回描述性错误，把底层容器的前置条件变成应用 API
的一部分。生产训练系统还应考虑优雅 shutdown、数据版本、优先级采样、
n-step return、跨进程共享和 checkpoint 一致性；这些都不由当前
`TransitionBuffer` 自动提供。

## 本节小结

Replay 的核心不是“随机打乱”，而是一个有容量、有覆盖策略、有数据语义
的训练协议。`TransitionBuffer` 已验证的能力是单进程 tensor 环形存储
和随机 batch；优先级、n-step、跨节点一致性和 policy version 管理仍属于
上层系统设计。
