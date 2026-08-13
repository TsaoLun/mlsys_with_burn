# Policy、观察转换与动作批处理

## Policy 不是算法

一个 policy 可以表示动作分布、确定性函数、带探索的包装器，甚至一个
调用远端 inference 服务的客户端。它不等于 DQN、PPO 或某种损失函数。
`burn-rl` 的 `Policy` trait 把最小职责拆成：

```text
forward(observation batch) → action distribution
action(observation batch, deterministic) → (action batch, contexts)
update(policy state)
state / load_record / to_device
```

`forward` 适合 learner 需要分布参数或 logits 的场景；`action` 负责把
分布变成真正送给环境的动作，并可以返回每个动作的 context，例如
epsilon、log probability、随机数种子或 value estimate。context 不是环境
动作本身，却是训练 metric 和 loss 可能需要的旁路数据。

`deterministic` 是一个明确的调用参数。评估通常要求 greedy/deterministic
行为，训练通常允许 exploration；把随机性隐藏在环境或全局变量里，会让
评估结果难以复现，也会让 replay 中的行为策略无法解释。

## 三种表示之间的转换

Burn 把环境与 policy 之间的转换分成两个 trait：

```text
Environment::State ── ToObservation<O> ──► Policy::Observation
Environment::Action ◄─ ToAction<A> ─────── Policy::Action
```

`ToObservation<O>::to_observation(&self, device)` 可以在转换时把 tensor
放到指定 Device；`ToAction<A>` 则把 policy 输出变成环境能接受的类型。
这比让 `Environment` 直接返回 `Tensor<2>` 更有边界感：同一个环境可以
接表格 policy、CPU policy 或 GPU policy，只要各自提供转换实现。

Burn 的 `Policy` 使用关联类型定义 `Observation`、`ActionDistribution`、
`Action`、`ActionContext` 和 `PolicyState`。这会带来较多 trait 约束，但
也让以下错误尽可能在编译期暴露：

- 把错误形状或错误类型的 observation 送给 policy；
- 将一个 policy 的 action 直接当作另一个环境的 action；
- 忘记为多环境 runner 实现 batch/unbatch；
- 从不同 policy state record 恢复不兼容的 policy。

## Batchable 与 batch 的语义

多环境采样经常同时获得多个 observation。`Batchable` 要求类型实现：

- `batch(Vec<Self>) -> Self`：把多个单样本合并；
- `unbatch(self) -> Vec<Self>`：把模型结果拆回各个环境。

这个 trait 不规定 tensor 的布局。对于二维 observation，常见布局是
`[batch, features]`；对于图像或序列，第一维也许只是 batch 轴。实现
`Batchable` 时必须把“第 0 维是样本维”写进测试，而不能只依赖一次
`cat` 恰好成功。

`PolicyInferenceServer`/`AsyncPolicy` 在源码中利用这个接口做进程内
自动 batching：请求先经 mpsc channel 进入 server，server 收集到足够
请求后调用一次 `action` 或 `forward`，再把结果按请求顺序发送回去。
活跃 agent 数小于最大 batch 时，server 也有 flush 路径，避免最后几个
环境永久等待。

这是一种吞吐与延迟的交换：

$$
T\_{\text{request}} =
T\_{\text{queue-wait}}+
T\_{\text{batched-inference}}+
T\_{\text{unbatch}}+
T\_{\text{return}}.
$$

增大 batch 可能提高设备利用率，但会增加最早到达请求的等待时间，并且
改变随机 policy 的调用粒度。需要同时记录 batch size、queue wait、
steps/s 和 p95/p99，而不是只看一次 forward 的平均时间。

## Policy state 与 checkpoint

`PolicyState` 只要求一个关联的 `Record` 类型以及
`into_record`/`load_record`。神经网络 policy 可以使用 `ModuleRecord`；
epsilon schedule 的 step、normalizer、recurrent hidden state 或 target
network 是否保存，则由应用的 `PolicyState` 决定。

`burn-rl::AsyncPolicy::load_record` 直接标记为未实现，并要求先在
inner policy 上加载 record，再创建 async wrapper。这是一个有用的边界：
线程化 inference wrapper 不应该偷偷决定模型恢复协议。启动服务或
训练进程时，先完成 topology/config/record 的兼容性检查，再把可运行
policy 交给异步层。

`AsyncPolicy` 还是 native std 线程和 mpsc 的实现，不是跨机器 RPC。它的
`Clone` 是共享发送端的句柄；真正的 policy 状态由 server 线程独占。这个
所有权安排避免多个环境线程同时修改 module，但也要求 update、device
切换和 action 请求服从同一消息顺序。

## 本节小结

`Policy` 描述“如何从 observation 产生 action”，`PolicyLearner` 描述
“如何更新 policy”，`Environment` 描述“动作如何改变世界”。通过
`ToObservation`、`ToAction` 和 `Batchable` 连接三者，Rust 类型系统把
表示边界显式化；通过 `AsyncPolicy` 可以在单进程内做 batching，但仍需
由应用设计 checkpoint、网络协议和失败恢复。
