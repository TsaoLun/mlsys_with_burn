# 8.4 Rollout 吞吐、异步环境与推理队列

## 采样系统的瓶颈

强化学习常见的系统不平衡是：环境产生数据太慢，或者 learner 更新太
慢，导致另一侧等待。单环境同步循环可以写成：

```text
read state → policy action → env.step → record transition → repeat
```

它容易理解，却把环境和 inference 串行化。若单步时间分别为
$T_{\text{env}}$ 和 $T_{\text{policy}}$，单环境吞吐近似为：

$$
\text{throughput}_{1}\approx
\frac{1}{T_{\text{env}}+T_{\text{policy}}+T_{\text{transfer}}}.
$$

创建 $N$ 个环境并行采样后，理想上可以接近
$N/T_{\text{env}}$，但实际吞吐受 CPU 核数、锁、内存带宽、policy batch
上限、queue wait 和设备传输限制。多环境不是免费加速器。

## 固定 Burn 的三种 runner

固定 `burn-train` 的 RL 代码提供三层 runner：

1. `AgentEnvBaseLoop` 在当前线程顺序执行一个环境；
2. `AgentEnvAsyncLoop` 把一个环境放到线程中，用 channel 请求 step 或
   episode，并把 `TimeStep`/`Trajectory` 传回；
3. `MultiAgentEnvLoop` 这里的 “agent” 实际上是多个环境接口，它创建
   多个环境线程，并把它们连接到一个 `AsyncPolicy`。

第三个类型名容易造成误读：它是多环境 rollout runner，不是 OpenMLSys
意义上的多智能体博弈系统。多个环境可以共享一个 policy 的 inference
server，却没有因此获得联合动作、通信、团队奖励或 equilibrium solver。

`AgentEnvAsyncLoop` 还使用了 double batching：环境线程保持一步领先，
主线程消费一条 transition 后再发出下一条请求。这样 policy server 可以
在多个环境之间合批，但也增加了 channel 和状态机的复杂度。任何异步
设计都应该先验证：

- 每个环境的 transition 是否仍按自身顺序出现；
- `env_id` 是否能将返回结果路由回正确环境；
- reset 后的第一条 state 是否没有混入上一个 episode；
- interrupt/shutdown 时是否会释放等待中的 sender/receiver。

## OffPolicyStrategy 的执行顺序

固定 `burn-train` 的 `OffPolicyStrategy` 把配置映射为一条循环：

```text
MultiAgentEnvLoop::run_steps(train_interval)
        │
        ▼
 state/action 转换 + TransitionBuffer::push
        │
        ├─ buffer.len >= train_batch_size
        │    且达到 warmup_steps
        │
        └─ sample(train_batch_size)
                │
                ▼
       PolicyLearner::train × train_steps
                │
        evaluation / checkpoint / metrics
```

`OffPolicyConfig` 的 `num_envs`、`autobatch_size`、`replay_buffer_size`、
`train_interval`、`train_steps`、`eval_interval`、`eval_episodes`、
`train_batch_size` 和 `warmup_steps` 共同决定这个循环。它们不能只看
单个参数：

- 增大 `train_interval` 可以减少调度开销，却让 replay 更新更不频繁；
- 增大 `train_steps` 可能让 learner 追上采样，也可能让行为 policy 更旧；
- `warmup_steps` 小于 batch size 时，实际仍需先积累足够 replay；
- evaluation 会用独立的 inference runner，并采用配置的 deterministic
  行为；
- inference device 与 learner device 可以不同，转换实现必须处理这条
  边界。

固定源码中的 `RLTraining` 还连接 metrics、renderer、event processor、
interrupter 和 checkpoint。它负责训练过程的编排，不替 `PolicyLearner`
决定 loss 或 optimizer。

## 吞吐与延迟的测量

采样服务至少要分别记录：

```text
environment step time
policy queue wait
batched forward time
state/action conversion time
replay push/sample time
learner update time
```

平均值不足以描述交互系统。环境偶发慢一步时，队列头部请求可能出现
长尾；因此应报告 steps/s、batch size、queue wait 的 p50/p95/p99 和
evaluation latency。比较两个实现时还需固定环境数、episode 长度、
observation shape、backend、dtype、设备和随机策略。

本地 CPU 实验选择确定性小环境，目的是验证顺序和数据形状，而不是制造
有意义的 simulator benchmark。若换成 GPU 仿真或真正游戏环境，应另外
记录环境版本、并行方式、CPU/GPU 拓扑、同步点和 warmup。

## 可复现性

RL 的随机性至少来自环境转移、policy exploration、replay indices 和
worker 调度。固定 seed 只能控制显式使用该 RNG 的部分；线程时序、设备
kernel 或异步 queue 仍可能影响结果。一个可恢复的实验需要保存：

- environment 与 observation schema 版本；
- policy/model/optimizer record；
- exploration schedule 当前 step；
- replay 是否保存、保存到哪个 transition；
- sampler/worker seed；
- hyperparameters 和代码 revision。

固定 `burn-rl::AsyncPolicy` 使用 native channel/thread，固定 replay
使用 tensor random；它们提供机制，但没有替应用建立跨 worker 的统一
随机种子协议。

## 本节小结

异步 runner 的价值是重叠环境、推理和 learner 的工作，代价是 queue、
顺序、reset、shutdown 和复现协议。`burn-train` 已提供单环境、多环境、
off-policy 编排入口；系统仍需根据真实 simulator 和硬件测量，而不能
从 API 名称推导吞吐结论。
