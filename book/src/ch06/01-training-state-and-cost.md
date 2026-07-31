# 训练状态、迭代与成本模型

## 训练系统维护什么状态

设模型参数为 $\theta$，一个 batch 为 $B_t$，训练目标为
$L(\theta; B_t)$。一次最小训练迭代可以写成：

$$
y_t = f_{\theta_t}(B_t),\qquad
g_t = \nabla_{\theta_t} L(y_t, B_t),\qquad
\theta_{t+1} = U(\theta_t, g_t, s_t, \eta_t),
$$

其中 $s_t$ 是优化器状态，$\eta_t$ 是当前学习率，$U$ 是更新规则。SGD
可以没有额外的动量状态，而 Adam 至少还要维护一阶和二阶矩。于是“模型
参数”并不是训练状态的全部：

```text
训练状态
├── Module parameters θ
├── optimizer state s
├── learning-rate scheduler state
├── epoch / iteration / sampler position
├── metric and early-stopping state
└── checkpoint / RNG / data-shard protocol
```

只保存 $\theta$ 而丢失 $s_t$，恢复后得到的是“从同一参数重新开始”，不是
优化轨迹的精确延续。只保存一个随机 seed，也不能自动恢复已经消耗了多少
个随机数、当前 epoch 的 sampler 位置或多设备 shard。

## 正确性先于加速

一个训练系统至少要保持三种不变量：

1. **梯度语义**：更新使用的是当前参数和当前 batch 对应的梯度；
2. **状态一致**：需要同步的模型副本在下一步使用兼容的参数版本；
3. **进度可解释**：epoch、iteration、metric 和 checkpoint 编号彼此对应。

并行化只是在这些不变量成立的前提下减少墙钟时间。比如数据并行中，设备
$i$ 对本地 batch 计算 $g_i$，同步 SGD 通常需要：

$$
\bar{g} = \frac{1}{\sum_i n_i}\sum_i n_i g_i,
$$

其中 $n_i$ 是设备 $i$ 的样本数。只有在每个局部梯度已经按本地样本数
正确归一化，且 collective 的 `Mean` 语义与目标 batch 一致时，简单平均
才等价于单设备的大 batch。最后一个 batch、drop-last 和不同 shard 大小
会改变这个等价关系。

## 时间和内存成本

单设备的一步可以粗略写为：

$$
T_{\text{step}} =
T_{\text{load}} + T_{\text{forward}} + T_{\text{backward}}
+ T_{\text{update}} + T_{\text{metric}}.
$$

数据并行增加设备后，理想的计算项近似除以设备数，但同步项会增加：

$$
T_{\text{parallel}} \approx
\max_i(T_{\text{load},i}+T_{\text{compute},i})
+ T_{\text{collective}} + T_{\text{wait}}.
$$

`max` 很重要：同步训练通常等待最慢的设备，落后者（straggler）会把局部
加速转成全局等待。模型更大时，通信 tensor 的字节数、设备间拓扑和
collective 算法都会进入 `T_collective`。

内存也会随训练状态增加。粗略地说：

$$
M_{\text{train}} =
M_{\text{parameters}} + M_{\text{gradients}}
+ M_{\text{optimizer}} + M_{\text{activations}}.
$$

梯度累积可以在不立刻增加单次设备 batch 的情况下增大有效 batch，但它会
延后 optimizer step，并保持多次 backward 产生的梯度。梯度 checkpointing
则以重新计算换峰值内存；这两个机制都不能自动解决数据读取或跨节点带宽
瓶颈。

## 同步与异步

同步训练的一个轮廓是：

```text
各设备 forward/backward
        │
        ├── AllReduce / 梯度聚合
        │
        └── 各设备使用同一更新结果 → 下一步
```

它通常更容易解释收敛和 checkpoint，但会暴露慢设备与通信延迟。参数服务器
或其他异步协议可以让较快 worker 不等待较慢 worker，却引入 stale gradient：
梯度可能由旧版本 $\theta_{t-k}$ 计算。异步系统需要额外定义版本、冲突、
停止和恢复语义，不能仅把同步循环中的 `wait` 删除。

## 对 Burn 的定位

OpenMLSys v1 `overview.md` 和 `methods.md` 使用“切分—并行—合并”的
分而治之框架。本书保留这个框架，但把它映射到固定 Burn 的明确层次：

- 数据进入 batch 的边界在第 5 章 `DataLoader`/`Batcher`；
- forward/backward 与 `GradientsParams` 属于 autodiff 和 `TrainStep`；
- 参数更新与 optimizer state 属于 `ModuleOptimizer`；
- metric、epoch、checkpoint 和策略装配属于 `burn-train`；
- AllReduce 需要 backend 的 `DistributedOps`，不是普通 `Tensor` API
  自动拥有的跨节点能力。

因此，后续看到 `ExecutionStrategy::MultiDevice` 或 DDP 时，应先问它维护的
状态是什么、谁负责等待、谁负责保存，而不是只比较设备数量。
