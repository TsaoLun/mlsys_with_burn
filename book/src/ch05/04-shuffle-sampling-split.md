# Shuffle、采样与数据划分

## Shuffle 是索引重排

`ShuffledDataset` 并不复制或改写底层 item。它先建立
`0..dataset.len()` 的索引，再用 `StdRng` 打乱索引；`get(i)` 通过索引
表访问底层 Dataset。这样 shuffle 的成本主要是索引 vector 和随机排列，
而不是样本本身的复制。对 100 万样本，索引表是
$10^6 \times 8\ \text{B} = 8\ \text{MiB}$ 的 `usize` vector 加一次
$O(N)$ 排列；若改为复制样本本身（比如每条 1 KiB 的文本），则要移动
约 $1\ \text{GiB}$。索引重排把 shuffle 的内存开销与样本大小解耦，
这也是它能作用于任意存储后端（包括 SQLite）的原因。

`RngSource` 支持三种来源：

- 默认系统随机源；
- 固定 `u64` seed；
- 已有的 `StdRng`，也可以从父 RNG fork 出独立流。

固定 seed 只在相同快照、相同数据集长度、相同调用顺序和相同随机算法下
提供可复现的排列。它不是跨版本、跨实现或跨多线程调度的普遍复现承诺。

## 一个 loader 的 epoch 行为

`DataLoaderBuilder::shuffle(seed)` 把 RNG 保存到 loader。每次调用
`iter()` 时，固定版本会从 RNG 得到下一次排列。因此：

```text
独立 loader + 相同 seed → 第一轮排列相同
同一个 loader 的连续两轮 → RNG 前进，排列通常不同
```

这和每轮重新用同一个 seed 构造 loader 不同。实验分别检查这两个情况，
避免把“固定 seed”误写成“所有 epoch 永远相同”。

## `SamplerDataset` 不是 shuffle

`SamplerDataset` 用一个随机分布把源数据集包装成指定大小：

- with replacement 可以重复抽到 item；
- without replacement 在一个周期内不重复，周期结束后重新填充和打乱；
- `SizeConfig` 可以是源大小、固定大小或比例。

训练 epoch 想要一次覆盖每个样本时，通常先考虑不放回的全量 shuffle；
需要过采样、欠采样或固定抽样预算时，才应明确使用 Sampler。二者的
“随机”语义和 `len()` 都不同。

## 多设备和 worker 的划分

固定 Burn 提供两个相关但不同的划分点：

1. `split_dataloader(dataloader, devices)` 按连续范围切分 loader，最后
   一个分片接收余数，再调用 `to_device`；
2. `MultiThreadDataLoader` 初始化时按 worker 数切分 Dataset；当 batch
   size 已知时使用 `PartialDataset::split_chunks`，尽量按完整 batch
   分配 chunk。

这些操作只描述本地 Dataset 的索引范围和 batch 目标设备。它们不是
分布式通信协议，也没有在这里提供跨节点 sampler、全局 epoch barrier
或 AllReduce。第 6 章再讨论训练系统如何把数据划分和多设备更新联系起来。

## 顺序与覆盖率的测试策略

单 worker 可以直接断言完整 ID 序列；多 worker 应至少断言：

- 所有期望 ID 都出现；
- 没有重复或遗漏；
- 每个 ID 的变换值仍正确；
- 进度的 `items_processed == items_total`。

若应用还要求全局顺序，就必须把这个要求写进额外协议，而不能只凭
`shuffle(seed)` 或 `num_workers` 参数推断。
