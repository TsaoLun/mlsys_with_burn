# 实验：可复现数据管道

实验位于 `examples/ch05-data-pipeline`，只使用固定 Burn revision 的
Flex CPU 和 `burn::data`。数据集只有 12 个整数样本，不下载外部文件，
因此可以在没有专有驱动的环境中测试 Dataset、shuffle、batching、worker
和进度语义。

## 1. 构造惰性 Dataset

```rust,ignore
{{#include ../../../examples/ch05-data-pipeline/src/lib.rs:dataset}}
```

原始样本的 `id` 和 `value` 都是 `0..12`。`PrepareSample` 把 value 变成
`2 * value + 1`，但 `MapperDataset` 不会在构造时生成一个新的 vector；
读取 item 时才执行 map。

## 2. Batcher 与 Device

```rust,ignore
{{#include ../../../examples/ch05-data-pipeline/src/lib.rs:batcher}}
```

实验 Batcher 保留两个 host vector，并记录收到的 Device。真实模型可以把
同一位置替换为 `Tensor::from_floats`、stack、padding 或标签编码；本实验
不把 Tensor 数值读回混入数据顺序测试。

## 3. 运行一轮

```rust,ignore
{{#include ../../../examples/ch05-data-pipeline/src/lib.rs:pipeline}}
```

运行命令：

```bash
cargo run -p ch05-data-pipeline
cargo test -p ch05-data-pipeline
```

单 worker、batch size 为 3、未 shuffle 时，测试断言：

```text
batch_sizes = [3, 3, 3, 3]
ids         = [0, 1, 2, ..., 11]
values      = [1, 3, 5, ..., 23]
progress    = 12 / 12 items
```

## 4. 固定 seed 与 epoch

两个独立 loader 使用 `seed = 42` 时，第一轮的 ID 排列相同且覆盖
`0..11`。同一个 loader 连续创建两个 iterator 时，RNG 会前进，测试要求
两轮排列不同但仍然覆盖全部样本。

这两个测试分别验证“可复现的起点”和“epoch 间不重复使用同一个排列”。
如果应用需要从某一轮恢复，还需要保存 epoch、seed、调用顺序以及 sampler
状态，而不仅仅保存一个 seed。

## 5. 多 worker 观察

`num_workers = 2` 时，测试不要求 `ids` 展平后仍等于输入序列，而是排序
后检查：

1. 12 个 ID 全部出现；
2. 每个 ID 对应的 map 值正确；
3. batch 大小之和为 12；
4. progress 的 total 和 processed 都是 12。

主程序会打印本次运行的到达顺序。即使某次恰好为升序，也不能把该输出
当成多 worker 保序证据；要得到稳定全局顺序，应使用单 worker 或加入
显式序号/重排层。

## 6. 粗粒度吞吐观察

主程序对 `(batch_size, num_workers)` 的几组配置运行 warm-up 后的 20 个
epoch，并打印墙钟时间和 items/s。这里有意不在测试中断言时间，因为
线程调度、CPU 负载和编译缓存会改变结果。

要做有意义的性能实验，应逐步替换：

1. 内存整数样本为固定大小文件样本；
2. 纯整数 map 为可测量的解码/变换；
3. 一次短测量为多轮 warm-up 与统计分位数；
4. host Batcher 为 Tensor 构造和设备传输；
5. “是否更快”问题为带设备、数据量和同步边界的完整报告。

## 7. 接到第 6–7 章

如果希望继续观察样本如何进入真实训练和 artifact，可运行
[综合实验：数据到推理](../capstone.md)。它使用相同的
`PartialDataset`/`MapperDataset` 思路，但把 host batch 替换为 Tensor，
再连接 autodiff、SGD、`ModuleRecord` 和恢复后 inference。
