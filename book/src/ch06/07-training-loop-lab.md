# 实验：CPU 线性回归训练循环

## 实验目标与边界

实验位于 `examples/ch06-training-loop`，使用固定 Burn revision 的
`Device::flex().autodiff()` 和一个 `1 → 1` 的 `Linear` module。数据满足
$y=2x+1$，训练目标是观察 MSE loss 随 SGD 更新下降。

本实验验证的是训练循环的正确性：

```text
forward → loss → backward → GradientsParams → SGD → loss
```

它不验证 DDP、AllReduce、网络带宽、模型并行或 GPU 性能。固定快照的 Flex
后端没有 collective 实现，这个限制是实验设计的一部分，而不是遗漏。

## 1. 构造 CPU autodiff model

```rust,ignore
{{#include ../../../examples/ch06-training-loop/src/lib.rs:setup}}
```

`Device::flex().autodiff()` 给 model、输入和 target 建立同一 autodiff
设备。固定 seed 只让初始化可复现；它不是训练状态 checkpoint，也不替代
保存 epoch、scheduler 和 sampler。

输入使用五个小样本，故意不引入 DataLoader 线程和外部文件。第 5 章已经
验证了 batch 生产边界，本实验把关注点收窄到 batch 进入模型之后发生的
状态转移。

## 2. 执行训练 step

```rust,ignore
{{#include ../../../examples/ch06-training-loop/src/lib.rs:train_step}}
```

每次循环先保留一个 loss 标量用于观察，再调用 `backward`。随后
`GradientsParams::from_grads` 根据 module 参数建立优化器需要的映射，
`SgdConfig::new().init()` 返回的 optimizer 消费 model 并返回更新后的 model。

这段代码没有显式调用“清空梯度”。Burn 的这种手写路径中，每次
`loss.backward()` 返回一次新的 `Gradients`；若要实现梯度累积，应该显式
用 accumulator 或 `burn-train` 的 `grads_accumulation`，不能凭其他框架的
惯例猜测。

## 3. 运行与测试

```bash
cargo test -p ch06-training-loop
cargo run -p ch06-training-loop
```

测试断言：

1. `losses` 的长度等于 step 数；
2. 最终 loss 小于初始 loss；
3. weight 参数的绝对变化量大于零；
4. 零 step 和非正/非有限 learning rate 返回描述性错误。

主程序打印每一步的 loss 以及最终参数变化：

```text
step=1 loss=...
step=2 loss=...
...
initial_loss=... final_loss=... parameter_delta=...
```

具体浮点数可能因 backend、编译选项或上游实现变化而不同；测试使用趋势
和非零更新作为不变量，没有把某一串小数写成跨平台协议。

## 4. 可以怎样扩展

按下面顺序扩展，能保持每一步的因果边界：

1. 把五个 Tensor 换成第 5 章的 Dataset/Batcher；
2. 为 `Linear` 加验证 loader，并使用 `model.valid()`；
3. 记录 iteration、learning rate 和 batch 进度；
4. 把 `SgdConfig` 换成 `AdamConfig`，观察 optimizer state；
5. 将 model、optimizer 和 scheduler 保存为各自的 record，再测试恢复；
6. 实现一个 `TrainStep`，交给 `burn-train::Learner`；
7. 只有在有匹配后端和设备时，才尝试 `ExecutionStrategy::MultiDevice` 或
   DDP，并先验证局部梯度与单设备 reference。

最后一步尤其重要：不能因为本章已经能在 CPU 上训练一个线性模型，就声称
跨节点 DDP 已经工作。

## 5. 接到第 5–7 章

完整的 Dataset → autodiff → ModuleRecord → inference 路径见
[综合实验：数据到推理](../capstone-p1.md)。它把本实验的
`forward → loss → backward → SGD` 放入固定 train/validation split，并
增加 loader 守恒、record topology 错误和恢复后输出误差检查。
