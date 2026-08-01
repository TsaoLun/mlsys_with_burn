# 前向、反向与自定义训练循环

## 一次 step 的边界

一个最小的监督学习 step 通常有下面的顺序：

```text
取 batch
  → model.forward
  → loss
  → loss.backward
  → GradientsParams::from_grads
  → optimizer.step
  → 记录 loss / metric
```

`backward` 只产生这一次反向传播的梯度容器；它不会自行修改
`Module` 参数。参数更新是后续 optimizer 的职责。把这两个边界混在一起
会让“loss 下降”无法判断究竟是梯度、学习率、参数映射还是数据顺序出了
问题。

## `TrainStep` 表达应用模型

固定 Burn 的 `burn-train/src/learner/train_val.rs` 没有假设所有模型都使用
同一个输入或输出。`TrainStep` 由应用为自己的模型定义：

- `Input: Send + 'static`：可以由 DataLoader 跨线程发送；
- `Output: ItemLazy + 'static`：训练指标可以延迟提取；
- `step(&self, item)`：应用实现 forward、loss 和 backward；
- 默认 `optimize`：把 `GradientsParams` 交给 `ModuleOptimizer::step`；
- 默认 `optimize_multi`：把多个设备的梯度交给 `step_multi`。

模型还必须实现 `InferenceStep`、`AutodiffModule` 和 `Display`，才满足
`LearnerModel`。这是一种 Rust trait 边界：训练框架只依赖它需要的能力，
不会要求每个模型都采用固定的 batch struct。

## 从 autodiff 到 optimizer

第 2 章已经区分了 autodiff tape 和其他 IR。训练 step 中的具体关系是：

```text
Tensor operations
      │ register
      ▼
autodiff tape ── loss.backward() ──► Gradients
                                      │
                    from_grads + Module ParamId
                                      ▼
                              GradientsParams
                                      │
                          ModuleOptimizer::step
                                      ▼
                              new Module
```

`GradientsParams` 把梯度和 module 的 `ParamId` 对齐。它不是“按字段名字
猜参数”，也不是直接把一组 Tensor 写回任意内存。优化器通过 module mapper
访问参数，应用学习率、weight decay、momentum 或其他策略，再返回更新后的
module。

这也是 Rust 示例中常见的赋值形式：

```text
model = optimizer.step(lr, model, gradients)
```

它不是说每个 Tensor 都被复制到主机；它表达的是 module 所有权被消费并
产生一个新的 module 值。参数底层是否复用 buffer、何时异步执行，仍由
backend 决定；只有 `read` 或 `Device::sync` 等明确边界才能谈完成。

## 本章实验中的真实代码

实验把最小 step 保持在一个可读函数附近，下面的 include 是
`examples/ch06-training-loop/src/lib.rs` 的唯一代码来源：

```rust,ignore
{{#include ../../../examples/ch06-training-loop/src/lib.rs:train_step}}
```

其中 `loss.clone()` 用于先读出一个标量观测值，再把同一个 loss 交给
`backward`。这不是为了复制训练图，而是为了让日志读取和反向消费各自
拥有一个 Rust handle。读者可以把 MSE 换成分类 loss，同时保留同样的
梯度边界。

## 验证模式与训练模式

验证不需要保留训练用的 autodiff tape。Burn 的训练策略在 validation
阶段调用 `model.valid()`，再执行 `InferenceStep`。这至少表达了两个
系统意图：

1. 验证不应该把上一轮训练的梯度继续累积；
2. 训练与验证的 metric 输入可以不同，但必须有清楚的事件归属。

`valid()` 不是通用的“把所有状态重置”按钮。BatchNorm、Dropout 或自定义
模块是否有额外的训练/推理语义，要由模块实现和调用者验证。

## 常见错误

- 只调用 `backward`，忘记把梯度交给 optimizer，loss 自然不会因参数更新
  而改变；
- 在更新后继续把旧 module 当作当前参数使用；
- 把输入 batch 的 host device 和 autodiff device 混为一个类型；
- 把一次 loss 读回的时间当成整个训练 step 的设备完成时间；
- 先用多设备/异步执行掩盖单设备梯度错误；
- 只打印 loss，不保存 step、epoch、学习率和数据范围，导致曲线无法复现。

最小训练循环的价值正是把这些边界暴露出来。`burn-train` 可以在其上
增加事件、指标和检查点，但不会改变 forward/backward 的数学含义。
