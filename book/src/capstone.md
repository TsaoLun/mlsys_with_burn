# 综合实验：数据 → 训练 → ModuleRecord → 推理

第 5–7 章分别讨论数据处理、训练系统和模型 artifact。如果每章只运行
自己的小例子，读者仍可能看不见系统状态如何穿过边界。本实验用一个
确定性的二维回归任务把三章串起来：

```text
20 个样本
  ├─ PartialDataset + MapperDataset
  ├─ train 16 / validation 4
  ├─ DataLoader(num_workers=0, fixed shuffle seed)
  ├─ autodiff forward → MSE → backward → SGD
  ├─ model.valid() → ModuleRecord → bytes → try_load_record
  └─ 恢复后的 CPU inference
```

## 状态穿过哪些边界

跑通后请同时看这些字段——它们对应数据、训练和产物三条边界：

1. train/validation 的排序后样本 ID 分别严格等于 `0..15` 和 `16..19`，
   因而没有重复、重叠或遗漏；
2. train 有 4 个 batch、validation 有 1 个 batch；每个 batch 的 feature
   shape 是 `[4, 2]`，target shape 是 `[4, 1]`；
3. `items_processed`、batch 数、loss、参数变化和所有浮点输出有限；
4. 训练后 loss 低于同一初始模型在第一次更新前的 loss；
5. record tensor 数、恢复后输出 shape 和最大绝对误差；
6. 将 record 加载到错误 topology 时，Burn 返回
   `RecordError::Validation`，而不是静默截断或“看起来能运行”。

这里的 `initial_loss` 和 `final_loss` 都是同一组训练参数在训练集上的
loss：前者在第一次 SGD 更新前计算，后者在训练完成后、用普通 Device
重新计算。validation loader 从一开始就使用 `Device::flex()`，不保留
autodiff tape。这让“训练时需要梯度”和“评估/推理不需要 tape”成为可观察
的边界，而不是拿两个不同随机初始化的模型比较 loss。

## 示例入口

主程序只输出稳定语义字段，不输出墙钟耗时，便于你快速对照结果：

```rust,ignore
{{#include ../../examples/ch05-ch07-capstone/src/main.rs:capstone_main}}
```

数据、训练、artifact 和恢复逻辑位于同一个库函数。阅读时建议按函数内
注释分成四段对照：`loader audit → 训练循环 → Record 往返 → 推理验证`，
而不是把它当成一个必须逐行记住的长函数。

```rust,ignore
{{#include ../../examples/ch05-ch07-capstone/src/lib.rs:capstone_pipeline}}
```

运行：

```text
cargo test -p ch05-ch07-capstone --locked --offline
cargo run -p ch05-ch07-capstone --locked --offline
```

输出中的 `max_abs_error` 是 CPU 浮点往返保存/恢复（round-trip）的数值一致性检查，不是
模型精度或服务 latency 结论。`ModuleRecord` 验证的是 Burn 参数 artifact
边界；ONNX 转换、HTTP 服务、Remote、GPU 和多节点部署仍然是第 7 章明确
标出的可选路径。

## 把它变成你自己的实验

运行通过只说明你能复现本书给出的参考结果。把它变成学习成果，请再做一
件小改动并记录结果：

1. 任选一个变量：`BATCH_SIZE`、样本数、训练 epoch、学习率或模型输入维数；
2. 先预测哪些不变量仍应成立（样本守恒、shape、拓扑错误、往返误差）；
3. 修改源码并运行测试；
4. 用一段实验笔记记录：配置、预测、实际输出、失败原因或解释。

例如把 batch size 从 4 改为 5 时，样本数仍应是 16/4，但 batch 数和每个
batch 的 shape 都会变化；如果你只改断言而不解释为什么会变，这条实验还
没有形成系统结论。你不需要提交代码，只需要能用输出证明你理解了每个边界。

## 与 OpenMLSys 的比较

OpenMLSys v1 的 `chapter_programming_interface/ml_workflow.md` 提供完整
机器学习 workflow 的问题框架；模型部署部分进一步讨论转换、artifact 和
inference。这里保留 workflow 的状态转移，但把实现重写为 Rust 的
Dataset/Batcher、所有权、`AutodiffModule::valid` 和 `ModuleRecord`。
因此本实验是可运行的最小 workflow，不是对原作 Python 框架或硬件平台的
性能对等。若需对照 OpenMLSys 原作映射，见[范围、证据与对照附录](appendix-scope-and-evidence.md)。
