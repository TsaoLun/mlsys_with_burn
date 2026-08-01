# P1 贯穿实验：数据 → 训练 → ModuleRecord → 推理

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

## 验收协议

实验必须同时检查：

1. train/validation 样本 ID 不重叠，并且合计覆盖 20 个样本；
2. 每个 batch 的 feature shape 是 `[batch, 2]`，target shape 是
   `[batch, 1]`；
3. `items_processed`、batch 数、loss、参数变化和所有浮点输出有限；
4. 训练后 loss 低于固定初始化模型的 loss；
5. record tensor 数、恢复后输出 shape 和最大绝对误差；
6. 将 record 加载到错误 topology 时，固定 Burn 返回
   `RecordError::Validation`，而不是静默截断或“看起来能运行”。

这里的 `final_loss` 是训练完成后、用普通 Device 重新计算的训练集 loss；
validation loader 从一开始就使用 `Device::flex()`，不保留 autodiff tape。
这让“训练时需要梯度”和“评估/推理不需要 tape”成为可观察的边界。

## 示例入口

主程序只输出稳定语义字段，不输出墙钟耗时，因此默认运行可以作为
workspace smoke test：

```rust,ignore
{{#include ../../examples/ch05-ch07-capstone/src/main.rs:capstone_main}}
```

数据、训练、artifact 和恢复逻辑位于同一个库函数：

```rust,ignore
{{#include ../../examples/ch05-ch07-capstone/src/lib.rs:capstone_pipeline}}
```

运行：

```text
cargo test -p ch05-ch07-capstone --locked --offline
cargo run -p ch05-ch07-capstone --locked --offline
```

输出中的 `max_abs_error` 是 CPU 浮点 round-trip 的数值一致性检查，不是
模型精度或服务 latency 结论。`ModuleRecord` 验证的是 Burn 参数 artifact
边界；ONNX 转换、HTTP 服务、Remote、GPU 和多节点部署仍然是第 7 章明确
标出的可选轨道。

## 与 OpenMLSys 的比较

OpenMLSys v1 的 `chapter_programming_interface/ml_workflow.md` 提供完整
机器学习 workflow 的问题框架；模型部署部分进一步讨论转换、artifact 和
inference。这里保留 workflow 的状态转移，但把实现重写为 Rust 的
Dataset/Batcher、所有权、`AutodiffModule::valid` 和 `ModuleRecord`。
因此本实验是“协议和最小实现的可比较证据”，不是对原作 Python 框架或
硬件平台的性能 parity。

## 证据标签

- `CPU 可运行验证`：固定命令运行数据分片、训练、record 和 inference；
- `固定源码核验`：Burn `PartialDataset`、DataLoader、autodiff、SGD 和
  `ModuleRecord`；
- `框架无关模型/协议模拟`：数据契约、错误 topology 和 artifact 验证；
- `需要 CUDA/NCCL/网络/旧 revision 的可选扩展`：GPU、分布式训练、ONNX
  fixture 和服务治理；
- `明确未覆盖`：把二维回归或 CPU elapsed time 外推成生产性能。
