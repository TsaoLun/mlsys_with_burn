# P1 贯穿实验验收记录

## 对照目标

本实验对应 OpenMLSys v1 的：

- `chapter_programming_interface/ml_workflow.md`：输入、输出、状态和错误
  契约的完整 workflow；
- `chapter_data_processing/program_model.md` 与 `data_order.md`：数据集、
  batching、切分和顺序边界；
- `chapter_model_deployment/model_deployment_introduction.md`、
  `model_inference.md`：artifact、加载和 inference 的系统边界。

本书把这些内容重组为 `PartialDataset → MapperDataset → DataLoader/Batcher
→ autodiff training → ModuleRecord → inference` 的单一 CPU-first 路径。
它保留原理和契约，替换框架专用实现，不声称平台或性能 parity。

## 固定协议

- 20 个确定性二维回归样本，按 ID 0–15/16–19 划分 train/validation；
- batch size 4，`num_workers=0`，shuffle seed 41；
- 训练 Device 是 `Device::flex().autodiff()`，validation/inference Device
  是普通 `Device::flex()`；
- 训练模型在第一次 SGD 更新前记录 `initial_loss`，训练完成后记录
  `final_loss`；两者来自同一组初始参数，而不是两个独立初始化的模型；
- 32 个 epoch，`MSE → backward → GradientsParams → SgdConfig::step`；
- `model.valid() → into_record() → into_bytes() → from_bytes() →
  try_load_record() → inference`；
- 错误 topology 加载必须返回 `RecordError::Validation`；
- 主程序只打印样本/批次/loss/参数变化/record tensor/shape/误差等稳定字段。

## 验收字段

| 字段 | 不变量 |
|---|---|
| `train_samples` / `validation_samples` | 16 / 4，合计 20 |
| `train_batches` / `validation_batches` | 4 / 1 |
| loader IDs | train 排序后为 0–15，validation 排序后为 16–19 |
| batch shape | 4 个 train batch 和 1 个 validation batch 均为 input `[4, 2]`、target `[4, 1]` |
| loss/parameter | 有限，最终训练 loss 小于同一初始模型的 loss，参数变化大于 0 |
| record | 2 个参数 tensor，错误 topology 被拒绝 |
| inference | output `[3, 1]`，恢复前后最大绝对误差 `< 1e-6` |

## 证据和边界

- `CPU 可运行验证`：workspace 中的 crate test 和 `cargo run`；
- `源码核验`：主线 Burn 固定 revision 的 Dataset、DataLoader、
  AutodiffModule、optimizer 和 ModuleRecord；
- `协议/成本模型`：workflow 状态、数据契约和错误 topology；
- `可选平台实验`：真实 GPU、NCCL/DDP、ONNX fixture、网络服务；
- `未覆盖`：真实生产数据吞吐、模型质量 benchmark、HTTP/Remote 治理。

这个实验不会修改 `pins.toml`，不使用 `burn-onnx`、Remote、DDP、CUDA、
HTTP 或本地 path dependency。它是跨章学习路径和最小可运行证据，不是对
OpenMLSys 全部部署/训练平台的替代。
