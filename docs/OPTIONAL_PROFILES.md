# 可选运行 Profile

本书主线在正文同步介绍 GPU / 多 Runtime / 集合通信等概念，并用固定版本
源码对照；**默认可运行路径仍是 CPU**。本文件记录有额外环境时可选用的
跑通轨。无环境者跳过本文件不影响九章主线。

约束见 D022：下列 profile **不得**进入默认 `make check`。

## 有环境 / 无环境怎么读

| 读者情况 | 建议 |
|---|---|
| 仅 CPU | 按章做默认示例；GPU/Runtime/collective 当正文+源码导读 |
| 有图形驱动（Metal/Vulkan/DX12 等） | 加跑 `wgpu` profile，核对数值为止 |
| 有 CUDA/NCCL 或多机 | 先读第 3/6/9 章源码入口，再自建实验；本书不提供默认 CI 命令 |
| 要对照 ONNX | 走 `onnx-fixture` 说明，遵守 D010，不混进根 workspace |

读者入口：[如何运行本书示例](../book/src/running-examples.md)。

## Profile：`wgpu`（第 3 章）

- **目的**：同一 CubeCL Kernel 在 `WgpuRuntime` 上与 host reference 对照。
- **命令**：

```bash
cargo run  -p ch03-cubecl-kernel --features wgpu --locked
cargo test -p ch03-cubecl-kernel --features wgpu --locked
```

- **前提**：CubeCL/WGPU 可用 adapter；首次编译可能较慢。
- **测量协议（若自行计时）**：
  1. 正确性与计时分离；先断言与 host reference 一致；
  2. 区分首次 JIT/cache 与稳态；
  3. 在读回/同步之后再停表；
  4. 记录 OS、adapter、CubeCL/Burn revision（`pins.toml`）；
  5. **不要**把本次结果写成全书 GPU GEMM 或占用率结论。
- **正文位置**：第 3 章实验小节；多 Runtime 表见 `03-cubecl-programming.md`。

## Profile：`onnx-fixture`（第 7 章对照）

- **目的**：阅读 `burn-onnx` 固定 revision 的 ONNX→BurnGraph→codegen
  边界；不是根 workspace 默认依赖。
- **约束**：D010——旧 Burn revision，禁止 `path`/`[patch]` 拉进根
  workspace；不要与 `ch07-record-roundtrip` 混成一条 `cargo test`。
- **建议做法**：独立 checkout / 独立 manifest（若维护者提供）或只读源码；
  任何端到端跑通结果单独记录环境与 revision。
- **正文位置**：第 7 章 ONNX 节与部署闭环图。

## Profile：`cuda` / collective（未来追加）

仅当 `pins.toml` 对应源码与本机驱动同时允许时，才追加可复制命令。当前
口径：

1. 先按第 3 章读 `CudaRuntime` / 第 6 章读 `DistributedOps` 源码；
2. 再决定是否自建多设备实验；
3. 报告必须分开：语义正确性 vs 墙钟；单机 vs 跨节点；Flex 无 collective
   的边界不得被“能编译”掩盖。

NCCL 真机跑通**不是**主线第一次介绍集合通信的方式——主线已在第 6/9 章
用模型与源码讲过。

## 与默认门禁的关系

| 门禁 | 包含可选 profile？ |
|---|---|
| `make check` | 否 |
| `make check-local-sources` | 否（只多读本地镜像） |
| 本章默认 `cargo test -p chNN-…` | 否 |
| 本文件中的显式命令 | 是（读者自愿） |
