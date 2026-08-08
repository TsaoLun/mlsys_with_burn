# 逐文件对照矩阵导读

本书改编自 OpenMLSys，但不是逐章翻译：章节被重组，Python 框架实现被
替换为固定快照的 Burn/Rust 证据，原作的硬件数据和外链不作为当前能力。
为了让“哪一段原作变成了哪一节正文、核验到哪一层”始终可查，项目维护
了一份**逐文件对照矩阵**（crosswalk）：

- [在仓库中阅读对照矩阵](https://github.com/TsaoLun/mlsys_with_burn/blob/main/planning/comparison/openmlsys-v1-crosswalk.md)
  （路径为 `planning/comparison/openmlsys-v1-crosswalk.md`）。

矩阵以固定 OpenMLSys v1 revision
`9c289782ccbb165ac8ad7c960ecffc12942a5560` 的中文章节为输入，逐文件
记录映射到本书哪一章哪一节、保留了什么、改写了什么，以及证据状态。
每章末尾的“来源与改编说明”列出本章实际使用的文件清单；对照矩阵则是
全书的总账，二者口径一致。

## 证据标签

正文各章开头和[比较卡](comparison-cards.md)使用同一组标签，说明一个
结论当前有哪种证据支撑：

- `源码核验`：说法直接来自 `pins.toml` 固定 revision 的源码或测试，
  读者可以按给出的路径逐行复核；
- `CPU 可运行验证`：`examples/` 中有默认 CPU 路径即可运行的示例，
  行为由测试断言；
- `协议/成本模型`：框架无关的模型或纯 Rust 协议模拟，用于解释设计，
  不代表任何真实 runtime 的性能或行为；
- `可选平台实验`：需要真实 GPU、NCCL、网络或特定旧 revision 等额外
  环境，本书固定快照未默认验证；
- `未覆盖`：明确不声称的能力边界。

这些标签是本书的阅读指南，不是 Burn 官方能力等级，也不是平台对等
（parity）承诺。

## 对照矩阵的 C/S/R/L/E 字段

矩阵中每个主题记录五类证据：

- **C（Correctness）**：原理、术语和能力边界正确；
- **S（Source）**：固定 OpenMLSys/Burn/CubeCL/CubeK 源码路径可定位；
- **R（Runnable）**：有 CPU 可运行实验，或明确是协议/成本模型；
- **L（Learning）**：前置状态、后续章节和贯穿 workflow 可追踪；
- **E（Engineering）**：导航、来源、许可证和构建可复核。

状态取值为 `verified`（当前快照和命令已核验）、`model`（协议或成本
模型）、`source-only`（源码可定位但当前工作区没有端到端实验）、
`excluded`（明确不进入九章主线，如推荐系统、联邦学习、可解释 AI、
机器人和附录）和 `optional`（需要额外平台环境）。

## 如何使用

1. 读正文时遇到 `源码核验` 的说法，可按章节末节给出的源码入口
   对照固定 revision 阅读；
2. 想确认某个 OpenMLSys 主题在本书中的去向，查对照矩阵的核心路径
   映射；
3. 想判断一个结论能否外推到 GPU、集群或生产部署，先看它带的是
   `CPU 可运行验证` 还是 `协议/成本模型`/`可选平台实验` 标签。
