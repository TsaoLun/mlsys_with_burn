# 范围、证据与对照

本附录收录版本号、与 OpenMLSys 的文件对照，以及「结论靠什么支撑」的
标签定义。主文九章按系统问题阅读即可。

## 固定版本

正文与示例对齐以下固定版本：

- OpenMLSys：`9c289782ccbb165ac8ad7c960ecffc12942a5560`
- Burn：`976aa9c5ec1d2dd3412710f99759e3c44bdff03d`（版本线 `0.22.0-pre.1`）
- CubeCL：`be278a1e76aed881e2cc6b165414ee6103ca4634`
- CubeK：`f82a6d07ebf35a1d446893b32712458744d80f13`
- burn-onnx：`af2dfb43af43bf363dc2d7d858d933d86e2a65a8`（使用较早的 Burn 提交，只作源码对照，不进入根 workspace 的 Cargo 依赖）

仓库根目录 `pins.toml` 与 `Cargo.lock` 是构建侧真相；本表供阅读对照。

## 结论靠什么支撑（标签定义）

下列标签帮助判断结论靠什么支撑，不是 Burn 官方能力等级：

- `源码核验`：说法直接来自本书固定版本的源码或测试，可按章末路径逐行对照；
- `CPU 可运行验证`：`examples/` 中有默认 CPU 路径即可运行的示例；
- `协议/成本模型`：框架无关的模型或纯 Rust 协议模拟，用于解释设计，不代表任何真实 runtime 的性能或行为；
- `可选平台实验`：需要真实 GPU、NCCL、网络或特定旧版本等额外环境，本书默认示例未覆盖；
- `未覆盖`：明确不声称的能力边界。

## 九章与综合实验范围一览

### 第 1 章 导论

- `CPU 可运行验证`：`ch01-stack-probe` 验证固定 Device/Backend/Tensor 路径；
- `源码核验`：Burn、CubeCL、CubeK 和 burn-onnx 的职责与 revision；
- `协议/成本模型`：workload card、系统分层和成本预算；
- `可选平台实验`：真实 GPU 性能、DDP、ONNX 端到端和集群控制面；
- `未覆盖`：OpenMLSys 的推荐系统、联邦学习、可解释 AI、机器人及附录。

### 第 2 章 编程接口与计算图

- `CPU 可运行验证`：Tensor、Module、autodiff、分支 tape 实验，以及
  无依赖的迷你反向 tape（`ch02-mini-autodiff`）；
- `源码核验`：`Tensor`/`Device`/`Module`、参数状态与一阶 autodiff；
- `协议/成本模型`：workflow 输入/输出/状态/错误契约；
- `可选平台实验`：完整静态图 runtime、device graph capture 与跨设备训练；
- `未覆盖`：把 autodiff tape、Fusion IR 和 device graph capture
  当作同一实现的结论。

### 第 3 章 AI 加速器与编程

- `CPU 可运行验证`：CubeCL CPU Kernel、host reference、tile load 模型
  和纯 Rust 分块 GEMM 语义验证（`ch03-gemm-ladder` 默认路径）；
- `源码核验`：CubeCL/CubeK 的拓扑、buffer、算子与 backend 入口；
- `协议/成本模型`：GEMM、算术强度和 Roofline 方向；
- `可选平台实验`：`--features wgpu`（scale Kernel 与 GEMM 阶梯实测，
  见 `docs/OPTIONAL_PROFILES.md`）、真实 GPU GEMM 跨设备比较、
  autotune 性能和厂商设备比较；
- `未覆盖`：用 CPU 正确性结果替代 GPU 带宽、吞吐或 launch 结论。

### 第 4 章 AI 编译器与运行时系统

- `CPU 可运行验证`：FusionInspector 的计划结构、数值等价和同步边界，
  以及无依赖的迷你 Pass 流水线（`ch04-mini-pass-pipeline`，含浮点
  非法变换反例）；
- `源码核验`：OperationIr、Fusion stream、CubeCL Scope、编译和
  HandleContainer 的生命周期入口；
- `协议/成本模型`：Pass、lowering、cache 和 launch/read 因果链；
- `可选平台实验`：真实 kernel launch、硬件 graph capture 和设备性能；
- `未覆盖`：将 Fusion block 数、cache hit、kernel launch count 和
  wall-clock time 当成同一个指标。

### 第 5 章 数据处理系统

- `CPU 可运行验证`：Dataset、Mapper、Batcher、DataLoader、固定 seed
  和多 worker 数据守恒；
- `源码核验`：内存 Dataset、SQLite、采样、worker 和 Device 边界；
- `协议/成本模型`：文件索引、背压、retry、epoch commit 和
  reorder buffer；
- `可选平台实验`：真实存储吞吐、pinned memory、跨节点 sampler 和
  设备数据通道；
- `未覆盖`：把数据守恒或一次 CPU 测量描述成全局保序/真实吞吐。

### 第 6 章 训练系统

- `CPU 可运行验证`：forward/backward、SGD、loss 下降、参数变化和
  checkpoint 基础状态；
- `源码核验`：`TrainStep`、`Learner`、optimizer、`MultiDevice`、
  `DistributedContext` 与 collective 入口；
- `协议/成本模型`：AllReduce、parameter-server、pipeline bubble
  和 checkpoint version；
- `可选平台实验`：Flex CPU 之外的 DDP、跨节点网络和真实通信性能；
- `未覆盖`：把单机训练 loop 当作分布式训练、集群容错或 NCCL 证明。

### 第 7 章 模型服务

- `CPU 可运行验证`：当前 workspace 的 `ModuleRecord`/Burnpack 参数往返保存与恢复、
  恢复后的 inference，以及纯 Rust 的 PTQ 校准与 int8 GEMM 误差实验
  （`ch07-ptq-calibration`）；
- `源码核验`：burn-onnx 的 graph/codegen/load strategy、Remote、
  WASM/no_std 和当前 workspace 的 artifact 入口；
- `协议/成本模型`：manifest、checksum、版本、rollback、batch/queue、
  安全威胁模型，以及连续批处理与 KV 预算的虚拟时间队列模型
  （`ch07-serving-queue-sim`）；
- `可选平台实验`：真实 ONNX fixture、服务治理、浏览器/Remote 部署和
  设备性能；
- `未覆盖`：burn-onnx 旧 revision 与当前 workspace Burn 的端到端混用。

### 第 8 章 强化学习系统

- `CPU 可运行验证`：Environment、Policy 组合、done/truncated、replay
  shape 和表格 TD update；
- `源码核验`：`burn-rl` 的 Environment/Policy/TransitionBuffer 与
  `burn-train` 的 rollout/evaluation/checkpoint 边界；
- `协议/成本模型`：policy freshness、behavior/target metadata、
  双智能体 action/reward vector 和 credit assignment；
- `可选平台实验`：真实 simulator、神经网络 DQN、Actor–Learner 和
  MARL 集群；
- `未覆盖`：把抽象组合 API 描述成完整 DQN/PPO/SAC/MARL runtime。

### 第 9 章 大规模 GPU 集群管理

- `CPU 可运行验证`：队列、gang admission、拓扑放置、通信成本、故障
  retry、checkpoint replay 和资源归还；
- `源码核验`：Burn/CubeCL 的设备、stream、memory、collective 和
  training data-plane 入口；
- `协议/成本模型`：控制面、故障域、队列公平、链路热点和
  machine-readable trace；
- `可选平台实验`：真实 GPU 集群、NCCL/RDMA、网络拥塞、多租户 runtime
  和弹性 membership；
- `未覆盖`：把模拟器虚拟时间、放置结果或通信 penalty 当作 GPU
  benchmark。

### 综合实验：数据到推理

- `CPU 可运行验证`：固定命令运行数据分片、训练、record 和 inference；
- `源码核验`：Burn `PartialDataset`、DataLoader、autodiff、SGD 和
  `ModuleRecord`；
- `协议/成本模型`：数据契约、错误 topology 和 artifact 验证；
- `可选平台实验`：GPU、分布式训练、ONNX fixture 和服务治理；
- `未覆盖`：把二维回归或 CPU elapsed time 外推成生产性能。

### 算子解剖：tanh 的完整一生

- `CPU 可运行验证`：`ch02-ch04-op-anatomy` 的前向/反向/组合数值断言
  （API→dispatch→Flex 与 autodiff 反向规则）；
- `源码核验`：tanh 在 API、契约、dispatch、autodiff、Flex、CubeCL、
  Fusion、IR、backend-tests 各层的固定源码位置；
- `可选平台实验`：CubeCL/Fusion 层的实际执行（第 3 章 wgpu 路径与
  第 4 章 FusionInspector 可部分观察）；
- `未覆盖`：修改上游源码的演练；GPU 层默认不运行。

## 第 1 章与 OpenMLSys 导论的对照维度

这一节把第 1 章的系统地图和 OpenMLSys v1 的导论文件逐项对齐。对照的
对象是本书固定版本，而不是某个会变化的在线页面：

- 原作的应用、设计目标、架构和生态材料保留为框架无关的系统问题；
- 本书把实现路径重写为 `Tensor → autodiff → IR/Fusion → Kernel → Runtime`；
- 第 5–7 章继续把数据、训练、artifact 和推理连接成一个可运行 workflow；
- 第 9 章补充控制面，但不把 Burn 的训练数据面说成集群 scheduler。

表中五个维度（C/S/R/L/E）的定义见下文
「对照矩阵的 C/S/R/L/E 字段」：

| 维度 | 第 1 章可核验内容 | 不能从第 1 章推出的结论 |
|---|---|---|
| C（正确性） | workload card、系统分层和 Burn/CubeCL/CubeK 职责 | 所有后端能力相同 |
| S（源码） | 章末源码入口和 OpenMLSys 逐文件对照矩阵 | 最新版本 API |
| R（可运行性） | CPU `ch01-stack-probe` 的 Device/Backend/Tensor 路径 | GPU 性能或网络吞吐 |
| L（学习路径） | 由应用负载连接第 2–9 章 | 本书覆盖 OpenMLSys 全部专题 |
| E（工程复核） | 章节导航、来源文件和许可证入口 | 上游项目官方背书 |

OpenMLSys 的推荐系统、联邦学习、可解释 AI、机器人和机器学习附录不在
本书首版九章正文范围。它们是可追踪的范围差异，不应被“九章”这一数字误读
为对原作全部内容的能力对等性（parity）声明。

## 主题比较卡

下面把 OpenMLSys v1 的核心系统主题改写成本书定义的主题比较卡。每张卡都回答
五个问题：原作讨论什么、当前本书用什么模型、固定源码在哪里、读者能
运行什么观察、哪些能力和硬件条件不能直接比较。

卡片中的证据标签沿用上文「结论靠什么支撑」的五个定义，是本书的证据
层级，不是平台能力对等（parity）承诺。
这些卡片是横向摘要，不替代逐文件对照矩阵（入口见
下文「逐文件对照矩阵」）；第 1–2 章的
接口、计算图和编程模型对照仍以对应章节和逐文件对照矩阵为准。下列卡片聚焦第
3–9 章中最容易把“概念、源码入口、协议模型”误读成“完整运行时”的主题。

### 第 3 章：GEMM 与加速器

- **原作问题**：加速器架构、线程/存储层次、GEMM 优化阶梯和设备性能。
- **OpenMLSys 文件**：`chapter_accelerator/accelerator_architecture.md`、
  `accelerator_programming.md`、`accelerator_practise.md`。
- **本书模型**：固定 shape、dtype、backend、warm-up、同步点、重复次数和
  host reference 的测量协议；算术强度只用来解释复用方向。
- **固定入口**：`cubecl/crates/cubecl-core/src/`、
  `cubecl/crates/cubecl-runtime/src/`、`cubek/crates/cubek-matmul/src/`。
- **可运行观察**：`ch03-cubecl-kernel` 验证 CPU Kernel 正确性（有图形
  驱动的环境可用 `--features wgpu` 对照同一 Kernel 的 WGPU 运行结果）；
  `ch03-tile-loads` 验证 tile load/intensity 计数。
- **不可直接比较**：CPU correctness 不能替代 GPU shared memory、带宽、
  autotune 或厂商 GEMM benchmark。标签为 `源码核验 + CPU 可运行验证 +
  协议/成本模型`；真实 GPU 是 `可选平台实验`。

### 第 4 章：IR、Fusion、cache 与 launch

- **原作问题**：前端 IR、Pass、kernel selection、编译、内存和运行时调度。
- **OpenMLSys 文件**：`chapter_frontend_and_ir/intermediate_representation.md`、
  `common_frontend_optimization_pass.md`、`chapter_backend_and_runtime/`
  下的 optimizer/compiler/runtime 文件。
- **本书模型**：`OperationIr → Fusion block → CubeCL Scope/
  KernelDefinition → compiler/JIT/cache → launch/read/sync` 的对象级追踪。
- **固定入口**：`burn/crates/burn-ir/src/`、
  `burn/crates/burn-fusion/src/`、`burn/crates/burn-cubecl-fusion/src/`、
  `cubecl/crates/cubecl-runtime/src/`。
- **可运行观察**：`ch04-fusion-inspector` 重复相同
  `add → mul → exp`，比较计划结构和数值；`BURN_FUSION_LOG=full` 是
  可选日志观察。
- **不可直接比较**：Fusion block 数、cache hit、kernel launch count 和
  wall-clock time 不是同一指标；上游 API 不提供稳定私有 cache key。
  标签为 `源码核验 + CPU 可运行验证`，硬件 launch 是 `可选平台实验`。

### 第 5 章：数据处理

- **原作问题**：数据读取、顺序、shuffle、并行 worker、预取、背压和吞吐。
- **OpenMLSys 文件**：`chapter_data_processing/requirements.md`、
  `program_model.md`、`data_order.md`、`performance.md`。
- **本书模型**：用 `F/P/G` 表示 fetch/produce/consume，增加 deterministic
  shard/offset/decode、queue capacity、retry、epoch commit 和 reorder
  invariants。
- **固定入口**：`burn/crates/burn-dataset/src/` 和
  `burn/crates/burn-core/src/data/dataloader/`。
- **可运行观察**：`ch05-data-pipeline` 的 Dataset/Mapper/Batcher/DataLoader
  测试验证数据守恒、分片和背压协议；综合实验把 Tensor batch 交给
  第 6 章训练。
- **不可直接比较**：内存样本和虚拟 queue 不代表磁盘、网络、pinned
  memory 或全局保序吞吐。标签为 `源码核验 + CPU 可运行验证 +
  协议/成本模型`。

### 第 6 章：分布式训练

- **原作问题**：数据/模型/流水线并行、collective、parameter server、
  stale gradient、quorum 和 checkpoint 一致性。
- **OpenMLSys 文件**：`chapter_distributed_training/methods.md`、
  `collective.md`、`parameter_servers.md`、`cluster.md`。
- **本书模型**：加权 AllReduce、版本化 stale gradient、quorum、1F1B
  bubble 和单调 checkpoint commit 的纯 Rust 协议模型测试。
- **固定入口**：`burn/crates/burn-train/src/`、
  `burn/crates/burn-communication/src/` 和
  `burn/crates/burn-core/src/tensor/distributed.rs`。
- **可运行观察**：`ch06-training-loop` 的纯 Rust 协议 helper 测试
  weighted average、staleness、quorum、pipeline slots 和 checkpoint
  version；另由 CPU autodiff loop 验证单设备训练。
- **不可直接比较**：协议结果不等于 DDP/NCCL/跨节点性能或故障恢复；
  Flex CPU collective 仍是 `未覆盖`，真实通信为 `可选平台实验`。

### 第 7 章：模型部署

- **原作问题**：转换、压缩、artifact、推理 runtime、安全、batching 和
  rollback。
- **OpenMLSys 文件**：`chapter_model_deployment/model_converter_and_optimizer.md`、
  `model_compression.md`、`model_inference.md`、`model_security.md`。
- **本书模型**：manifest 的 version/payload length/checksum、回滚条件和
  同 shape 动态 batching 契约（contract）；当前 workspace 的 artifact 与 burn-onnx revision
  分开。
- **固定入口**：`burn/crates/burn-core/src/store/`、
  `burn/crates/burn-core/src/module/base.rs` 和
  `burn-onnx/crates/burn-import/src/`。
- **可运行观察**：`ch07-record-roundtrip` 验证 Burnpack 参数往返保存与恢复；
  同 crate 的纯 Rust contract harness 用教学用的非密码学 checksum
  验证 payload length、版本、rollback 和 dynamic batch。
- **不可直接比较**：manifest/checksum 不是完整供应链安全；旧 revision
  的 ONNX fixture、HTTP、Remote、WASM 和 GPU service 是可选/未覆盖。

### 第 8 章：强化学习

- **原作问题**：MDP、trajectory、replay、on/off-policy、多环境、MARL 和
  Actor–Learner freshness。
- **OpenMLSys 文件**：`chapter_reinforcement_learning/rl_introduction.md`、
  `single_node_rl.md`、`marl.md`、`marl_sys.md`。
- **本书模型**：确定性 `MockPolicy`、behavior/target version lag、
  joint action/reward vector 和已验证的 done/truncated/TD 语义。
- **固定入口**：`burn/crates/burn-rl/src/`、
  `burn/crates/burn-train/src/learner/`。
- **可运行观察**：`ch08-rl-rollout` 的 CPU 环境/replay/TD 测试和协议模型测试；
  不依赖 gym 或外部 simulator。
- **不可直接比较**：mock policy 不是 DQN/PPO/SAC；joint vector 不是
  MARL credit assignment runtime；完整 Actor–Learner 是可选/未覆盖。

### 第 9 章：GPU 集群与控制面

- **原作问题**：cluster topology、队列、gang scheduling、通信、故障、
  checkpoint 和 observability。
- **OpenMLSys 文件**：`chapter_distributed_training/overview.md`、
  `cluster.md`、`collective.md`、`parameter_servers.md`。
- **本书模型**：GPU/node/rack 拓扑、FIFO/topology-aware placement、
  `alpha + beta * bytes`、failure domain、queue wait、retry 和 versioned
  machine-readable trace。
- **固定入口**：Burn 的设备/通信数据面入口
  `burn/crates/burn-communication/src/`、`burn/crates/burn-train/src/`；
  控制面模型位于 `ch09-cluster-simulator`。
- **可运行观察**：CPU 离散事件模拟器验证 gang admission、资源归还、
  拓扑成本、checkpoint replay、队列指标和 `TRACE_SCHEMA_VERSION`。
- **不可直接比较**：虚拟时间、cross-rack penalty 和 trace 不代表 GPU、
  NCCL、RDMA、网络拥塞、多租户 runtime 或弹性 membership benchmark。

### 与综合实验的关系

综合实验把第 5–7 章串成一条 CPU 可运行路径：
`Dataset → autodiff → ModuleRecord → inference`；它回答“状态怎样跨过
数据、训练和 artifact 边界”。本页的比较卡回答“同一主题在 OpenMLSys、
固定 Burn 源码、CPU 实验和协议模型之间分别有哪种证据”。前者是纵向学习
路径，后者是横向审计摘要，二者不能相互替代。

## 逐文件对照矩阵

本书改编自 OpenMLSys，但不是逐章翻译：章节被重组，Python 框架实现被
替换为本书固定版本的 Burn/Rust 证据，原作的硬件数据和外链不作为当前能力。
为了让“哪一段原作变成了哪一节正文、核验到哪一层”始终可查，项目维护
了一份**逐文件对照矩阵**（crosswalk）：

- [在仓库中阅读对照矩阵](https://github.com/TsaoLun/mlsys_with_burn/blob/main/planning/comparison/openmlsys-v1-crosswalk.md)
  （路径为 `planning/comparison/openmlsys-v1-crosswalk.md`）。

矩阵以固定 OpenMLSys v1 revision
`9c289782ccbb165ac8ad7c960ecffc12942a5560` 的中文章节为输入，逐文件
记录映射到本书哪一章哪一节、保留了什么、改写了什么，以及结论靠什么支撑。
各章文件级改编说明见[来源与改编总录](appendix-sources.md)；对照矩阵是全书总账，二者口径一致。

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

## 如何使用本附录

1. 读正文时遇到 `源码核验` 的说法，可按章节末节给出的源码入口
   对照固定 revision 阅读；
2. 想确认某个 OpenMLSys 主题在本书中的去向，先读比较卡（先看对应
   章节的框架无关模型，再运行卡片列出的 CPU 示例），再查对照矩阵的
   核心路径映射；
3. 想判断一个结论能否外推到 GPU、集群或生产部署，先看它带的是
   `CPU 可运行验证` 还是 `协议/成本模型`/`可选平台实验` 标签——
   没有同时标出证据标签、硬件前提和未覆盖边界的结论，本书不把它
   当作可对外比较的结论。
