# OpenMLSys 核心主题比较卡

本页把 OpenMLSys v1 的核心系统主题改写成可审计的比较卡。每张卡都回答
五个问题：原作讨论什么、当前本书用什么模型、固定源码在哪里、读者能
运行什么观察、哪些能力和硬件条件不能直接比较。

证据标签统一为：`源码核验`、`CPU 可运行验证`、`协议/成本模型`、
`可选平台实验`、`未覆盖`。标签是证据层级，不是平台 parity 承诺。
本页是面向读者的横向摘要，不替代
`planning/comparison/openmlsys-v1-crosswalk.md` 的逐文件映射；第 1–2 章的
接口、计算图和编程模型对照仍以对应章节和 crosswalk 为准。本页聚焦第
3–9 章中最容易把“概念、源码入口、协议模型”误读成“完整运行时”的主题。

## 第 3 章：GEMM 与加速器

- **原作问题**：加速器架构、线程/存储层次、GEMM 优化阶梯和设备性能。
- **OpenMLSys 文件**：`chapter_accelerator/accelerator_architecture.md`、
  `accelerator_programming.md`、`accelerator_practise.md`。
- **本书模型**：固定 shape、dtype、backend、warm-up、同步点、重复次数和
  host reference 的测量协议；算术强度只用来解释复用方向。
- **固定入口**：`cubecl/crates/cubecl-core/src/`、
  `cubecl/crates/cubecl-runtime/src/`、`cubek/crates/cubek-matmul/src/`。
- **可运行观察**：`ch03-cubecl-kernel` 验证 CPU Kernel 正确性；
  `ch03-tile-loads` 验证 tile load/intensity 计数。
- **不可直接比较**：CPU correctness 不能替代 GPU shared memory、带宽、
  autotune 或厂商 GEMM benchmark。标签为 `源码核验 + CPU 可运行验证 +
  协议/成本模型`；真实 GPU 是 `可选平台实验`。

## 第 4 章：IR、Fusion、cache 与 launch

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
  wall-clock time 不是同一指标；固定 API 不提供稳定私有 cache key。
  标签为 `源码核验 + CPU 可运行验证`，硬件 launch 是 `可选平台实验`。

## 第 5 章：数据处理

- **原作问题**：数据读取、顺序、shuffle、并行 worker、预取、背压和吞吐。
- **OpenMLSys 文件**：`chapter_data_processing/requirements.md`、
  `program_model.md`、`data_order.md`、`performance.md`。
- **本书模型**：用 `F/P/G` 表示 fetch/produce/consume，增加 deterministic
  shard/offset/decode、queue capacity、retry、epoch commit 和 reorder
  invariants。
- **固定入口**：`burn/crates/burn-dataset/src/` 和
  `burn/crates/burn-core/src/data/dataloader/`。
- **可运行观察**：`ch05-data-pipeline` 的 Dataset/Mapper/Batcher/DataLoader
  测试验证数据守恒、分片和背压协议；贯穿 capstone 把 Tensor batch 交给
  第 6 章训练。
- **不可直接比较**：内存样本和虚拟 queue 不代表磁盘、网络、pinned
  memory 或全局保序吞吐。标签为 `源码核验 + CPU 可运行验证 +
  协议/成本模型`。

## 第 6 章：分布式训练

- **原作问题**：数据/模型/流水线并行、collective、parameter server、
  stale gradient、quorum 和 checkpoint 一致性。
- **OpenMLSys 文件**：`chapter_distributed_training/methods.md`、
  `collective.md`、`parameter_servers.md`、`cluster.md`。
- **本书模型**：加权 AllReduce、版本化 stale gradient、quorum、1F1B
  bubble 和单调 checkpoint commit 的纯 Rust 协议卡。
- **固定入口**：`burn/crates/burn-train/src/`、
  `burn/crates/burn-communication/src/` 和
  `burn/crates/burn-core/src/tensor/distributed.rs`。
- **可运行观察**：`ch06-training-loop` 的纯 Rust 协议 helper 测试
  weighted average、staleness、quorum、pipeline slots 和 checkpoint
  version；另由 CPU autodiff loop 验证单设备训练。
- **不可直接比较**：协议结果不等于 DDP/NCCL/跨节点性能或故障恢复；
  Flex CPU collective 仍是 `未覆盖`，真实通信为 `可选平台实验`。

## 第 7 章：模型部署

- **原作问题**：转换、压缩、artifact、推理 runtime、安全、batching 和
  rollback。
- **OpenMLSys 文件**：`chapter_model_deployment/model_converter_and_optimizer.md`、
  `model_compression.md`、`model_inference.md`、`model_security.md`。
- **本书模型**：manifest 的 version/payload length/checksum、回滚条件和
  同 shape 动态 batching contract；主线 artifact 与 burn-onnx revision
  分开。
- **固定入口**：`burn/crates/burn-core/src/store/`、
  `burn/crates/burn-core/src/module/base.rs` 和
  `burn-onnx/crates/burn-import/src/`。
- **可运行观察**：`ch07-record-roundtrip` 验证 Burnpack round-trip；
  同 crate 的纯 Rust contract harness 用教学用的非密码学 checksum
  验证 payload length、版本、rollback 和 dynamic batch。
- **不可直接比较**：manifest/checksum 不是完整供应链安全；旧 revision
  的 ONNX fixture、HTTP、Remote、WASM 和 GPU service 是可选/未覆盖。

## 第 8 章：强化学习

- **原作问题**：MDP、trajectory、replay、on/off-policy、多环境、MARL 和
  Actor–Learner freshness。
- **OpenMLSys 文件**：`chapter_reinforcement_learning/rl_introduction.md`、
  `single_node_rl.md`、`marl.md`、`marl_sys.md`。
- **本书模型**：确定性 `MockPolicy`、behavior/target version lag、
  joint action/reward vector 和已验证的 done/truncated/TD 语义。
- **固定入口**：`burn/crates/burn-rl/src/`、
  `burn/crates/burn-train/src/learner/`。
- **可运行观察**：`ch08-rl-rollout` 的 CPU 环境/replay/TD 测试和协议卡；
  不依赖 gym 或外部 simulator。
- **不可直接比较**：mock policy 不是 DQN/PPO/SAC；joint vector 不是
  MARL credit assignment runtime；完整 Actor–Learner 是可选/未覆盖。

## 第 9 章：GPU 集群与控制面

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

## 与 P1 贯穿实验的关系

P1 贯穿实验把第 5–7 章串成一条真实 CPU-first 路径：
`Dataset → autodiff → ModuleRecord → inference`；它回答“状态怎样跨过
数据、训练和 artifact 边界”。本页的比较卡回答“同一主题在 OpenMLSys、
固定 Burn 源码、CPU 实验和协议模型之间分别有哪种证据”。前者是纵向学习
路径，后者是横向审计摘要，二者不能相互替代。

## 如何使用这些卡片

先读对应章节的框架无关模型，再运行卡片列出的 CPU 示例，最后回到
`planning/comparison/openmlsys-v1-crosswalk.md` 检查原作逐文件范围和
固定 revision。若一个结论没有同时标出证据标签、硬件前提和未覆盖边界，
它就不能作为本书的发布级比较结论。
