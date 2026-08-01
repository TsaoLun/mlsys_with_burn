# OpenMLSys v1 对照矩阵

本文件是“可比较”而不是“逐字翻译”的目录真相。它以固定
`openmlsys/v1/zh_chapters/SUMMARY.md` 为输入，把 OpenMLSys v1 的核心
机器学习系统路径映射到本书的九章、固定 Burn/CubeCL/CubeK 证据和可运行
观察。

## 判定方法

每个主题记录五类证据：

- 统一验收卡字段：`C/S/R/L/E`；
- **C（Correctness）**：原理、术语和能力边界正确；
- **S（Source）**：固定 OpenMLSys/Burn/CubeCL/CubeK 源码路径可定位；
- **R（Runnable）**：有 CPU 可运行实验，或明确是协议/成本模型；
- **L（Learning）**：前置状态、后续章节和贯穿 workflow 可追踪；
- **E（Engineering）**：导航、来源、许可证和构建可复核。

状态值使用：

- `verified`：当前快照和命令已经核验；
- `model`：框架无关模型或协议模拟，不代表真实 runtime；
- `source-only`：固定源码可定位，但当前 workspace 没有端到端实验；
- `excluded`：明确不进入九章主线；
- `optional`：需要真实 GPU、网络、旧 revision 或其他环境前提。

固定快照：

- OpenMLSys：`9c289782ccbb165ac8ad7c960ecffc12942a5560`
- Burn：`976aa9c5ec1d2dd3412710f99759e3c44bdff03d`
- CubeCL：`be278a1e76aed881e2cc6b165414ee6103ca4634`
- CubeK：`f82a6d07ebf35a1d446893b32712458744d80f13`
- burn-onnx：`af2dfb43af43bf363dc2d7d858d933d86e2a65a8`

## 核心路径

### 导论

- `openmlsys/v1/zh_chapters/chapter_introduction/index.md`
  → `book/src/ch01-introduction.md`；保留系统地图和阅读目标；证据
  `C=verified S=verified R=verified L=verified E=verified`。
- `chapter_introduction/applications.md`
  → `ch01/01-applications-and-loads.md`；保留应用到 workload card 的
  转换；新增 Burn/CubeCL 生命周期；`C/S/R/L=verified E=verified`。
- `chapter_introduction/design.md`
  → `ch01/02-design-goals.md`；保留设计目标和吞吐/内存权衡；`C/S/L=verified`，
  `R=model E=verified`。
- `chapter_introduction/architecture.md`
  → `ch01/03-system-architecture.md`；保留系统分层，改为
  Tensor→IR→Kernel→Runtime；`C/S/L=verified R=model E=verified`。
- `chapter_introduction/ecosystem.md`
  → `ch01/04-burn-stack.md`、`ch01/05-lifecycle-and-ecosystem.md`；
  保留生态角色和异构系统边界；`C/S/L=verified R=source-only E=verified`。
- `chapter_introduction/readers.md`
  → `ch01/05-lifecycle-and-ecosystem.md`；保留读者先修和路线；`C/S/L/E=verified R=model`。

### 编程接口

- `chapter_programming_interface/index.md`
  → `book/src/ch02-programming-and-graph.md`；合并编程接口与计算图入口；
  `C/S/L/E=verified R=verified`。
- `chapter_programming_interface/development_history.md`
  → `ch02/01-interface-and-workflow.md`；保留接口演进背景，减少历史展开；
  `C/S=verified L=verified R=source-only E=verified`。
- `chapter_programming_interface/ml_workflow.md`
  → `ch02/01-interface-and-workflow.md`，并连接 ch05/ch06/ch07 capstone；
  保留 Load→Map→Batch→Model→Loss→Update→Evaluate→Save；
  `C/S/L=verified R=verified E=verified`。
- `chapter_programming_interface/neural_network_layer.md`
  → `ch02/03-module-and-state.md`；改写为 Module/Param/visitor；
  `C/S/L=verified R=verified E=verified`。
- `chapter_programming_interface/c_python_interaction.md`
  → `ch02/01-interface-and-workflow.md` 的 Rust/CubeCL 扩展边界；
  不复制 Pybind 教程；`C/S=verified L=verified R=source-only E=verified`。
- `chapter_programming_interface/ml_programming_paradigm.md`
  → `ch02/01-interface-and-workflow.md`、`ch02/06-types-ir-scheduling.md`；
  以 ownership、trait 和 backend abstraction 重写；`C/S/L=verified R=model E=verified`。
- `chapter_programming_interface/summary.md`
  → `ch02/08-exercises-and-sources.md`；保留接口总结和延伸方向；
  `C/S/L/E=verified R=source-only`。

### 计算图

- `chapter_computational_graph/index.md`
  → `book/src/ch02-programming-and-graph.md`；合并入口；`C/S/L/E=verified R=verified`。
- `chapter_computational_graph/background_and_functionality.md`
  → `ch02/04-computational-graph.md`；保留图的设计动机；`C/S/L=verified R=model E=verified`。
- `chapter_computational_graph/components_of_computational_graph.md`
  → `ch02/04-computational-graph.md`；保留节点、边、依赖和控制流；
  `C/S/L=verified R=model E=verified`。
- `chapter_computational_graph/generation_of_computational_graph.md`
  → `ch02/04-computational-graph.md`、`ch02/05-autodiff.md`；区分 eager
  tape 和静态图；`C/S/L=verified R=verified E=verified`。
- `chapter_computational_graph/schedule_of_computational_graph.md`
  → `ch02/06-types-ir-scheduling.md`、第 4 章；后移编译器调度；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_computational_graph/summary.md`
  → `ch02/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 前端与 IR

- `chapter_frontend_and_ir/index.md`
  → `ch04-compiler-and-runtime.md`；与后端/运行时合并；`C/S/L/E=verified R=verified`。
- `chapter_frontend_and_ir/ai_compiler_design_principle.md`
  → `ch04/01-stack-and-ir.md`；保留编译因果链；`C/S/L=verified R=model E=verified`。
- `chapter_frontend_and_ir/overview_of_frontend.md`
  → `ch04/01-stack-and-ir.md`；保留前端表示和 lowering 边界；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_frontend_and_ir/intermediate_representation.md`
  → `ch04/03-burn-ir-and-fusion.md`；改写为 OperationIr/TensorIr/Fusion；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_frontend_and_ir/ad.md`
  → `ch02/05-autodiff.md`、`ch04/02-static-analysis-and-passes.md`；
  区分 autodiff tape 与 compiler IR；`C/S/L=verified R=verified E=verified`。
- `chapter_frontend_and_ir/type_system_and_static_analysis.md`
  → `ch04/02-static-analysis-and-passes.md`；补充 Pass 契约；
  `C/S/L=verified R=model E=verified`。
- `chapter_frontend_and_ir/common_frontend_optimization_pass.md`
  → `ch04/02-static-analysis-and-passes.md`、`ch04/03-burn-ir-and-fusion.md`；
  保留常量传播/DCE/CSE/Fusion 的原理，限定固定实现；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_frontend_and_ir/summary.md`
  → `ch04/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 后端与运行时

- `chapter_backend_and_runtime/index.md`
  → `ch04-compiler-and-runtime.md`；`C/S/L/E=verified R=verified`。
- `chapter_backend_and_runtime/overview.md`
  → `ch04/01-stack-and-ir.md`；`C/S/L=verified R=model E=verified`。
- `chapter_backend_and_runtime/graph_optimizer.md`
  → `ch04/04-graph-and-kernel-selection.md`；`C/S/L=verified R=source-only E=verified`。
- `chapter_backend_and_runtime/kernel_selecter.md`
  → `ch04/04-graph-and-kernel-selection.md`；保留候选/策略/fallback；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_backend_and_runtime/memory_allocator.md`
  → `ch04/06-memory-streams-execution.md`；保留状态、复用、pool 和 buffer；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_backend_and_runtime/compute_schedule_and_execute.md`
  → `ch04/05-cubecl-lowering-and-jit.md`、`ch04/06-memory-streams-execution.md`；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_backend_and_runtime/op_compiler.md`
  → `ch04/05-cubecl-lowering-and-jit.md`；保留编译链，未声称通用 AOT；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_backend_and_runtime/summary.md`
  → `ch04/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 加速器

- `chapter_accelerator/index.md`
  → `ch03-accelerator.md`；`C/S/L/E=verified R=verified`。
- `chapter_accelerator/accelerator_introduction.md`
  → `ch03/01-workloads-and-design.md`；`C/S/L=verified R=model E=verified`。
- `chapter_accelerator/accelerator_architecture.md`
  → `ch03/02-gpu-machine-model.md`；保留存储层次和并行域；
  `C/S/L=verified R=model E=verified`。
- `chapter_accelerator/accelerator_programming.md`
  → `ch03/03-cubecl-programming.md`；以 Rust/CubeCL 重写；
  `C/S/L=verified R=verified E=verified`。
- `chapter_accelerator/accelerator_practise.md`
  → `ch03/05-gemm-optimization.md`、`ch03/07-cpu-kernel-lab.md`；
  保留 GEMM/tiling/测量协议，真实 GPU 为 optional；
  `C/S/L=verified R=model E=verified`。
- `chapter_accelerator/summary.md`
  → `ch03/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 数据处理

- `chapter_data_processing/index.md`
  → `ch05-data-processing.md`；`C/S/L/E=verified R=verified`。
- `chapter_data_processing/requirements.md`
  → `ch05/01-data-pipeline-and-cost.md`；保留 Load/Shuffle/Map/Batch/Send；
  `C/S/L=verified R=verified E=verified`。
- `chapter_data_processing/program_model.md`
  → `ch05/02-dataset-abstractions.md`、`ch05/03-batching-and-device.md`；
  `C/S/L=verified R=verified E=verified`。
- `chapter_data_processing/performance.md`
  → `ch05/01-data-pipeline-and-cost.md`、`ch05/06-storage-and-scaling.md`；
  保留 F/P/G、分片、索引和背压；`C/S/L=verified R=model E=verified`。
- `chapter_data_processing/data_order.md`
  → `ch05/05-multithread-and-order.md`；明确 Burn 多 worker 不保证全局保序；
  `C/S/L=verified R=verified E=verified`。
- `chapter_data_processing/extension.md`
  → `ch05/06-storage-and-scaling.md`、`ch05/08-exercises-and-sources.md`；
  `C/S/L=verified R=source-only E=verified`。
- `chapter_data_processing/summary.md`
  → `ch05/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 模型部署

- `chapter_model_deployment/index.md`
  → `ch07-model-serving.md`；`C/S/L/E=verified R=verified`。
- `chapter_model_deployment/model_deployment_introduction.md`
  → `ch07/01-deployment-boundary.md`；`C/S/L=verified R=model E=verified`。
- `chapter_model_deployment/model_converter_and_optimizer.md`
  → `ch07/02-onnx-and-codegen.md`、`ch07/04-compression-and-optimization.md`；
  ONNX/图优化仅在固定 burn-onnx 源码级核验；`C/S/L=verified R=source-only E=verified`。
- `chapter_model_deployment/model_compression.md`
  → `ch07/04-compression-and-optimization.md`；保留 PTQ/QAT/稀疏/蒸馏原理；
  runtime 为未覆盖；`C/S/L=verified R=model E=verified`。
- `chapter_model_deployment/model_inference.md`
  → `ch07/05-inference-runtime-and-service.md`；保留 latency/throughput/
  batching/layout；服务实现为 model；`C/S/L=verified R=source-only E=verified`。
- `chapter_model_deployment/model_security.md`
  → `ch07/03-record-and-artifacts.md`、`ch07/08-exercises-and-sources.md`；
  保留四层威胁模型；`C/S/L=verified R=model E=verified`。
- `chapter_model_deployment/summary.md`
  → `ch07/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 分布式训练

- `chapter_distributed_training/index.md`
  → `ch06-training-systems.md`、`ch09-gpu-cluster.md`；拆分训练数据面和
  集群控制面；`C/S/L/E=verified R=verified`。
- `chapter_distributed_training/overview.md`
  → `ch06/01-training-state-and-cost.md`、`ch09/01-cluster-workload-and-boundary.md`；
  `C/S/L=verified R=model E=verified`。
- `chapter_distributed_training/methods.md`
  → `ch06/05-local-data-parallel.md`、`ch09/04-topology-aware-placement-and-communication.md`；
  `C/S/L=verified R=model E=verified`。
- `chapter_distributed_training/cluster.md`
  → `ch09/02-gpu-node-and-network-topology.md`；`C/S/L=verified R=model E=verified`。
- `chapter_distributed_training/collective.md`
  → `ch06/06-collective-and-ddp.md`、`ch09/04-topology-aware-placement-and-communication.md`；
  `C/S/L=verified R=model E=verified`。
- `chapter_distributed_training/parameter_servers.md`
  → `ch06/06-collective-and-ddp.md`、`ch09/06-faults-checkpoints-and-observability.md`；
  保留 stale gradient/副本/一致性；runtime 未覆盖；
  `C/S/L=verified R=model E=verified`。
- `chapter_distributed_training/summary.md`
  → `ch09/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

### 强化学习

- `chapter_reinforcement_learning/index.md`
  → `ch08-rl-systems.md`；`C/S/L/E=verified R=verified`。
- `chapter_reinforcement_learning/rl_introduction.md`
  → `ch08/01-mdp-environment-and-trajectory.md`；`C/S/L=verified R=verified E=verified`。
- `chapter_reinforcement_learning/single_node_rl.md`
  → `ch08/02-policy-and-batching.md`、`ch08/03-replay-and-sampling.md`、
  `ch08/04-rollout-throughput.md`、`ch08/05-learning-and-off-policy.md`；
  `C/S/L=verified R=verified E=verified`。
- `chapter_reinforcement_learning/marl.md`
  → `ch08/06-multi-agent-boundary.md`；保留联合动作/奖励/非平稳性；
  通用 MARL API 未覆盖；`C/S/L=verified R=model E=verified`。
- `chapter_reinforcement_learning/marl_sys.md`
  → `ch08/04-rollout-throughput.md`、`ch08/06-multi-agent-boundary.md`；
  Actor–Learner/league 作为系统对照；`C/S/L=verified R=source-only E=verified`。
- `chapter_reinforcement_learning/summary.md`
  → `ch08/08-exercises-and-sources.md`；`C/S/L/E=verified R=source-only`。

## 明确排除的 OpenMLSys 范围

以下文件属于 OpenMLSys v1 的扩展篇/附录（包括推荐系统、联邦学习、可解释
AI、机器人和机器学习附录），不进入本书首版九章主线；排除是范围决策，
不是 Burn 能力缺口：

- `chapter_preface/index.md`
- `chapter_preface_advanced/index.md`
- `chapter_preface_extension/index.md`
- `chapter_recommender_system/index.md`
- `chapter_recommender_system/system_architecture.md`
- `chapter_recommender_system/multi_stage_recommender_system.md`
- `chapter_recommender_system/model_update.md`
- `chapter_recommender_system/case_study.md`
- `chapter_recommender_system/summary.md`
- `chapter_federated_learning/index.md`
- `chapter_federated_learning/overview.md`
- `chapter_federated_learning/horizontal_fl.md`
- `chapter_federated_learning/vertical_fl.md`
- `chapter_federated_learning/privacy_encryption_algorithm.md`
- `chapter_federated_learning/outlook.md`
- `chapter_federated_learning/summary.md`
- `chapter_explainable_AI/index.md`
- `chapter_explainable_AI/explainable_ai.md`
- `chapter_rl_sys/index.md`
- `chapter_rl_sys/rl_sys_intro.md`
- `chapter_rl_sys/robot_learning.md`
- `chapter_rl_sys/perception.md`
- `chapter_rl_sys/perception_code_ex.md`
- `chapter_rl_sys/planning.md`
- `chapter_rl_sys/planning_code_ex.md`
- `chapter_rl_sys/control.md`
- `chapter_rl_sys/control_code_ex.md`
- `chapter_rl_sys/robot_safety.md`
- `chapter_rl_sys/ros.md`
- `chapter_rl_sys/ros_code_ex.md`
- `chapter_rl_sys/summary.md`
- `appendix_machine_learning_introduction/index.md`
- `appendix_machine_learning_introduction/neural_network.md`
- `appendix_machine_learning_introduction/gradient_descent.md`
- `appendix_machine_learning_introduction/classic_machine_learning.md`

这些排除项在本书 `CHAPTER_MATRIX.md` 和发布说明中必须保持可见，不能让
读者误解为本书声称覆盖 OpenMLSys v1 全部内容。

## 固定源码入口

下面是本书反复引用、可由 `pins.toml` 定位的实现入口。它们是源码证据
入口，不表示每个入口都已经提供可运行的端到端平台实现：

- Tensor/Module/Record：`burn/crates/burn-core/src/tensor.rs`、
  `burn/crates/burn-core/src/module/`、`burn/crates/burn-core/src/store/`；
- Dataset/DataLoader：`burn/crates/burn-dataset/src/`；
- 训练与分布式：`burn/crates/burn-train/src/`、
  `burn/crates/burn-communication/src/`；
- IR/Fusion：`burn/crates/burn-ir/src/`、`burn/crates/burn-fusion/src/`、
  `burn/crates/burn-cubecl-fusion/src/`；
- CubeCL runtime：`cubecl/crates/cubecl-core/src/`、
  `cubecl/crates/cubecl-runtime/src/`；
- CubeK 算子：`cubek/crates/cubek-matmul/src/`、
  `cubek/crates/cubek-reduce/src/`、`cubek/crates/cubek-attention/src/`；
- RL：`burn/crates/burn-rl/src/`；
- ONNX：`burn-onnx/crates/burn-import/src/`、
  `burn-onnx/crates/burn-import/Cargo.toml`；
- 集群相关的 Burn 数据面：`burn/crates/burn-communication/src/` 和
  `burn/crates/burn-train/src/learner/supervised/strategies/`。

## 统一验收

对照完成后应通过：

```text
python3 tools/check_release.py
mdbook build book
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --all-targets --locked --offline
```

本矩阵本身是人工审阅记录；`tools/check_release.py` 只验证路径、格式和
结构，不把 `verified` 字样当作运行证据。运行证据必须来自对应示例测试、
固定源码或会话日志。
