# 来源与改编总录

本附录汇总各章对 OpenMLSys v1 的文件级改编说明。正文练习页只保留一句指针。许可见[许可、来源与独立性声明](attribution.md)；逐文件总账见[范围、证据与对照](appendix-scope-and-evidence.md)。

## 第 1 章

本章改编并重组了 OpenMLSys v1 以下文件：

- `chapter_introduction/index.md`
- `chapter_introduction/applications.md`
- `chapter_introduction/design.md`
- `chapter_introduction/architecture.md`
- `chapter_introduction/ecosystem.md`
- `chapter_introduction/readers.md`

保留的核心思想包括机器学习应用分类、框架六类设计目标、从接口到硬件的
系统分层和框架中心生态。主要修改包括：

- 将应用枚举改写为计算、数据、硬件与生命周期负载分析；
- 删除以 Python、MindSpore、Ascend 为默认栈的表述；
- 用 Burn 0.22 的 Device/Dispatch 架构映射框架层次；
- 增加 CubeCL、CubeK、burn-onnx 的职责与版本边界；
- 按本书九章重写范围和阅读路径；
- 新增固定源码快照方法、实验与练习。

本章没有复用 OpenMLSys 的 `framework-architecture.png` 和
`system-ecosystem.png` 图面；文本架构图是基于通用分层思想重新设计，并与
第 2、4 章与[术语表](glossary.md)使用同一套层名。

未迁入：原书以 Python/MindSpore/Ascend 为默认栈的图示与生态叙述。

OpenMLSys 原作及本章改编正文采用 CC BY-NC-SA 4.0。完整署名与许可证见
本书的“许可、来源与独立性声明”和仓库根目录 `NOTICE.md`。

## 第 2 章

本章改编并重组 OpenMLSys v1：

### 编程接口

- `chapter_programming_interface/index.md`
- `chapter_programming_interface/development_history.md`
- `chapter_programming_interface/ml_workflow.md`
- `chapter_programming_interface/neural_network_layer.md`
- `chapter_programming_interface/ml_programming_paradigm.md`

`c_python_interaction.md` 提供“高层接口与底层实现存在边界”的背景。本章
保留这一系统问题，改用 Rust trait、Device dispatch、CubeCL Kernel 和
显式错误/所有权边界解释扩展路径；没有复用其 Pybind11、MindSpore 或 CUDA
示例。自定义 Kernel 的 launch、lowering 和 Runtime 细节放到第 3、4 章。

### 计算图

- `chapter_computational_graph/background_and_functionality.md`
- `chapter_computational_graph/components_of_computational_graph.md`
- `chapter_computational_graph/generation_of_computational_graph.md`
- `chapter_computational_graph/schedule_of_computational_graph.md`

本章保留图表示、依赖、控制流、静动态图与拓扑调度思想，删除 TensorFlow 1
和 MindSpore 专用 API，并把数据流水线、模型并行后移。补全时增加了拓扑序
小例子、图外控制流与循环展开/循环依赖区分，以及两分支 autodiff 实验；
统一用语见[术语表](glossary.md)。

未迁入：原书长控制流教程、`tf.cond`/`while_loop` API 走读和框架专用训练
代码；完整训练工作流已用输入/输出/状态契约建立地图，数据与训练执行仍
分别后移第 5/6 章。

### 自动微分、类型与 IR

- `chapter_frontend_and_ir/ad.md`
- `chapter_frontend_and_ir/intermediate_representation.md`
- `chapter_frontend_and_ir/type_system_and_static_analysis.md`

本章保留求导方法、前向/反向模式和 IR 的通用定义；MindIR、框架前端 pass
和 MLIR 深入内容后移第 4 章。新增 Burn 0.22 Device/autodiff 动态 tape、
Rust 类型分工和全部可运行示例。

本章没有复制 OpenMLSys 图面。完整逐文件映射见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。OpenMLSys 原作和本章改编正文采用
CC BY-NC-SA 4.0，原创 Rust 示例采用 MIT OR Apache-2.0。

## 第 3 章

本章改编并重组 OpenMLSys v1 `chapter_accelerator/`：

- `index.md`
- `accelerator_introduction.md`
- `accelerator_architecture.md`
- `accelerator_programming.md`
- `accelerator_practise.md`
- `summary.md`

保留了加速器设计、GPU 存储层次、三级编程抽象、GEMM 公式以及 tiling、
向量化、共享内存和流水线的通用思想。Volta、Ascend、cuBLAS、WMMA、PTX、
TBE/AKG 与 RTX 3080 性能结果被压缩为历史或生态边界。

本章没有复制 OpenMLSys CUDA C++ 示例、`openmlsys-cuda` 代码或缺失图片；
全部实验改为固定 CubeCL revision 上的原创 Rust 代码。新增 CubeCL Runtime、
unsafe 合约、CubeK 分层、Burn 集成、fallback、autotune，以及 host 侧
`tile_load_counts` 加载模型（明确非真实共享内存）。术语见
[术语表](glossary.md)。

未迁入：完整 CUDA GEMM 阶梯实现、设备榜单式结论、厂商指令内联汇编教程。

OpenMLSys v2 固定版本只列出 GPU/CUDA/Triton/CUTLASS TODO，没有可迁移
正文。完整逐文件与源码事实映射见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。
OpenMLSys 原作和改编正文采用 CC BY-NC-SA 4.0，原创 Rust 示例采用
MIT OR Apache-2.0。

## 第 4 章

本章重组 OpenMLSys v1 两章内容：

### `chapter_frontend_and_ir/`

- `index.md`
- `overview_of_frontend.md`
- `ai_compiler_design_principle.md`
- `intermediate_representation.md`
- `ad.md`
- `type_system_and_static_analysis.md`
- `common_frontend_optimization_pass.md`
- `summary.md`

保留 IR 分类、多层编译、静态信息、经典 Pass 与 AD/IR 关系。TensorFlow、
JAX、MindIR 与 MindSpore 自动微分实现只作为原始范围核对，没有迁移其长
代码和框架结构图。

### `chapter_backend_and_runtime/`

- `index.md`
- `overview.md`
- `graph_optimizer.md`
- `kernel_selecter.md`
- `memory_allocator.md`
- `compute_schedule_and_execute.md`
- `op_compiler.md`
- `summary.md`

保留融合、layout/dtype、Kernel 选择、生命周期、内存池、异步调度和
compute/schedule 思想。MindSpore Graph Kernel、SOMAS、Ascend task 下沉
及厂商布局约束未映射为 Burn 能力；通信 buffer 内容后移第 6 章。

OpenMLSys v2 固定版本只有第 4 章 TODO，没有可迁移正文。本章没有复制
OpenMLSys ch04/ch05 图片或 Python/C++ 示例。新增 Burn OperationIr/Fusion、
CubeCL lowering/JIT/stream 内容，以及常量传播→DCE 手推、生命周期条带图
与三操作 Fusion 断言。术语见[术语表](glossary.md)。

未迁入：MindSpore Graph Kernel / SOMAS 实现细节、Ascend task 下沉、长
TVM schedule 教程（仅延伸阅读对照）。

Rust 实验参考固定 Burn `fusion_shape.rs` 的 add→exp 与同步切分回归模式，
重新设计了独立 Stream、可传播错误、稳定 summary、教学输出和双重结构/数值
断言。

完整逐文件和固定源码映射见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。
OpenMLSys 原作和改编正文采用 CC BY-NC-SA 4.0，原创 Rust 示例采用
MIT OR Apache-2.0。

## 第 5 章

本章改编并重组 OpenMLSys v1 的
`chapter_data_processing/`：

- `index.md`：保留数据模块的问题定义和学习目标，改为本书的 Rust/Burn
  阅读路线；
- `requirements.md`：保留 Load、Shuffle、Map、Batch、Send 与易用性、
  高效性、保序性三维框架；
- `program_model.md`：保留 Dataset 变换和自定义算子抽象，删除
  MindData、Spark 和长 Python 代码，改写为 `Dataset`、`Mapper`、
  `Batcher`；
- `performance.md`：保留 $F/P/G$ 成本模型、随机访问、异步生产消费和
  流水线/算子并行对照，删除 MindRecord/Unirecord 和厂商性能结论；
- `data_order.md`：保留保序问题与 Connector 的设计动机，明确其只是
  Burn 的对照概念，本版没有对应的序号等待实现；
- `extension.md`：保留 CPU 瓶颈、异构和分布式扩展的系统问题，改为
  边界与未来工作，不宣称 Burn 已提供通用异构数据预处理；
- `summary.md`：重写为本章的 Dataset/DataLoader 结论与验证边界。

OpenMLSys v2 固定版本的第 5 章仍是 TODO；本章依据 v1 中文文件。原章
引用的框架专用图片在固定 clone 中没有可复用的图像资源，本章没有复制
图片或 MindSpore/PyTorch/C++ 示例，结构关系使用原创文本图。

完整逐文件映射、固定 Burn 源码定位和不作出的能力承诺见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。OpenMLSys 原作和本章改编正文采用
CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT OR Apache-2.0。

## 第 6 章

本章改编并重组 OpenMLSys v1 的
`chapter_distributed_training/`：

- `index.md`：保留动机和并行方法地图，改为本书第 6 章的训练状态路线；
- `overview.md`：保留算力、内存、分而治之和 time-to-accuracy 问题，删除
  固定历史硬件数字与原图编号；
- `methods.md`：保留数据/模型/混合/流水线并行的语义，具体 Burn 只实现
  已核验的本机多设备和 DDP API；
- `collective.md`：保留集合通信算子、$\alpha+\beta l$ 成本和梯度平均，
  以 `DistributedContext`、backend `all_reduce` 和同步边界重写；
- `parameter_servers.md`：保留同步/异步、straggler、热点和副本一致性，
  明确固定 Burn 没有对应的 `burn-train` strategy；
- `cluster.md`：保留通信层次和带宽瓶颈，将调度、遥测、容错后移第 9 章；
- `summary.md`：重写为经过源码核验的能力清单。

没有复制 OpenMLSys 的 MindSpore、TensorFlow、PyTorch、Gloo、NCCL 代码或
章节图片；跨系统代码只在解释接口边界时以文字提及。完整逐文件核验、
Burn 版本定位和未承诺能力见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。OpenMLSys 改编正文采用
CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT OR Apache-2.0。

## 第 7 章

本章改编并重组 OpenMLSys v1 的 `chapter_model_deployment/`：

- `index.md`：保留训练到部署的主问题和学习目标，改为 artifact/runtime/
  service/policy 四层路线；
- `model_deployment_introduction.md`：保留转换、常量折叠、融合、数据
  排布和安全的系统动机，删除未经固定 Burn 验证的厂商实现结论；
- `model_converter_and_optimizer.md`：保留 ONNX 图/算子映射和离线优化，
  以 `ModelGen`、`BurnGraph`、Rust codegen 和 Burnpack 重写；
- `model_compression.md`：保留 PTQ/QAT、稀疏、剪枝和蒸馏的原理，明确
  本书所用的 Burn 版本中的 QAT 与通用量化流水线边界；
- `model_inference.md`：保留前/后处理、并行、访存和延迟问题，改为
  Burn Device/runtime 与应用 batcher 的边界；
- `model_security.md`：保留静态/动态保护和威胁模型，区分 artifact、
  transport、Remote authorization 与 TEE/混淆；
- `summary.md`：重写为固定源码证据和本章实验边界。

没有复制 OpenMLSys 的 MindSpore/PyTorch/ARM 汇编代码、图片或 Mate30
性能数字。完整 revision 关系、逐文件核验和不作出的能力承诺见
[范围、证据与对照附录](appendix-scope-and-evidence.md)。本书把 burn-onnx 与当前 workspace 的 Burn 版本分开验证，避免混用未对齐依赖。OpenMLSys 改编正文采用 CC BY-NC-SA 4.0；新增 Rust
示例采用 MIT OR Apache-2.0。

## 第 8 章

本章改编并重组 OpenMLSys v1 的
`chapter_reinforcement_learning/`：

- `index.md`：保留基础、单节点/分布式和多智能体的学习地图，改成
  本章的环境→采样→更新→系统边界路线；
- `rl_introduction.md`：保留 Agent/Environment、state/observation、
  action/reward、MDP、Markov property 和 discounted return，改用 Rust
  `Environment`/`StepResult` 与 `done`/`truncated` 解释；
- `single_node_rl.md`：保留 policy/value、adapter、learner、replay、
  online/offline 和多环境采样，改为 `burn-rl` traits、`TransitionBuffer`
  和 `burn-train` off-policy pipeline；
- `marl.md`：保留联合动作、奖励向量、合作/竞争/self-play 与非平稳性，
  明确固定 Burn 当前没有通用 MARL API；
- `marl_sys.md`：保留 Actor/Learner、league、模型评估/选择和 inference
  server 的系统问题，改为能力边界与未来协议，而不是现成 Burn runtime；
- `summary.md`：重写为采样吞吐、设备协同、checkpoint 和可复现性的核验
  清单。

没有复制 OpenMLSys 的图、框架专用代码、外部 simulator 或硬件性能数字。
[范围、证据与对照附录](appendix-scope-and-evidence.md) 记录逐文件来源、固定 Burn 路径和
实验范围。本章实验停在确定性环境与 replay/TD，不扩展成完整 DQN/MARL。
OpenMLSys 改编正文采用 CC BY-NC-SA 4.0；新增 Rust 示例采用 MIT
OR Apache-2.0。

## 第 9 章

本章改编并重组 OpenMLSys v1 `chapter_distributed_training/` 中的系统概述、
并行方法、集合通信、参数服务器和集群架构内容。新增的队列、配额、故障域、
遥测字段和 CPU 模拟器是框架无关的系统设计材料；固定 Burn/CubeCL 源码
只用于核验设备、通信、stream、内存和训练入口的边界。本章没有复用上游
硬件图片或历史性能数字。

本章保留 OpenMLSys 的分布式训练、拓扑、集合通信、参数服务器和故障
问题，重写为“workload card → control plane → collective data plane →
device runtime”的路线。Burn 部分改为固定源码证据和限制清单；没有把
`ExecutionStrategy`、`DistributedContext` 或 `ComputeClient` 称为集群
调度器。实验和新增 Rust 代码采用 MIT OR Apache-2.0；正文采用
CC BY-NC-SA 4.0。
