# 第 1–8 章原作对照与回补审计

## 审计范围与判定标准

本审计以 `pins.toml` 固定的 OpenMLSys v1
`9c289782ccbb165ac8ad7c960ecffc12942a5560`、Burn
`976aa9c5ec1d2dd3412710f99759e3c44bdff03d`、CubeCL
`be278a1e76aed881e2cc6b165414ee6103ca4634` 和 CubeK
`f82a6d07ebf35a1d446893b32712458744d80f13` 为准。这里的“浅”不是按
Markdown 行数判断，而是指以下任一闭环缺失：

1. 原作的系统问题只被列名，没有成本、约束或不变量；
2. 原理没有连接到当前固定源码中的具体抽象；
3. Burn/CubeCL 的能力边界没有和原作框架能力分开；
4. 读者没有可运行观察，或实验观察量与正文问题不对应；
5. 小节之间缺少状态、数据、表示或设备边界，无法形成连续学习路径。

## 总体结论

- 第 1、2、4 章已经有较好的纵向闭环，但第 2 章仍压缩了原作的完整
  ML workflow 和高层接口到低层扩展边界。
- 第 3 章的原理阶梯完整，主要缺口是原作 GEMM 实践中的 roofline/计算
  强度、调度决策和测量协议没有形成足够可操作的 CPU 对照。
- 第 5 章准确记录了 Burn DataLoader 的边界，但原作的数据存储索引、
  producer/consumer 背压和流水线/算子级并行仍可更具体。
- 第 6 章对 Burn DDP 的证据最完整，反而需要补上原作的模型/流水线/混合
  并行和参数服务器系统叙事，避免“未实现”变成“未解释”。
- 第 7 章已经明确 revision 偏差，但量化校准、稀疏加速条件、推理线程池
  和威胁模型仍偏概述；这些可以补原理，不能伪造 Burn 端到端支持。
- 第 8 章的系统边界完整，但 RL 算法概念从 MDP 直接跳到 DQN/TD；需要
  补 return/value/policy、Monte Carlo/TD、探索和数据分布的桥接。
- 第 9 章应继续负责集群控制面、拓扑、调度和容错，不应在本次回补中
  把第 6 章未实现的集群能力写成 Burn 已有功能。

## 逐章缺口矩阵

### 第 1 章：导论

**来源**：

- `openmlsys/v1/zh_chapters/chapter_introduction/applications.md`
- `design.md`
- `architecture.md`
- `ecosystem.md`
- `readers.md`
- `index.md`

**当前状态**：

- 已覆盖监督/无监督/RL、计算/数据/硬件/生命周期负载、系统分层、
  Burn 生态和固定快照阅读方法。
- 已有 `ch01-stack-probe`，能观察 pin、Flex 和执行栈入口。

**回补项**：

- 增加“同一模型在训练、离线推理、在线服务、边缘设备中的约束变化”
  对照，明确吞吐、尾延迟、内存、能耗、精度和恢复性的冲突。
- 增加一个从数据到 kernel、训练、artifact、服务的贯穿案例，作为第 2–8
  章的回链索引。
- 将“框架层/运行时层/硬件层”与本书三张地图逐一对齐，避免第 1 章
  的“后端”与后文 Backend/Device 混用。

**等级**：中。不是结构缺失，而是全书导航和系统权衡还可以更厚。

### 第 2 章：编程接口与计算图

**来源**：

- `chapter_programming_interface/ml_workflow.md`
- `neural_network_layer.md`
- `ml_programming_paradigm.md`
- `c_python_interaction.md`
- `chapter_computational_graph/`
- `chapter_frontend_and_ir/ad.md`
- `intermediate_representation.md`
- `type_system_and_static_analysis.md`

**当前状态**：

- 已覆盖 `Tensor<D, K>`、Device/Dispatch、Module/Param/Config、
  ModuleRecord、autodiff tape、控制流、三种表示和分支梯度实验。
- `ch02/01` 只有工作流的接口切面，数据和训练细节被后移是合理的。

**回补项**：

- 将 data → batch → model → loss → optimizer → train → evaluate →
  save/debug 写成一条有输入输出契约的工作流，标出第 5、6、7 章的接点。
- 补充 Layer/Module/参数 visitor 的组合关系和“普通 Tensor 为什么不是
  parameter”的生命周期例子。
- 增加 Rust 前端与低层 custom op/kernel 的 ABI、所有权、错误和 feature
  边界；以 Burn/CubeCL 的真实入口替换 Pybind11 教程，而不是删除问题。
- 增加 shape/dtype/device/ownership 四类错误的诊断路径，并在实验中至少
  观察一次编译期错误与一次运行时错误。

**等级**：高。这里是基础篇进入系统篇的接口桥梁。

### 第 3 章：AI 加速器与编程

**来源**：

- `chapter_accelerator/accelerator_introduction.md`
- `accelerator_architecture.md`
- `accelerator_programming.md`
- `accelerator_practise.md`

**当前状态**：

- 已覆盖 Host/Device、Cube/Unit/Plane、存储层次、GEMM、tile、
  vector、流水线、CubeCL/CubeK 和 kernel 正确性。
- `ch03-cubecl-kernel` 验证 CPU/可选 WGPU kernel，`ch03-tile-loads`
  验证 host 侧加载次数模型。

**回补项**：

- 增加 roofline 形式的计算强度、峰值计算/带宽上限和“为何测量不能
  只看 FLOPS”的数值例子。
- 将原作 naive → thread tile → shared tile → double buffer → matrix
  instruction 的决策条件写成逐步不变量：复用、寄存器、共享存储、
  occupancy、边界和同步。
- 补充 layout/向量化/矩阵 fragment 的兼容条件及 fallback，而不把
  CubeCL 的设备无关语法误写为所有设备相同的硬件语义。
- 让 CPU 加载模型与理论公式有一一对应的报告字段；GPU/共享内存仍标为
  后续平台实验。

**等级**：高。原作的实践细节多，当前主要停在概念图。

### 第 4 章：AI 编译器与运行时

**来源**：

- `chapter_frontend_and_ir/`
- `chapter_backend_and_runtime/`

**当前状态**：

- 已覆盖 autodiff tape、OperationIr/Fusion、CubeCL Scope、
  pass、kernel 选择、内存、stream、lowering、JIT/cache 和 FusionInspector。
- 已明确 Flex 不经过 Fusion，且 inspector block 数不是硬件 launch 数。

**回补项**：

- 为每个 Pass 添加输入不变量、等价性条件和失败/回退路径，尤其是
  constant propagation、DCE、CSE、fusion、layout/dtype 特化。
- 补充线性 IR、图 IR、混合 IR 与 SSA/CFG 的最小手推例子，说明它们
  如何影响调度和代码生成，而不是只做名称对照。
- 将 kernel selection → strategy filter → compile key → JIT/cache →
  launch/read/sync 串成一条带成本的时间线。
- 说明内存生命周期、alias/in-place、workspace 和 stream 依赖如何限制
  融合；现有 FusionInspector 实验只负责结构观察。

**等级**：中高。已有内容较厚，重点是因果链和不变量补全。

### 第 5 章：数据处理系统

**来源**：

- `chapter_data_processing/requirements.md`
- `program_model.md`
- `performance.md`
- `data_order.md`
- `extension.md`
- `summary.md`

**当前状态**：

- 已覆盖 Dataset、惰性 map、Selection/Shuffle/Sampler、Batcher、
  DataLoader、seed、multi-worker、SQLite、设备传递和到达顺序边界。
- 已有 F/P/G 的粗粒度模型和 CPU 数据管道实验。

**回补项**：

- 补充统一文件格式/索引块/数据块/分片的框架无关设计，明确 Burn
  当前没有通用 Unirecord/MindRecord 等格式。
- 把 producer/consumer 队列容量、背压、prefetch、缓存命中和内存峰值
  与 F/P/G 公式接起来。
- 解释流水线级并行、算子级并行、map/batch 向量化和数据图融合的差异，
  不把多 worker DataLoader 等同于任意算子并行。
- 加入失败重试、epoch 生命周期、重复/缺失样本和数据版本的协议检查。

**等级**：中高。Burn API 核验充分，原作性能设计仍偏薄。

### 第 6 章：训练系统

**来源**：

- `chapter_distributed_training/overview.md`
- `methods.md`
- `collective.md`
- `parameter_servers.md`
- `cluster.md`
- `summary.md`

**当前状态**：

- 已覆盖训练状态、TrainStep、Learner、optimizer/scheduler/checkpoint、
  SingleDevice/MultiDevice、DDP、CollectiveTensor 和 Flex 限制。
- 已有 CPU autodiff/SGD 实验和 D009。

**回补项**：

- 系统解释数据并行、模型并行、算子内/算子间并行、混合并行的内存/
  通信动机和适用条件。
- 增加 pipeline micro-batch、forward/backward schedule、bubble、
  activation cache/recomputation 的最小时间线。
- 扩展参数服务器为 push/pull、同步/异步、stale gradient、straggler、
  hot shard、leader/follower 和一致性/可用性权衡。
- 逐项再次标注固定 Burn 只验证 DDP API/backend collective 入口，不承诺
  pipeline scheduler、parameter server、elastic membership、集群调度
  或跨节点 checkpoint 共识。

**等级**：高。不是 Burn API 缺失，而是原作的系统设计被“未实现”压缩了。

### 第 7 章：模型服务

**来源**：

- `chapter_model_deployment/model_deployment_introduction.md`
- `model_converter_and_optimizer.md`
- `model_compression.md`
- `model_inference.md`
- `model_security.md`
- `summary.md`

**当前状态**：

- 已覆盖 artifact/runtime/service/security 边界、ONNX codegen、
  ModuleRecord/Burnpack、压缩总览、Remote、WASM/no_std 和 CPU round-trip。
- D010 已正确隔离 `burn-onnx` 旧 Burn revision。

**回补项**：

- 给 PTQ/QAT、对称/非对称、逐层/逐通道、校准和误差度量补公式与
  reference workflow，明确当前快照的量化边界。
- 区分结构化/非结构化稀疏的准确率、内存和 kernel 加速条件；补充蒸馏
  的 teacher/student loss 与部署 topology 变化。
- 将前处理、后处理、线程池、动态 batch、layout/Img2col/Winograd
  作为框架无关推理系统原理，并标注固定 Burn 没有统一服务框架。
- 增加 artifact manifest、schema/checksum/version/rollback 和模型
  安全威胁的最小协议；不把 Record、TLS、Remote authorization 和 TEE
  混成一个开关。

**等级**：中高。原理有骨架，压缩/推理/安全的工程细节还可补。

### 第 8 章：强化学习系统

**来源**：

- `chapter_reinforcement_learning/rl_introduction.md`
- `single_node_rl.md`
- `marl.md`
- `marl_sys.md`
- `summary.md`

**当前状态**：

- 已覆盖 MDP、Environment/StepResult、Policy/Batchable、
  TransitionBuffer、AsyncPolicy、OffPolicyStrategy、多环境和
  Actor–Learner/MARL 边界。
- 已有确定性 CPU rollout/replay/表格 TD 实验和 D011。

**回补项**：

- 在 MDP 与 DQN 之间补 return/value function/policy、Monte Carlo/TD、
  Q-learning/policy-gradient、exploration/exploitation 和 advantage 的
  概念桥。
- 区分 transition、trajectory、on-policy batch、off-policy replay、
  behavior/target policy 和 terminal/truncated bootstrap。
- 增加 rollout throughput、policy version、replay staleness、checkpoint/
  RNG 的可复现协议。
- 加深 MARL 的联合动作/观察、奖励向量、合作/竞争/混合博弈、self-play
  非平稳性和 league/evaluation 结构，同时保持固定 Burn 未提供这些
  通用 runtime 的限定。

**等级**：中。系统边界充分，算法数学和数据分布桥接不足。

## 跨章一致性检查

- 第 1 章的系统分层必须与 `docs/TERM_GLOSSARY.md`、第 2/4 章的
  `autodiff tape → Burn IR/Fusion → CubeCL → Runtime` 用词一致。
- `Device` 表示运行时设备选择；`Backend` 只在实现契约语境中使用；
  不把 Tensor shape/dtype 的运行时属性写成 Rust 编译期保证。
- `ModuleRecord`、`TransitionBuffer`、`DataLoader`、`MultiDevice`、
  `DDP`、Remote 和 `burn-onnx` 的能力声明必须能回指固定源码或实验。
- 每章入口、八个小节导航、练习、来源映射和 `SUMMARY.md` 必须互相存在；
  正文 Rust 代码只能通过 `examples/` include。
- 所有 GPU/网络/分布式/量化性能结论必须标明硬件、backend、dtype、
  shape、同步边界和测量方法，或者明确写成框架无关设计/未来工作。

## 回补后验收

1. 每章至少有一条“原作概念—固定源码—可观察实验/明确边界”链；
2. 新增的公式、时间线和系统图不复制原作图片或框架专用代码；
3. 受影响示例的测试与 Clippy 通过；
4. `mdbook build book`、`make check`、`make check-local-sources`、
   `git diff --check` 和 IDE lint 通过；
5. `planning/STATUS.md`、`planning/DECISIONS.md` 和 session log 记录
   实际完成项、未完成项、验证命令和第 9 章下一动作。
