# 实时状态

更新日期：2026-08-01

## 当前里程碑

M4：应用与扩展篇。

## 当前目标

建立第 9 章“大规模 GPU 集群管理”的 OpenMLSys 与 Burn/CubeCL
集群运行时来源映射。

## 进行中

- [ ] 映射 OpenMLSys v1 分布式训练中的集群/系统内容，核验 Burn 固定快照
  中 GPU、通信、调度、遥测与容错边界。

## 下一步

1. 逐文件审查 OpenMLSys v1 分布式训练的 cluster/system 相关文件。
2. 核验固定 Burn/CubeCL/CubeK 的 GPU、collective、通信和运行时入口，
   区分已有 backend 能力与外部调度器职责。
3. 设计 CPU 可测试的调度/通信成本或故障模型实验，区分集群控制面、
   工作负载运行时和设备 backend。

## 已完成

- [x] 确定项目名为 “MLSys with Burn”。
- [x] 确定正文采用 CC BY-NC-SA 4.0，原创代码采用 MIT OR Apache-2.0。
- [x] 确定五个上游仓库保持并列、只读，并由 `pins.toml` 记录快照。
- [x] 确定以 Burn 0.22.0-pre.1 版本线展开。
- [x] 完成根 Git、许可证、Agent 规则和实时计划文档。
- [x] 完成九章 mdBook 骨架和第 2 章 CPU 张量示例。
- [x] 完成上游 pin 校验工具、Makefile 和 GitHub Actions CI。
- [x] Cargo 统一使用 GitHub 固定 revision；本地上游镜像仅供 Agent 阅读。
- [x] 补齐 CC BY-NC-SA 4.0、MIT 和 Apache-2.0 完整许可证文本。
- [x] 完成第 1 章七节正文、来源映射、练习和 Flex 执行栈实验。
- [x] 完成第 2 章八节正文、逐文件来源映射和 Burn 0.22 API 核验。
- [x] 扩展第 2 章实验，覆盖广播、Module 参数统计和 Flex 自动微分。
- [x] 完成第 3 章八节正文、逐文件来源映射和 CubeCL/CubeK 源码核验。
- [x] 实现 CubeCL CPU scale Kernel，覆盖拓扑、raw buffer 与 unsafe 边界。
- [x] 在 CubeCL CPU 和 WGPU Runtime 上验证同一 Kernel 与 host reference。
- [x] 完成 M2 基础篇，形成 Burn Tensor API 到 CubeCL Kernel 的学习闭环。
- [x] 完成第 4 章八节正文、逐文件来源映射和 Burn/CubeCL 源码核验。
- [x] 实现 FusionInspector CPU 实验，验证 add→exp 融合与同步切分。
- [x] 区分 autodiff tape、Burn Fusion IR、CubeCL IR 和设备 graph capture。
- [x] 建立根 Git 基线提交 `e1769a5`。
- [x] 完成第 1–4 章补全：术语表、计算图/Pass/内存加厚、tile 加载模型、
  分支 autodiff 与三操作 Fusion 扩展；计划文档已删除。
- [x] 本机验证：`ch01`/`ch02`/`ch03-tile-loads` 测试与 Clippy、mdBook、
  pin 检查、`cargo fmt --all --check`。
- [x] 完成第 5 章八节正文、`SUMMARY.md` 导航和
  `planning/chapter-sources/ch05.md` 逐文件来源映射。
- [x] 核验固定 Burn `Dataset`、惰性 transform、`Batcher`、DataLoader、
  shuffle、采样、分片、SQLite 和多 worker 错误/顺序边界。
- [x] 新增 `examples/ch05-data-pipeline`，测试 map、batching、固定 seed、
  epoch RNG、multi-worker 数据守恒、Device 传递和参数错误。
- [x] 将 `burn` 的 `dataset` feature 接入固定 Git revision，更新
  `Cargo.lock`，未使用本地 path 或 `[patch]`。
- [x] 本机运行 `make check` 与 `make check-local-sources` 均通过。
- [x] 完成第 6 章八节正文、`SUMMARY.md` 导航和
  `planning/chapter-sources/ch06.md` 逐文件来源映射。
- [x] 核验固定 `burn-train` 的 `TrainStep`、`Learner`、optimizer、
  scheduler、checkpoint、本机 `MultiDevice` 和 DDP 策略。
- [x] 核验 `DistributedContext`、autodiff gradient registration、
  backend `all_reduce`/`sync_collective`，并确认 Flex CPU 没有
  collective 实现。
- [x] 新增 `examples/ch06-training-loop`，测试 CPU autodiff、SGD loss
  下降、参数变化和训练参数错误。
- [x] 增加 D009，明确 CPU 单设备实验与 DDP/跨节点能力边界。
- [x] 本机运行第 6 章示例检查、`make check` 与
  `make check-local-sources` 均通过。
- [x] 完成第 7 章八节正文、`SUMMARY.md` 导航和
  `planning/chapter-sources/ch07.md` 逐文件来源映射。
- [x] 核验固定 `burn-onnx` 的 ONNX→BurnGraph→Rust codegen→Burnpack
  路径、四种 `LoadStrategy`，以及主线 `ModuleRecord`、burn-store、
  Remote 和 WASM/no_std 边界。
- [x] 新增 `examples/ch07-record-roundtrip`，测试 CPU Linear 参数
  Burnpack 内存 round-trip、输出 shape 和数值误差。
- [x] 增加 D010，隔离 `burn-onnx` 旧 Burn revision 与当前主线实验。
- [x] 本机运行第 7 章示例、`make check` 与
  `make check-local-sources` 均通过。
- [x] 完成第 8 章八节正文、`SUMMARY.md` 导航和
  `planning/chapter-sources/ch08.md` 逐文件来源映射。
- [x] 核验固定 `burn-rl` 的 Environment、Policy、Batchable、
  TransitionBuffer、AsyncPolicy，以及 `burn-train` 的多环境 rollout、
  off-policy、evaluation 和 checkpoint 边界。
- [x] 新增 `examples/ch08-rl-rollout`，测试确定性环境的 done/truncated、
  circular replay、随机 batch shape 和表格 TD 更新。
- [x] 增加 D011，隔离 `burn-rl` 组合抽象与完整 DQN/MARL 算法实验。
- [x] 本机运行第 8 章示例、`make check` 与
  `make check-local-sources` 均通过。

## 本次交接

- 已完成：第 8 章从 OpenMLSys v1 的 MDP、单节点采样和多智能体系统
  重组为 environment→policy→trajectory/replay→learner→系统边界路线；
  覆盖固定 `burn-rl` traits、`TransitionBuffer`、`AsyncPolicy` 和
  `burn-train` off-policy 编排。
- 验证：`cargo test -p ch08-rl-rollout`（4 tests passed）、
  `cargo clippy -p ch08-rl-rollout --all-targets -- -D warnings`、
  `cargo run -p ch08-rl-rollout`、`mdbook build book`、`make check`、
  `make check-local-sources`、`git diff --check` 均通过；`make check`
  的 workspace lint 与测试全部通过。
- 偏差：没有将 gym-rs/SDL2、完整 DQN、GPU simulator、网络 Actor–Learner
  或 MARL league 加入根 workspace；原因是它们不是 `burn-rl` 的内置能力，
  且固定 DQN example 的 native simulator 依赖不适合作为默认 CPU CI。
  D011 已记录。
- 下一步：审查 OpenMLSys v1 分布式训练 cluster/system 内容，开始第 9 章
  “大规模 GPU 集群管理”的 Burn/CubeCL 能力映射。

## 已知问题

- `burn-onnx` 当前仓库版本为 0.22.0-pre.1，但其 manifest 仍 pin 到较早
  的 Burn commit；ONNX 章节必须按该关系单独验证，不能假定与本地 Burn
  HEAD 可互换。
- Burn 的分布式文档仍在演进，第 6、9 章不能只依赖 Burn Book。
- `burn-rl` 当前固定快照提供环境、policy、replay 和 runner 组合抽象，
  不提供通用 DQN/PPO/SAC、prioritized replay 或 MARL/Actor–Learner
  集群协议；第 8 章 D011 和来源映射已标出这些边界。
- `tracel-llvm v22.1.4-5` 的 bundler 资产在不同平台/缓存环境可能影响
  CubeCL CPU 路径；本次 Intel macOS 工作区的完整 `make check` 已通过，
  干净环境仍应以 CI 结果为准。

## 交接模板

完成一次工作后更新本文件：

- 已完成：具体文件与内容。
- 验证：实际运行的命令和结果。
- 偏差：与计划不同之处及原因。
- 下一步：一个可以直接执行的动作。
