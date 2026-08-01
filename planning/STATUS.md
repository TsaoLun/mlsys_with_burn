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
- [x] 建立 `planning/backfill/ch01-ch08-audit.md`，逐章对照固定 OpenMLSys
  v1、Burn/CubeCL/CubeK 证据、缺口等级、回补动作和能力边界。
- [x] 全面回补第 1–2 章的负载卡片、吞吐/内存预算、完整 ML workflow、
  Rust/CubeCL 扩展边界、Module visitor 和 Device/autodiff 观察。
- [x] 全面回补第 3–4 章的 Roofline/算术强度、GEMM 优化不变量、Pass
  契约、Fusion→Strategy→JIT/cache→launch/read 因果链；扩展
  `ch03-tile-loads` 的 intensity 模型。
- [x] 全面回补第 5–6 章的队列背压、文件索引、重试/epoch 提交、流水线
  micro-batch bubble、并行内存动机和参数服务器版本协议。
- [x] 全面回补第 7–8 章的 PTQ 校准、稀疏收益条件、推理 worker/layout、
  artifact 威胁模型、MC/TD、探索策略版本、Actor–Learner freshness 和
  MARL credit assignment。
- [x] 更新 `docs/TERM_GLOSSARY.md`、D012 和第 1–8 章来源映射，统一
  workload/算术强度、done/truncated、behavior/target policy 等术语。
- [x] 本次回补验证：受影响示例测试与 Clippy、`cargo run` 观察输出、
  `mdbook build book`、`make check`、`make check-local-sources` 和
  `git diff --check` 均通过。
- [x] 统一修复第 1–8 章 Markdown 数学公式的下标转义，并处理独立公式
  续行的 `+` 列表解析；重新构建后复查 86 个 display 公式、244 个行内
  公式候选，未发现 `<em>`/`<ul>`/`<ol>` 破坏，含公式页面均加载 MathJax。

## 本次交接

- 已完成：第 1–8 章全面回补与复核。新增审计矩阵，并把原作中较薄的
  workload、workflow、GEMM/Roofline、Pass/IR、数据背压、训练并行、
  PTQ/安全、RL 算法与 Actor–Learner/MARL 系统内容补回正文；每项均保留
  固定源码证据或明确未覆盖边界。
- 验证：第 2–8 章受影响示例单包 tests/Clippy 通过；`ch02-tensor-basics`
  输出普通/autodiff Device 标志，`ch03-tile-loads` 输出
  naive/tiled load 与 intensity，`ch07-record-roundtrip` 与
  `ch08-rl-rollout` 输出协议观察；`mdbook build book`、`make check`、
  `make check-local-sources`、`git diff --check` 和 IDE lint 均通过。
- 公式复核：16 个正文文件统一使用 Markdown 数学下标转义；`make book`
  成功，源码中无未转义数学下标和 display 列表标记，生成 HTML 的 86 个
  display 公式与 244 个行内公式候选均未出现 Markdown 结构污染。
- 偏差：没有新增真实 GPU 共享内存 GEMM、服务压测、PTQ/QAT runtime、
  网络 Actor–Learner、MARL league 或 pipeline/parameter-server runtime；
  这些内容依旧以框架无关模型、固定源码边界和练习表达，符合 D009–D012。
- 下一步：审查 OpenMLSys v1 分布式训练 cluster/system 文件，核验
  Burn/CubeCL GPU、collective、通信、调度与遥测边界，开始第 9 章
  “大规模 GPU 集群管理”的来源映射。

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
