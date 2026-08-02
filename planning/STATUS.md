# 实时状态

更新日期：2026-08-02

## 当前里程碑

M5：首个稳定版审计准备。

## 当前目标

完成以 OpenMLSys v1 固定 revision 为参照的 P0/P1 对照发布审计，并保持
九章候选版的 CPU-first、源码证据和可选平台边界可复核；静态书站经
GitHub Pages 可读。

## 进行中

- [ ] P0/P1 已完成；等待发布者决定是否创建候选 tag/发布归档。
- [ ] 仓库 Settings → Pages → Source 需设为 GitHub Actions，并在推送
  `main` 后确认 `https://tsaolun.github.io/mlsys_with_burn/` 可访问。

## 下一步

1. 在 GitHub 仓库启用 Pages（Source = GitHub Actions），推送或手动运行
   `Deploy Pages` workflow，确认站点根路径与含公式章节可打开。
2. 由发布者审阅 `planning/comparison/openmlsys-v1-crosswalk.md` 和
   `tools/check_release.py` 的机器可读输出，决定候选版归档/tag。
3. 若要增加真实 GPU、NCCL、ONNX、DDP、DQN/MARL 或网络实验，建立独立
   平台 profile，不能改变默认 CPU gate。
4. 继续跟踪 Burn 预发布快照；更新 pins 前先新增决策记录并重跑全书审计。

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
- [x] 完成第 9 章八节正文、`SUMMARY.md` 导航和
  `planning/chapter-sources/ch09.md` 来源映射，覆盖集群负载、GPU/rack/
  ToR/Spine 拓扑、队列、gang scheduling、拓扑放置、通信、多租户、故障、
  checkpoint 和遥测边界。
- [x] 新增 `examples/ch09-cluster-simulator`，使用纯 Rust 虚拟时间验证
  FIFO/topology-aware placement、gang admission、`alpha + beta * bytes`
  通信成本、checkpoint replay、失败重试、资源归还和确定性 trace。
- [x] 增加 D013，明确第 9 章 CPU 控制面模拟与真实 GPU/NCCL/跨节点集群
  能力隔离；更新集群术语、来源记录和会话日志。
- [x] 第 9 章验证：示例 6 项测试、Clippy、运行观察、`mdbook build book`、
  `make check`、`make check-local-sources`、`git diff --check` 均通过；
  全书数学静态复查无未转义下标和 Markdown 结构污染。
- [x] 建立 `planning/comparison/openmlsys-v1-crosswalk.md`，覆盖 OpenMLSys
  v1 固定章节 Markdown、扩展篇排除清单、Burn/CubeCL/CubeK/burn-onnx
  源码入口和 C/S/R/L/E 五类证据；更新 `CHAPTER_MATRIX` 与九份来源映射。
- [x] 新增 `tools/check_release.py`，自动检查 SUMMARY/八小节、include/
  anchor、source crosswalk、pins/Cargo.lock、许可证、链接、公式、生成
  HTML MathJax、代码片段 annotation、Git hygiene 和 offline metadata。
- [x] 更新 Makefile/CI 的 `--locked`、offline Cargo gate、mdBook test、
  doctest、十个 CPU smoke、capstone smoke 和 release audit；新增
  `release.toml` 并固定 Actions commit SHA。
- [x] 更新中英文 README、书内 README/attribution、NOTICE、AUTHORING、
  glossary，明确九章候选版、工具版本、快照、burn-onnx revision、MathJax
  CDN 边界和非官方关系；增加 D014/D015。
- [x] 新增第 1 章第八小节、`book/src/capstone-p1.md`、`planning/capstone-p1.md`
  和 `examples/ch05-ch07-capstone`，通过确定性 20 样本完成
  Dataset→训练→ModuleRecord→恢复后 inference。
- [x] 第 2 章负向 detach/tape 实验和第 4 章重复 IR/Fusion/cache 观察通过
  测试、Clippy、CPU run；`BURN_FUSION_LOG=full` 观察到固定 runtime 的
  cache-hit 日志，但测试只断言计划/输出一致。
- [x] 新增 `book/src/comparison-cards.md`，并在第 5–9 章示例中加入
  shard/背压、collective/staleness、artifact contract、policy freshness、
  trace schema 等纯 Rust 协议测试和统一证据标签。
- [x] P0/P1 终验收通过：`make check`、`make check-local-sources`、
  workspace Clippy/test/doctest、mdBook build/test、release audit、离线
  metadata、`cargo fmt --all --check` 和 `git diff --check`。
- [x] 新增 GitHub Pages 部署：`.github/workflows/deploy-pages.yml`、
  `book.toml` 的 `site-url`、D016、`release.toml` pages 元数据和中英文
  README 在线阅读链接；不提交 `book/book/`，不改默认 CPU gate。

## 本次交接

- 已完成：为九章候选版增加 GitHub Pages 静态发布（D016）。独立 deploy
  workflow 使用固定 `mdbook 0.4.51` 与 pinned Pages actions；`book.toml`
  使用 project-site 路径 `/mlsys_with_burn/`；README / STATUS /
  `release.toml` 记录预期 URL。
- 验证：本地 `mdbook build book`，产物含 `index.html`；workflow 会写入
  `.nojekyll`。现有 CI 完整 Rust gate 保持不变。
- 偏差：线上可达性依赖仓库 Settings → Pages 选择 GitHub Actions，并在
  推送 `main` 后由 Actions 实际部署；本机无法代替该一次性配置。
- 下一步：启用 Pages source 并触发 `Deploy Pages`；随后再决定候选
  tag/归档。

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
