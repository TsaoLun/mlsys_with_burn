# 决策记录

## D001：项目名称

- 日期：2026-07-30
- 决策：项目名使用 “MLSys with Burn”，仓库建议名为
  `mlsys-with-burn`。
- 原因：准确表达 Burn 主线，同时避免 `openmlsys-*` 带来的官方归属误解。

## D002：上游管理

- 日期：2026-07-30
- 决策：五个上游仓库保持根目录下的独立并列 clone，由 `.gitignore`
  排除，并由 `pins.toml` 记录快照。
- 原因：避免将多个大型 workspace 合并，也不在基础设施阶段移动现有
  工作区。教材代码通过固定 git revision 保证公开构建可复现。

## D003：版本基线

- 日期：2026-07-30
- 决策：首个写作周期以 Burn 0.22.0-pre.1 为基线，并记录其实际
  CubeCL、CubeK 和 burn-onnx 关系。
- 原因：这些项目仍处预发布开发期，教材需要稳定快照而不是跟随 `main`。

## D004：许可边界

- 日期：2026-07-30
- 决策：衍生教材及项目文档采用 CC BY-NC-SA 4.0；原创示例和工具采用
  MIT OR Apache-2.0。
- 原因：满足 OpenMLSys 的 BY、NC、SA 要求，同时让原创代码保持常见的
  Rust 生态许可证。

## D005：内容架构

- 日期：2026-07-30
- 决策：采用 OpenMLSys v2 的九章结构，以 v1 为素材来源；先完成第 2 章
  纵向切片，再按基础篇、系统篇、应用篇推进。
- 原因：v2 正文尚未完成，而第 2 章最能验证“原理—Burn API—底层源码—
  可运行实验”的完整工作流。

## D006：远程依赖与本地源码镜像

- 日期：2026-07-30
- 决策：Cargo 构建只使用 `pins.toml` 对应的 GitHub revision。Burn
  `0.22.0-pre.1` 的 commit 决定其 CubeCL 与 CubeK revision；禁止使用
  本地 `path` 依赖或 `[patch]` 覆盖。
- 原因：公开构建和 CI 不应依赖特定目录布局。根目录下可选的 Burn、
  CubeCL、CubeK、burn-onnx 和 OpenMLSys clone 被 Git 忽略，仅供 Agent
  快速搜索和源码阅读。

## D007：先回补第 1–4 章再写系统篇后续章

- 日期：2026-07-31
- 决策：在继续第 5 章之前，完成术语对齐、计算图/Pass/内存原理加厚、
  加速器最小可执行阶梯与教学图补强；补全文风以本书既有第 1–4 章为准，
  不以 OpenMLSys 教程腔为模板。跨章用语以 `docs/TERM_GLOSSARY.md` 为准。
- 原因：现有四章已形成可核验 Burn 纵向路径，但相对 OpenMLSys 在原理
  厚度、图示和动手阶梯上偏薄；先加固基础再写数据处理与训练，避免术语
  漂移和读者断层放大。
- 状态：已执行；交接见
  `planning/session-logs/2026-07-31-backfill-ch01-ch04.md`。临时计划文件
  已删除。

## D008：第 5 章以数据守恒与顺序边界分开验证

- 日期：2026-08-01
- 决策：第 5 章实验将 Dataset map、batching、seed 和多 worker 作为
  独立可观察量；多 worker 默认只断言样本守恒、变换值和 progress，不把
  batch 到达顺序写成 Burn 的保序能力。
- 原因：固定 Burn `MultiThreadDataLoader` 通过分片 worker 和 bounded
  message channel 返回 batch，消息没有全局样本序号或消费者侧重排；这与
  OpenMLSys v1 用 MindSpore Connector 实现的保序语义不同。把两者混写会
  将来源系统的能力错误外推到 Burn。
- 影响：需要稳定全局顺序的示例使用 `num_workers = 0`，或另行实现带序号
  的 reorder layer；第 6 章讨论多设备训练时的 sampler/checkpoint 协议。

## D009：第 6 章以 CPU 单设备训练验证循环，隔离 DDP 后端边界

- 日期：2026-08-01
- 决策：第 6 章实验使用 `Device::flex().autodiff()` 和
  `burn-optim` 手写训练循环，验证 forward → backward →
  optimizer step → loss 进展；不把 CPU Flex 作为 DDP 或 AllReduce
  的运行验证路径。
- 原因：固定 Burn 源码中的 `burn-train` DDP API 和 `DistributedContext`
  已存在，但 `burn-flex/src/ops/transaction.rs` 明确使用不支持 collective
  operations 的默认实现。可运行的 DDP 还需要实现 collective 的后端、匹配
  的设备集合以及每个节点一致的启动配置；仅凭 CPU API 编译不能证明跨设备/
  跨节点通信成立。
- 影响：正文分别描述 `MultiDevice` 本机策略、DDP 的 verified API 边界和
  参数服务器/流水线并行的未覆盖范围；后续若增加 CUDA/NCCL 或远程运行实验，
  必须单独记录设备、进程、通信库和验证结果。

## D010：第 7 章隔离 burn-onnx 的 Burn revision，实验使用主线 ModuleRecord

- 日期：2026-08-01
- 决策：第 7 章按固定 `burn-onnx` 源码核验 ONNX→Rust→Burnpack
  的生成和加载策略，但不把 `burn-onnx` 加入当前根 workspace 的同一
  Burn 依赖图；CPU 实验使用主线 Burn 的 `ModuleRecord` 内存
  round-trip，验证参数 artifact 加载后推理输出保持一致。
- 原因：`pins.toml` 中 `burn-onnx` 为 `af2dfb...`，其 manifest 明确把
  `burn`、`burn-flex`、`burn-store` 指向 `78f10aec...`，而根项目当前
  Burn pin 为 `976aa9c...`。相同 crate 名称的不同 revision 不能证明
  `Tensor`、`Module` 和 store 类型兼容；强行混用会使实验依赖两个不一致
  的 Burn API/类型世界。
- 影响：正文把 ONNX importer 的源码核验与当前主线 Record 实验分开；
  本章不声称根 workspace 已完成 ONNX 导入或 Remote/WASM 端到端部署。
  将来更新 pin 时，必须先对齐 burn-onnx 的 Burn revision，再增加真正的
  ONNX fixture 与目标平台验证。

## D011：第 8 章以确定性环境隔离 burn-rl 抽象与具体 RL 算法

- 日期：2026-08-01
- 决策：第 8 章基础实验使用自定义确定性 `Environment`、主线 Burn
  `TransitionBuffer` 和表格 TD/Q-learning 更新，验证 rollout、终止边界、
  replay sampling 与 bootstrapping；不把实验扩展为完整 DQN、PPO、SAC、
  多智能体或分布式 actor/learner 系统。
- 原因：固定 `burn-rl` 提供的是环境、policy、batching、transition
  buffer 和 device conversion traits，具体 learner/loss/optimizer 由用户
  实现。固定 `burn-train` 的 `OffPolicyStrategy` 可以编排多环境、异步
  inference、replay、`PolicyLearner` 和 checkpoint，但完整 DQN example
  还依赖 native gym-rs/SDL2，并自行实现 TD target、target network、
  optimizer 和自定义 record。将这些工程依赖放进基础 CPU 实验会把
  simulator/算法/训练编排混成一个不可隔离的验证。
- 影响：正文把 Burn 已提供的 RL 组合抽象与算法实现边界分开；实验可以在
  无外部 simulator、无 GPU 和无网络的环境运行。后续若加入 DQN 或多环境
  benchmark，必须单独记录 simulator、随机种子、policy update、设备、
  checkpoint 和吞吐测量。

## D012：第 1–8 章回补以原理—源码—观察闭环为验收单位

- 日期：2026-08-01
- 决策：在开始第 9 章前，对第 1–8 章执行一次全面回补。每章新增或修订
  的内容必须同时说明框架无关原理、固定 OpenMLSys/Burn/CubeCL 来源、
  可运行观察或明确未覆盖边界；保留 Flex CPU 基础路径，不把 GPU、网络、
  量化、完整 DQN/MARL 或集群性能写成已验证能力。
- 原因：原有章节已经有 API 和实验骨架，但与 OpenMLSys v1 对照时，部分
  系统设计被压缩为名词列表，容易让读者把“有 trait/入口”误读为“有完整
  runtime”。GEMM roofline、数据背压、流水线 bubble、参数服务器版本、
  PTQ 校准和 RL return/TD 等内容需要补原理，而不是继续增加未经验证的
  框架代码。
- 影响：新增 `planning/backfill/ch01-ch08-audit.md` 作为逐章缺口矩阵，
  更新各章正文、来源映射和术语边界；实验只增加可在当前 workspace
  观察的指标（如 tile load/intensity、Device autodiff 标志），复杂能力
  继续以源码证据、成本模型或练习表达。完成后第 9 章从集群控制面、
  GPU/通信运行时和故障边界继续，不重新打开已隔离的上游依赖问题。

## D013：第 9 章以 CPU 集群模拟器隔离控制面与真实 GPU 集群

- 日期：2026-08-01
- 决策：第 9 章默认实验使用纯 Rust、确定性虚拟时间模拟器，覆盖
  FIFO/topology-aware placement、gang admission、`alpha + beta * bytes`
  通信成本、checkpoint replay、失败重试和资源归还；不依赖 Burn、
  CUDA、NCCL、网络或真实 GPU。
- 原因：固定 Burn/CubeCL 快照可以核验本机/设备的 DDP、collective、
  ComputeClient、stream 和 memory 入口，但没有集群作业队列、租户配额、
  拓扑放置、rank rendezvous、elastic membership、故障 detector 或集群级
  telemetry。把这些控制面能力写成 Burn 已实现，会把 API 入口误读为完整
  集群 runtime。
- 影响：正文把模拟结果限定为协议和成本模型观察；真实 CUDA/NCCL 或跨节点
  benchmark 需要另行记录硬件、driver、通信库、launcher、拓扑和故障
  环境，不能由 CPU 模拟器或源码存在外推。

## D014：以核心目录 crosswalk 和 C/S/R/L/E 作为 OpenMLSys 比较口径

- 日期：2026-08-01
- 决策：本书与 OpenMLSys v1 的比较以
  `planning/comparison/openmlsys-v1-crosswalk.md` 为逐文件基线。每个
  核心主题都记录 Correctness、Source、Runnable、Learning path 和
  Engineering 五类证据，并单独列出推荐系统、联邦学习、可解释 AI、
  机器人和附录等范围差异。
- 原因：逐字翻译或按章节标题一一对应会隐藏本书对 Burn/Rust 的重组，也
  会把源码入口误写成平台 parity。crosswalk 允许一对多/多对一映射，并
  让读者回答“原作讲什么、固定 Burn 验证到哪层、差异为何存在”。
- 影响：发布审计检查 crosswalk 覆盖固定 OpenMLSys Markdown、revision、
  来源入口和九章证据状态；未覆盖能力必须标为协议模型、可选平台实验或
  明确未覆盖。

## D015：固定快照版采用锁定构建、离线 Cargo gate 和在线 MathJax 边界

- 日期：2026-08-01
- 决策：发布基线使用 `pins.toml` 的完整 Git revision、`Cargo.lock`、
  `cargo --locked` 和 Cargo offline gate；Rust/mdBook/Python 版本写入
  `release.toml`，CI action 固定到完整 commit SHA。mdBook 公式继续由
  MathJax 渲染，并明确“Cargo 可离线构建”不等于浏览器 CDN 可离线阅读。
- 原因：预发布 Burn/CubeCL/CubeK 仍可能变化，未锁定依赖和可变 CI action
  会让同一章节无法复核；MathJax 是阅读产物边界，不能伪装为 Cargo 依赖。
- 影响：`tools/check_release.py` 检查 SUMMARY/include/source/license/link/
  formula、生成 HTML 和 offline metadata；默认 CI 不需要本地上游镜像，
  `--check-local-sources` 只作为额外源码路径审计。

## D016：候选版静态书站托管在 GitHub Pages

- 日期：2026-08-02
- 决策：九章候选版的可读 HTML 由 GitHub Pages 托管，URL 为
  `https://tsaolun.github.io/mlsys_with_burn/`。构建产物来自固定
  `mdbook 0.4.51`，由独立 workflow
  `.github/workflows/deploy-pages.yml` 在 `main` 推送或手动触发时发布；
  不提交 `book/book/`，不把完整 Rust/Cargo 测试绑进 deploy job，也不改
  默认 CPU gate。不采用 Deno Deploy。
- 原因：教材已是 mdBook 静态站点；GitHub Pages 与现有 pinned CI 工具链
  对齐，且适合 project-site 子路径。Deno 更适合 edge/API，对纯静态产物
  无额外验收收益。
- 影响：`book/book.toml` 设置 `site-url = "/mlsys_with_burn/"`；deploy
  workflow 写入 `.nojekyll` 并上传 Pages artifact；仓库需在 Settings →
  Pages 选择 Source = GitHub Actions。MathJax CDN 边界仍按 D015。

