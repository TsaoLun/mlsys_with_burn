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
  可运行观察或未覆盖边界；保留 Flex CPU 基础路径，不把 GPU、网络、
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
  来源入口和九章证据状态；未覆盖能力必须标为协议/成本模型、可选平台
  实验或未覆盖（标签口径以 D018 与 `docs/TERM_GLOSSARY.md` 为准）。

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

## D017：P1 以同一初始状态验收，比较卡只作横向证据摘要

- 日期：2026-08-02
- 决策：P1 的 `initial_loss` 必须来自将要训练的同一个模型、且在第一次
  SGD 更新前计算；loader 验收必须检查 train/validation 的精确 ID 集合、
  batch 数和完整 shape，而不能只检查样本总数与第二维。P1 作为
  `Dataset → autodiff → ModuleRecord → inference` 的纵向学习路径保留；
  `comparison-cards.md` 作为 OpenMLSys/Burn/CPU/协议证据的横向摘要保留，
  逐文件映射仍以 crosswalk 为唯一真相。
- 原因：两个独立随机初始化的模型之间比较 loss 不能证明训练使同一状态
  变好；仅检查总数和 shape 第二维不能捕获重复 ID、错误 split 或 batch
  首维错误。P1 与比较卡解决的是不同学习和审计问题，删除任一项都会损失
  一种可复核性。
- 影响：固定协议和示例测试保持 CPU-first；比较卡显式标注纯 Rust 协议
  helper、教学用非密码学 checksum 和 CPU 模拟器，不把它们写成 Burn
  collective、生产安全或真实集群能力。

## D018：读者面向修订——导航顺序、标题体例与统一证据标签

- 日期：2026-08-08
- 决策：综合实验与比较卡从卷首移到书末的“贯穿实验与对照”部分；
  第 7–9 章小节标题去掉 `7.x` 等编号前缀，与第 1–6 章一致；综合实验页
  文件名从 `capstone-p1.md` 改为 `capstone.md`；新增书内页面
  `crosswalk-guide.md` 作为逐文件对照矩阵的读者入口（GitHub 链接 +
  标签定义），正文不再裸引用 `planning/` 路径；全书证据标签统一为
  `CPU 可运行验证`、`源码核验`、`协议/成本模型`、`可选平台实验`、
  `未覆盖` 五类，章节着陆页曾保留字面量小节名“证据状态”（已被 D020
  改为学习者用语）。
- 原因：读者按导航顺序应先读九章再进入贯穿实验；两套标题体例和两套
  标签措辞会让读者误以为内容分属不同版本；对照矩阵是本书的核心卖点，
  在线读者必须可达。`check_release.py` 限制 mdBook include 只能来自
  `examples/`，故对照矩阵以导读页 + 仓库链接发布，而不是内嵌复制，
  避免双份真相漂移。
- 影响：Pages 站点导航变化；后续新增章节沿用无编号小节标题和五类
  标签；`docs/TERM_GLOSSARY.md` 与 `docs/AUTHORING.md` 的证据分类列
  必须与正文同口径；若将来放宽 include 白名单，需要先评估对照矩阵
  内嵌的审计语言是否适合正文。

## D019：教材公式保留 `$`/`$$`，由主题配置启用 MathJax 美元分隔符

- 日期：2026-08-08
- 决策：正文继续使用 `$...$`（行内）与 `$$...$$`（独立公式）；不把全书
  改写成 mdBook 文档默认的 `\\(...\\)` / `\\[...\\]`。通过
  `book/theme/head.hbs` 在 MathJax 2 脚本之前注入 `tex2jax` 配置，显式
  启用 `$`/`$$`，并与官方括号分隔符并存；`book.toml` 设置
  `theme = "theme"`。行内公式禁止跨行。
- 原因：mdBook 0.4.51 开启 `mathjax-support` 后只注入默认
  `TeX-AMS-MML_HTMLorMML`，该配置不识别 `$...$`。线上页因此把
  `$3\times2=6$` 等公式原样显示。全书已按 `$`/`$$` 与 `\_` 下标约定
  写就；改为双反斜杠括号会扩大改动面，并与常见 Markdown 预览不一致。
- 影响：发布审计需检查生成 HTML 含 MathJax 脚本与 `$` 的 `inlineMath`
  配置；浏览器仍依赖 MathJax CDN（D015）。升级 mdBook/MathJax 主版本时
  必须复核主题片段是否仍在脚本之前生效。

## D020：正文只面向学习者——章首与练习去审计腔

- 日期：2026-08-09
- 决策：章节着陆页与综合实验页的小节名由「证据状态」改为「本章你能
  验证什么」；五类证据标签保留，但引导句改为学习者口吻。九章练习前言
  不再出现「可选平台实验 / 默认 CPU CI」审计句，改为说明【挑战】题需要
  额外环境或自行设计。`tools/check_release.py` 校验新小节名。实验节用
  「你会观察到 / 本实验刻意不做」替代「测试断言 / 验收」；`pins.toml`、
  决策编号与 `planning/` 路径不进入学习者主路径。
- 原因：可复核性仍必要，但审计元数据不应占据章首与练习的第一印象；
  前言已证明教材语域可以干净，问题来自模板化审计块而非主题本身。
- 影响：AUTHORING、术语表、对照导读与 STATUS 与新口径对齐；比较卡与
  对照矩阵仍可使用五类标签，但章首链接用语改为「横向主题比较」。
  章首五标签墙已被 D021 整包移入附录。

## D021：项目自洽材料整包后移附录

- 日期：2026-08-09
- 决策：删除九章着陆页与综合实验的「本章你能验证什么」五标签块；删除
  `ch01/08-comparison-and-sources.md`、卷中 `crosswalk-guide.md` 与
  `comparison-cards.md`。新建附录 `appendix-scope-and-evidence.md`
  （固定版本、标签定义、九章范围一览、比较卡、对照导读、C/S/R/L/E）与
  `appendix-sources.md`（九章来源与改编长文）。章末练习只留一句指针。
  SUMMARY「贯穿实验与对照」改为「贯穿实验」且只含综合实验；第 1 章为
  7 小节。`check_release.py` 校验附录存在，不再要求章首五标签。
- 原因：证据层级对项目可复核重要，对学习者不是章首刚需；OpenMLSys v1
  亦无此体例。主路径应先讲系统，自洽账本后置。
- 影响：AUTHORING / TERM_GLOSSARY 与 D020 口径以本决策为准；lab「刻意
  不做」与 `running-examples.md` 仍留主路径。

## D022：可选跑通 Profile 不得进入默认门禁

- 日期：2026-08-09
- 决策：GPU/WGPU/ONNX/CUDA/collective 等**可选真机或独立对照**跑通，
  统一登记在 `docs/OPTIONAL_PROFILES.md`，并由
  `book/src/running-examples.md` 指向。主线正文已同步 GPU / Runtime /
  通信叙事（M6）；可选 profile 只服务「有环境者跑起来」。默认
  `make check`、各章默认 `cargo test` 与 CI **不得**依赖专有 GPU、
  NCCL、真实集群或根 workspace 外的 ONNX 端到端（D010 仍有效）。
- 原因：把「讲 GPU」与「默认必须跑 GPU」拆开，避免无驱动读者被挡在
  第 3–4 章外，也避免可选失败污染发布 gate。
- 影响：新增可选 profile 时先更新 OPTIONAL_PROFILES 与本节约束；不得
  为通过真机测试而放宽默认 feature 或加入 `[patch]` 本地镜像。

## D023：教科书化第一批——文献出口、提示专属、边界语域、结构图

- 日期：2026-08-13
- 决策：为改善读者体验做四项体例决定。（1）新增
  `book/src/references.md` 全书参考文献页：按章分组、每条一句导读、
  只用有把握的 arXiv/DOI/官方链接；各章延伸阅读一行指针链接过去；
  文献只作原理与产业背景延伸，不作为固定版本能力证据。（2）练习
  折叠提示必须题目专属：至少含指向具体小节的链接、示例观察点或
  源码路径，并加一句实质性方向；禁止零信息量套话与章内复用。
  （3）正文边界表述收敛：同一能力边界在一个小节内只声明一次；连续
  否定枚举改写为「系统需要 / 已提供 / 应用要补」分工表；「固定 X」
  修饰语在文件首次交代后省略，需要时用「本书固定版本」「当前实现」。
  （4）结构关系图优先自制 SVG，`text` 围栏保留给代码轨迹/消息序列；
  本批新增 ch02 工作流、ch04 生命周期同步对比、ch05 worker 通路、
  ch06 DDP 分层、ch08 Actor–Learner 五张图。
- 原因：读者视角对照原作发现四个最伤教材体验的问题——延伸阅读是
  「断头路」（全书零论文出处）、提示套话化（同句复用最多 13 次/章）、
  每千字 3 次「边界」的审计腔、图文比远低于原作（19 对 232 处引用）。
  这些都能在不放松证据纪律的前提下修复：边界事实保留，表达方式收敛。
- 影响：AUTHORING 已加入对应体例；`check_release.py` 的链接检查
  覆盖 references.md 的内部锚点（外链不校验，失效时按题名检索）；
  后续新增章节内容按本决策执行。原有「不能外推」的章末系统结论
  条目保留，不受本决策影响。

## D024：LLM 服务机制以协议模型进入第 7 章

- 日期：2026-08-13
- 决策：新增 `examples/ch07-serving-queue-sim`（纯 Rust 虚拟时间
  协议模型），把连续批处理（continuous batching）与 KV cache 容量
  预算两个机制纳入第 7 章的可运行实验：静态批 vs 连续批的延迟/
  吞吐/空转对比、KV 预算对并发与吞吐的单调约束。第 1、7 章的
  LLM 主题声明从「机制完全未展开」改为「机制有框架无关协议模型，
  Burn 服务 runtime 与真实 KV 管理仍未覆盖」。模型刻意简化：
  prefill 一步完成（无 chunked prefill）、KV 按 prompt+decode
  预留（无抢占/换出），简化必须写在正文。
- 原因：KV cache 与 continuous batching 是 2026 年读者最关心的
  服务主题，原声明只给「未覆盖」与文献出口；本书已有同类先例
  （第 9 章集群模拟器、第 5 章背压协议）证明「协议/成本模型」
  证据类可以承载这类机制而不越过能力边界。深度分析亦指出这是
  超越原作最自然的机会点。
- 影响：`ExecutionStrategy` 等 Burn 能力断言不变；附录第 7 章
  范围的「协议/成本模型」条目加入队列模型；LLM **专章**（训练、
  投机采样、MoE 等）仍留给后续版本，本决策不改变该规划。
