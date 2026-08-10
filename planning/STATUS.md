# 实时状态

更新日期：2026-08-10

## 当前里程碑

M5：首个稳定版审计准备（可与 M6 并行；M5 tag 不阻塞于 M6）。

## 当前目标

完成以 OpenMLSys v1 固定 revision 为参照的 P0/P1 对照发布审计，并保持
九章候选版的 CPU-first、源码证据和可选平台边界可复核；静态书站经
GitHub Pages 可读。读者正文按 D020/D021 只面向学习者；项目自洽材料在附录。
M6（深度、GPU 叙事与体感补强）正文与可选 profile 文档已落地（D022）。

## 进行中

- [ ] P0/P1 已完成；等待发布者决定是否创建候选 tag/发布归档。
- [ ] 仓库 Settings → Pages → Source 需设为 GitHub Actions，并在推送
  `main` 后确认 `https://tsaolun.github.io/mlsys_with_burn/` 可访问。
- [x] 修复线上 `$...$` 公式不渲染：自定义 theme 启用 MathJax 美元分隔符
  （D019）；全书 42 个含公式页面 Puppeteer 核验通过。
- [x] 学习者文风改写（D020）与自洽材料后移附录（D021）：章首五标签/
  对照页/来源长文已迁入 `appendix-scope-and-evidence.md` 与
  `appendix-sources.md`。
- [x] M6a/M6b/M6c：章末系统结论、设备/Runtime 地图、ch2–4/6/7/9 GPU
  与原理加厚、`docs/OPTIONAL_PROFILES.md` + D022。

## 下一步

1. 提交并推送 D020/D021/D022 与 M6 正文（若尚未上线），确认 Pages
   导航、附录与配图正常。
2. 由发布者审阅 `planning/comparison/openmlsys-v1-crosswalk.md` 和
   `tools/check_release.py` 的机器可读输出，决定候选版归档/tag。
3. 真机 CUDA/NCCL 仅在 pins 与环境允许时，向
   `docs/OPTIONAL_PROFILES.md` 追加命令；不得改默认 CPU gate（D022）。
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
- [x] 复审并收紧 P1 贯穿实验：`initial_loss` 与训练使用同一初始模型，
  loader 严格检查 train/validation ID 集合、batch 数和完整 shape；比较卡
  明确 crosswalk 是逐文件真相，并标出协议 helper、教学 checksum 和 CPU
  模拟器的证据边界；记录 D017。
- [x] 将比较卡标题统一为“第 N 章：主题”，并同步第 3–9 章正文中的
  Markdown 锚点链接；不改变 SUMMARY 导航或章节内容范围。
- [x] 清理读者可见的项目内部术语：P1/capstone 改为“综合实验”，
  crosswalk 改为“逐文件对照矩阵”，补充 C/S/R/L/E 的完整含义，解释
  CPU-first、smoke test、parity、release audit 和 offline gate，并将
  可选轨道、主线、协议卡等表达改为通用描述。
- [x] P0/P1 终验收通过：`make check`、`make check-local-sources`、
  workspace Clippy/test/doctest、mdBook build/test、release audit、离线
  metadata、`cargo fmt --all --check` 和 `git diff --check`。
- [x] 新增 GitHub Pages 部署：`.github/workflows/deploy-pages.yml`、
  `book.toml` 的 `site-url`、D016、`release.toml` pages 元数据和中英文
  README 在线阅读链接；不提交 `book/book/`，不改默认 CPU gate。
- [x] 读者视角 P0 修订（D018）：第 7–9 章小节标题去编号前缀；
  综合实验与比较卡移至书末“贯穿实验与对照”；`capstone-p1.md` 改名
  `capstone.md`；新增 `crosswalk-guide.md` 作为对照矩阵读者入口；
  全书证据标签统一为五类规范措辞；正文裸 `planning/` 引用改为
  GitHub 链接或导读页链接；首页删除审计语言。
- [x] 读者基础设施 P1：新增 `running-examples.md`（环境、首次构建、
  示例-章节对照表、已知 tracel-llvm 边界）和 `glossary.md`（按主题
  分组的中英术语表）；前言加入术语表/搜索说明；SUMMARY 增加
  “附录”部分；首页链接两个新页面。
- [x] WGPU 可选实验路径：`ch03-cubecl-kernel` 主程序在
  `--features wgpu` 下依次运行 CPU 与 WGPU Runtime 并对照 host
  reference（本机实测输出 `wgpu<wgsl>` 一致）；第 3 章实验节、
  运行指南和比较卡同步记录命令与边界；默认 CPU gate 不变。
- [x] 全书首批自制 SVG 图：`img/ch03-roofline.svg`、
  `img/ch06-pipeline-1f1b.svg`（替换原 ASCII 时间线）、
  `img/ch09-network-topology.svg`，浏览器目视核验通过。
- [x] 全书九章深度加厚（grok 复核后由 k3 执行）：每章补定量推演与
  worked example——ch01 预算实例；ch02 autodiff 重写（反向成本分析、
  带数字反向推演、tape 生命周期、激活内存预算、源码导读）+
  广播梯度/参数统计/拓扑序演算；ch03 算术强度双演算、合并访存、
  tile 计数推导；ch04 融合流量定量、浮点非法变换反例；ch05 队列
  消化/尾批/shuffle 内存演算；ch06 straggler、1F1B 空泡公式、环形
  AllReduce 流量推导；ch07 PTQ 带数字校准演算；ch08 折扣因子视野、
  replay 新鲜度、rollout 瓶颈演算；ch09 队首阻塞、α+β 数值、
  Young checkpoint 间隔公式。九章来源映射均补加厚记录。
- [x] 第二批配图与 LLM 边界：新增 `img/ch01-system-layers.svg`、
  `img/ch04-compiler-pipeline.svg`、`img/ch05-fpg-backpressure.svg`
  并替换对应 ASCII 图；第 1 章新增“关于大模型时代的主题”小节、
  第 7 章推理节新增“与大模型服务的边界”，明确 KV cache、
  continuous batching、MoE、RLHF 等为首版未覆盖的专题；正文
  `docs/TERM_GLOSSARY.md` 引用全部切换到书内术语表页。
- [x] 结构配图第二轮 + 练习体例：新增/重画 8 张 SVG（计算图、拓扑
  内存、autodiff tape、Cube 层次、训练闭环、部署路径、RL 环路；
  编译流水线改为左→右）；九章练习统一【基础】/【进阶】/【挑战】
  与折叠提示；`docs/AUTHORING.md` 补充练习体例。

## 本次交接

- 已完成（2026-08-10）：第 3.2 节对照表区分 Cube / Plane：Cube 为可共享
  资源的工作组，Plane 为组内更小协同子集；避免两行都写「一组 unit」。
- 验证：目视核对 `02-gpu-machine-model.md` 表与图标签。
- 偏差：无。
- 下一步：确认后提交推送；其余仍为候选 tag/归档决策。

## 前次交接（2026-08-09 读者口吻扫尾）

- 已完成（2026-08-09 读者口吻扫尾）：主路径去掉 CI/验收/D0xx/「源码核验」
  /「根 workspace」等审计腔；改为「默认示例」「本书示例」「你需要核对什么」
  等读者说法。附录证据账本按 D021 保留标签语。综合实验、ch3/6/7/8、
  `running-examples.md` 等已改。
- 验证：`mdbook build book`；`python3 tools/check_release.py
  --require-built-book --json` → `ok=true`、`errors=[]`、`warnings=[]`。
- 偏差：附录与 `docs/`/`planning/` 仍含项目自洽用语（有意）。
- 下一步：提交推送；确认 Pages；再决定候选 tag/归档。

## 前次交接（2026-08-09 M6）

- 已完成（2026-08-09 M6）：相对 OpenMLSys 的内容/结构补强。
  - M6a：AUTHORING 三轨与系统结论体例；九章收束；设备/Runtime 与
    控制面/数据面配图。
  - M6b：ch3/4 GPU·多 Runtime·stream；ch2 接口史/计算图；ch7 部署闭环；
    ch6 collective 源码导读；ch9 机柜数值衔接；产业对照短表。
  - M6c：`docs/OPTIONAL_PROFILES.md`、D022、`running-examples.md` 交叉引用。
  - 会话日志：`planning/session-logs/2026-08-09-m6-content-structure.md`。
- 验证：当时 `mdbook build` 与 release audit 通过；未改默认 Cargo feature。
- 偏差：真机 CUDA/NCCL 仍为“源码先、跑通后”；随后做了读者口吻扫尾。
- 下一步：并入口吻扫尾后提交。

## 前次交接（2026-08-09 D021）

- 已完成（2026-08-09 D021）：项目自洽材料整包后移附录。删除章首五标签、
  `ch01/08`、`crosswalk-guide.md`、`comparison-cards.md`；新建
  `appendix-scope-and-evidence.md` 与 `appendix-sources.md`；章末来源
  改为一句指针；SUMMARY 附录收纳二者；`check_release.py` 改校验附录
  与 ch01=7 小节；新增 D021，更新 AUTHORING/TERM_GLOSSARY/STATUS。
- 验证：`mdbook build book`；`python3 tools/check_release.py
  --require-built-book --json` → `ok=true`、`errors=[]`、`warnings=[]`；
  `git diff --check` 通过；主路径无「本章你能验证什么 / crosswalk-guide /
  comparison-cards / ch01/08」残留。
- 偏差：五标签与比较卡内容仍存在于附录（供需要者查阅）；lab 边界句与
  running-examples 留在主路径。
- 下一步：并入 M6 一并提交。

## 前次交接（2026-08-09 学习者文风）

- 已完成（2026-08-09 学习者文风）：按盘点三批改写。章首「证据状态」→
  「本章你能验证什么」；九章练习前言去 CI 句；README、ch02 阶段表、
  各章 lab 改为「你会学到/观察到」；降频固定快照/pins；来源节去
  planning/D0xx；新增 D020；同步 check_release/AUTHORING/术语表。
- 验证：当时 `mdbook build` 与 release audit 通过；随后由 D021 继续
  把五标签墙移出章首。
- 偏差：对照页当时仍在贯穿区；已由本次 D021 后移。
- 下一步：当时为提交；现并入 D021 一并提交。

## 前次交接（2026-08-08 MathJax）

- 已完成（2026-08-08 MathJax）：全面排查线上公式渲染失败。根因是
  mdBook 默认 MathJax 2 配置不识别正文使用的 `$...$`/`$$...$$`。新增
  `book/theme/head.hbs` 注入 `tex2jax` 美元分隔符（D019），修正 ch07/
  ch09 两处跨行行内公式与若干表述；加强 `check_release.py` 与
  AUTHORING 约定。
- 验证：`mdbook build book`；`check_release.py --require-built-book`
  `errors=[]`；Puppeteer 对 42 个含公式页面 typeset 后无裸 `$...$`
  残留且均有 `.MathJax` 节点。
- 偏差：无；浏览器仍依赖 MathJax CDN（D015）。
- 下一步：合并推送后确认 Pages 线上公式；再决定候选 tag/归档。

## 前次交接（2026-08-08 第六批）

- 已完成（2026-08-08 第六批）：结构配图与练习完善。新增
  `ch02-expr-graph.svg`、`ch02-topo-memory.svg`、
  `ch02-autodiff-tape.svg`、`ch03-cube-hierarchy.svg`、
  `ch06-training-loop.svg`、`ch07-serving-pipeline.svg`、
  `ch08-rl-loop.svg`，重画 `ch04-compiler-pipeline.svg`；九章练习
  页加难度标签与 `<details>` 提示（无完整答案）；更新 AUTHORING、
  chapter-sources 与 session log。
- 验证：13 张 SVG XML 校验通过；图片相对路径均可解析；
  `mdbook build book` 通过；`check_release.py --require-built-book`
  `errors=[]`、`warnings=[]`；`git diff --check` 通过；完整
  `make check` 退出码 0。
- 偏差：无。
- 下一步：提交全部修订并推送 main；确认 Pages 配图与折叠提示可渲染；
  后续候选为第二轮段落级加厚或更多结构图。

## 前次交接（2026-08-08 第五批）

- 已完成（2026-08-08 第五批）：全书九章内容深度加厚。方法：按
  “动机→推导→机制→源码导读→可观察→边界”配方逐节补缺口，以定量
  推演（worked example）为主；所有新事实按固定 revision 源码或示例
  测试核验，未引入未核验断言。逐章明细见九份
  `planning/chapter-sources/chNN.md` 的“2026-08-08 深度加厚记录”。
- 验证：每章 `mdbook build book` + `check_release.py
  --require-built-book`（`errors=[]`）；收尾完整 `make check`
  退出码 0、`git diff --check` 通过。
- 偏差：本地 `burn/` 镜像 HEAD 超前于 pin（pin 是祖先），
  `make check-local-sources` 会失败；源码核验均以
  `git show <pin>:<path>` 进行，不受影响。
- 下一步：当时为练习提示与第 2/8 章配图；现已由第六批完成。

## 前次交接（2026-08-08 grok 事实复核）

- 已完成（2026-08-08 事实复核）：对暂存读者修订做口径与事实核验。
  同步 `docs/TERM_GLOSSARY.md` 证据分类与正文五类标签；修正
  `planning/chapter-sources`、`planning/capstone-p1.md`、D012/D014/
  D018 中的旧标签；收紧第 7 章 LLM 边界（区分服务 runtime 与
  burn-onnx Attention 的 past KV 图转换）；术语表 Flex 表述改为
  “默认路径不走 Fusion/CubeCL”；`crosswalk-guide.md` 的 C/S/R/L/E
  字段说明与对照矩阵对齐。
- 验证：OpenMLSys/工具链/11 示例 crate/图片相对路径/GitHub blob
  路径/九章+综合实验证据标签集合均通过脚本核对；本地镜像确认
  `burn-flex` 无 `burn-fusion` 依赖、CubeCL 存在 `wgpu` feature 与
  `WgpuRuntime`/`CpuRuntime` 导出；六张 SVG XML 此前已校验。
- 偏差：先前交接称读者术语表与作者术语表“定义保持一致”不准确——
  证据分类行曾滞后，现已修正。
- 下一步：提交全部读者修订（含本复核）并推送 main；后续候选为
  练习提示/难度标注、第 2 章计算图配图和更多原理段落加厚。

## 前次交接（2026-08-08 第四批）

- 已完成（2026-08-08 第四批）：第二批配图（第 1/4/5 章分层、编译
  流水线、F/P/G 背压模型，替换 ASCII）、LLM 时代主题断层说明
  （第 1.5 节与第 7.5 节各一段，标记为首版未覆盖专题）、正文
  `docs/TERM_GLOSSARY.md` 七处引用切换到书内 `glossary.md`。
- 验证：三张新 SVG 经 XML 校验与浏览器截图目视核验；`mdbook build
  book`、release audit `errors=[]`、完整 `make check` 退出码 0、
  `git diff --check` 通过。
- 偏差：无（证据分类口径同步见上一条交接）。
- 下一步：当时为提交四批修订；现已并入事实复核。

## 前次交接（2026-08-08 第三批）

- 已完成（2026-08-08 第三批）：WGPU 可选实验路径与三张 SVG 图。
  修改 `examples/ch03-cubecl-kernel/src/main.rs`（feature 门控的 GPU
  对照运行）、第 3 章实验节、`running-examples.md`、比较卡；新增
  `book/src/img/` 三张图并接入第 3/6/9 章正文。
- 验证：`cargo run/test/clippy -p ch03-cubecl-kernel`（默认与
  `--features wgpu` 两种组合，均 `--locked`）通过；WGPU 实测输出与
  host reference 一致；三张 SVG 经 XML 校验与浏览器截图目视核验；
  完整 `make check` 退出码 0，release audit `errors=[]`。
- 偏差：修复了一处 lint 回归（`scale_reference` 导入需随 feature
  门控）；写入 `.svg` 时专用文件工具损坏了多字节字符，改用 Python
  写盘后通过 XML 校验——后续新增 SVG 应先跑 XML 解析校验。
- 下一步：提交三批修订并推送 main，确认 Pages 导航与图片可访问；
  后续候选为第 1/7 章 LLM 主题断层说明、更多章节配图、练习提示。

## 前次交接（2026-08-08 第二批）

- 已完成（2026-08-08 第二批）：读者基础设施 P1。新增
  `book/src/running-examples.md` 与 `book/src/glossary.md`；前言、首页、
  SUMMARY 同步更新；新增“附录”导航部分。
- 验证：`mdbook build book`、`check_release.py --require-built-book`
  （`errors=[]`、`warnings=[]`）、`git diff --check`、完整
  `make check` 均通过，退出码 0。
- 偏差：无；术语表为读者版（定义 + 章节链接），作者版用语约束仍以
  `docs/TERM_GLOSSARY.md` 为准，二者定义保持一致。
- 下一步：提交两批修订并推送 main，确认 Pages 导航；后续候选为
  WGPU 可选实验路径、关键图示（GPU 拓扑/roofline/流水线）和 LLM
  专题规划声明。

## 前次交接（2026-08-08 第一批）

- 已完成：读者视角 P0 修订（D018）。修改 `book/src/` 下 SUMMARY、
  首页、9 个章节着陆页、综合实验页（改名 `capstone.md`）、比较卡、
  新增 `crosswalk-guide.md`，以及第 1–9 章共 33 个小节文件；规划侧
  更新 `planning/chapter-sources/ch05.md`。
- 验证：`mdbook build book`、`tools/check_release.py
  --require-built-book`（`errors=[]`、`warnings=[]`）、
  `git diff --check`、完整 `make check`（含 offline gate 与 release
  audit）均通过，最终退出码 0。
- 偏差：无；`check_release.py` 的 include 白名单（仅 `examples/`）
  保持不变，对照矩阵以导读页 + GitHub 链接发布而非内嵌。
- 下一步：提交本批修订并推送 main，确认 Pages 部署后导航与新增
  页面可访问；P1 候选为书内术语表页、“如何运行示例”页和 WGPU
  可选实验路径。

## 前次交接（2026-08-02）

- 已完成：为九章候选版增加 GitHub Pages 静态发布（D016）。独立 deploy
  workflow 使用固定 `mdbook 0.4.51` 与 pinned Pages actions；`book.toml`
  使用 project-site 路径 `/mlsys_with_burn/`；README / STATUS /
  `release.toml` 记录预期 URL。
- 验证：本地 `mdbook build book`，产物含 `index.html`；workflow 会写入
  `.nojekyll`。现有 CI 完整 Rust gate 保持不变。
- 偏差：线上可达性依赖仓库 Settings → Pages 选择 GitHub Actions，并在
  推送 `main` 后由 Actions 实际部署；本机无法代替该一次性配置。
- 已完成：P1 与比较卡复审修正，避免以不同随机初始化模型比较 loss，
  并避免把协议模型误读为 Burn/生产 runtime。
- 验证：`cargo fmt --all --check`、capstone `cargo test`/Clippy（均
  `--locked --offline`）、两次 `cargo run` 输出一致、`mdbook build/test`、
  `tools/check_release.py --require-built-book --json` 和 `git diff --check`
  均通过；release audit `errors=[]`。
- 已完成：比较卡标题和 7 个章节引用锚点已统一；全书 `make check` 通过，
  包含 upstream check、mdBook build/test、workspace test/doctest/Clippy、
  10 个 CPU smoke、capstone smoke、offline gate 和 release audit。
- 验证：`make check` 最终退出码为 0；release audit
  `errors=[]`、`warnings=[]`。
- 已完成：读者可见术语清理完成；项目内部 P1 代号仅保留在 planning、
  示例 crate 和文件路径中，正文改用“综合实验”等通用描述。
- 验证：术语修正后的 `make check` 退出码为 0；release audit
  `errors=[]`、`warnings=[]`，IDE lint 无错误。
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
