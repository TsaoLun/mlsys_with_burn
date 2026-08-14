# 实时状态

更新日期：2026-08-14

## 当前里程碑

M5：首个稳定版审计准备（可与内容修订并行；M5 tag 不阻塞）。

## 当前目标

读者主路径按 OpenMLSys 系统问题组织九章，并补上当代训练并行、推理服务
与集群控制面；Burn 用来指出实现落点。项目自洽（版本、证据标签、对照
矩阵）留在附录与工具链（D025）。静态书站经 GitHub Pages 可读。

## 进行中

- [ ] 等待发布者决定是否创建候选 tag/发布归档。
- [x] D025：正文重编——九章章首、阅读路径、系统结论、产业/crate 地图；
  第 8 章标为可选；主路径去掉「可核对 / CPU-first / 快照」叙事。
  见 `planning/session-logs/2026-08-14-curriculum-reframe.md`。
- [x] D026：并行策略整数实验、服务队列 TTFT/分块 prefill、mean 归约
  反向，以及 `capstone-infra.md` 合读页。
  见 `planning/session-logs/2026-08-14-infra-labs.md`。
- [x] 仓库 `origin/main` 已到 `82c0475`（用户手提交的 D025 重编），
  Pages 站点 `https://tsaolun.github.io/mlsys_with_burn/` 此前已可访问；
  本机仍不能代替管理员核对 Settings → Pages 的 Source/environment 保护。

## 下一步

1. 由发布者审阅 D026 与 `capstone-infra.md`，再决定候选 tag/归档。
2. 推送后抽查 Pages：`capstone-infra`、第 6 章动手版、第 7 章 TTFT 表。
3. 在能访问 `tracel-llvm` 固定资产的环境重跑完整 `make check`。
4. 真机 CUDA/NCCL 仅在 pins 与环境允许时追加可选命令；不得改默认 CPU
   gate（D022）。
5. 后续内容增量（不阻塞 tag）：KV 抢占/换出、真实数据集训练（需可选
   下载决策）、GEMM 阶梯更高级。

## 本次交接

- 已完成（2026-08-14）：D025 之后的三项内容增量（D026）。
  - `examples/ch06-parallel-strategies`：环形 AllReduce、GPipe/1F1B、
    ZeRO、TP AllGather 的整数成本模型。
  - `ch07-serving-queue-sim`：TTFT/TPOT 字段与 chunked prefill；
    连续批 = 无上界 chunk。
  - `ch02-ch04-op-anatomy`：`mean` 反向为 \(1/n\) 广播。
  - 正文动手版、练习改写、`book/src/capstone-infra.md`、Makefile/
    workspace/附录/来源映射同步。
- 验证：见本文件稍后更新的命令结果（提交后跑测试）。
- 偏差：未实现 KV 抢占；未声称 Burn 提供 vLLM/Megatron runtime。
- 下一步：发布者审阅；Pages 抽查新页。

## 已完成

- [x] 修复线上 `$...$` 公式不渲染：自定义 theme 启用 MathJax 美元分隔符
  （D019）；全书 42 个含公式页面 Puppeteer 核验通过。
- [x] 学习者文风改写（D020）与自洽材料后移附录（D021）：章首五标签/
  对照页/来源长文已迁入 `appendix-scope-and-evidence.md` 与
  `appendix-sources.md`。
- [x] M6a/M6b/M6c：章末系统结论、设备/Runtime 地图、ch2–4/6/7/9 GPU
  与原理加厚、`docs/OPTIONAL_PROFILES.md` + D022。
- [x] 内容与结构加固批次（2026-08-12）：修正实验语义/成本模型、补齐
  章节导航与桥接、修复损坏 SVG、修正误导性练习提示，并扩展
  `check_release.py` 防回归。
- [x] 全书审计修复批次（2026-08-13，P0–P3）：消除三处自相矛盾与
  5 处失效标题引用；op-anatomy 补 `matmul` State 真实摘录（含
  「对侧被追踪才 checkpoint」细节）；三面否定墙改为归属分工表
  （ch06/06、ch09/04、ch09/06）；ch09 六节小结去同款收尾；定语式
  「固定 X」约 35 处收敛；附录许可证 9→1、重复定义合并、比较卡降级；
  ch06/ch09 双总结去重；新增 5 张承重 SVG（dispatch 树、生态、存储
  层次、tape→optimizer、OffPolicy 循环）并删对应 ASCII/重复表格；
  ch07 章末补两个新实验、术语表补 PTQ/校准/scale-zero-point/连续
  批处理/KV cache 五词。验证：`mdbook build/test`、`cargo fmt`、
  `check_release --require-built-book` 全绿；图片引用与 SVG 使用
  无缺失；完整 `make check` 仍被本机 tracel-llvm 404 阻断（既有）。
  见 `planning/session-logs/2026-08-13-audit-repair.md`。

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

- 已完成（2026-08-13 夜 2）：深度批次五——CubeK 四层下钻与
  autotune 键（详见
  `planning/session-logs/2026-08-13-depth-batch-five.md`）。
  - ch03/04 续接「第七层：Routine 内部的四层组件」：
    batch/global/stage/tile 职责表（按 pinned 模块注释）、tile 五
    变体与 requires_accelerator、优化阶梯映射为显式类型、GEMM 阶梯
    实验定位为 global+Register 压平形态。
  - ch03/06 新增「一个 tune key 长什么样」：MatmulAutotuneKey 逐
    字段机制来源（anchor 尺寸分桶与 level 底数、stride 因子 2^10
    封顶的 swizzle 注释、512/2048 分桶、按键剪枝候选）；核实
    LocalTuner 按设备 ID 分 Tuner。
  - ch03/05 与 ch03/07 各一句互链。
- 验证：断言全部按 pin `git show/grep` 核实；`mdbook build/test`、
  `check_release.py --require-built-book`（`ok=true`）、
  `git diff --check` 通过；无代码改动。
- 下一步：提交推送；机制纵深剩余候选——`#[derive(CubeType)]`
  展开、attention/reduce 的四层对照走读、Fusion 即时 kernel 与
  CubeK 预制 kernel 的选择边界。

- 已完成（2026-08-13 夜）：深度批次四——宏黑箱、ONNX 算子旅程、
  解剖对照（详见 `planning/session-logs/2026-08-13-depth-batch-four.md`）。
  - ch02/03 与 ch03/03 各新增「宏在替你写什么」：引用本机
    cargo-expand 的**真实展开产物**讲透 `derive(Module)`（逐字段
    visit/map、字段名即参数路径来源）与 `#[cube]`（expand 登记 IR、
    define 产出 KernelDefinition 即缓存键、launch 装配），标注
    「节选并简化」。
  - ch07/02 新增「一个算子的旅程：Gemm」：注册→属性/形状→
    模式识别（Gemm 特例融合为 Linear，权重布局元数据）→ quote!
    代码生成→fixture 五站，全按 burn-onnx pin 核实。
  - 解剖页 add/sum 真实摘录对照（State 由数学决定），示例增两断言
    （共 6 测试）。
- 验证：6 测试 `--locked` 通过、clippy/fmt、`mdbook build/test`、
  `check_release.py --require-built-book`（`ok=true`）、
  `git diff --check` 均通过。
- 下一步：提交推送；机制纵深候选——matmul 的 CubeK
  tile/stage/global 下钻、autotune 键端到端追踪、
  `#[derive(CubeType)]` 展开对照。

- 已完成（2026-08-13 晚）：深度批次三——「算子解剖」贯穿页
  （详见 `planning/session-logs/2026-08-13-op-anatomy.md`）。定位
  澄清：不做明面「贡献者内容」，以机制纵深使贡献成为副产品。
  - `book/src/op-anatomy.md`：tanh 的十层解剖（API→契约→dispatch→
    autodiff 反向/checkpoint 策略→Flex→CubeCL→Fusion 双注册→IR
    词汇→backend-tests 同一断言切后端），全部按 pin 摘录并附失效
    模式与「换算子怎么走」。
  - `examples/ch02-ch04-op-anatomy`（4 测试）：前向/反向/数值核对/
    乘积法则断言，输出三行 0.00e0 引入页面。
  - ch02/05、ch04/03 交叉指针；附录、running-examples、Makefile、
    workspace 同步。
- 验证：4 测试通过、clippy/fmt、`mdbook build/test`（90 页）、
  `check_release.py --require-built-book`（`ok=true`）、
  `git diff --check` 均通过。
- 下一步：提交推送；同一哲学的后续候选——`#[cube]`/
  `#[derive(Module)]` 宏展开机制小节、ch07「一个 ONNX 算子的
  旅程」、二元/归约算子解剖对照。

- 已完成（2026-08-13 傍晚）：深度批次二——三个纯 Rust 可运行深度
  实验（详见 `planning/session-logs/2026-08-13-depth-batch-two.md`）。
  - `ch04-mini-pass-pipeline`（8 测试）：亲手写常量折叠/DCE/CSE +
    融合分组，附故意非法的 fast-math 消去（正文浮点反例的可运行
    版）；接入 ch04/02、ch04/07 §10 与练习。
  - `ch07-ptq-calibration`（6 测试）：min-max vs 百分位、
    per-channel、int8 GEMM；按「整体 MSE 掩盖校准交易」的真实误差
    结构写断言与正文；接入 ch07/04。
  - `ch07-serving-queue-sim`（5 测试，D024）：连续批处理 vs 静态批
    与 KV 预算的虚拟时间队列模型（延迟 268→90 ms、空转槽步
    5646→0、KV 扫描单调）；ch07/05 与 ch01/05 的 LLM 声明改为
    「机制模型可运行、Burn runtime 未覆盖」。
  - 工程：workspace/Makefile/running-examples/附录/练习/
    chapter-sources 同步；D024 记录扩围。
- 验证：三 crate 测试 19 项全过、clippy 零警告、
  `cargo fmt --all --check`、`mdbook build/test`、
  `check_release.py --require-built-book`（`ok=true`）、
  `git diff --check` 均通过。
- 偏差：PTQ 两个断言首版失败，按误差结构修正（见会话日志），失败
  本身写进了正文教学。
- 下一步：提交推送；深度候选剩 F（真实数据集训练，需可选下载
  决策）、GEMM 阶梯 3–5 级、迷你 tape 向量化（已留作练习）。

- 已完成（2026-08-13 下午）：深度批次一——全书第一处真实设备测量与
  「亲手造一遍」实验（详见
  `planning/session-logs/2026-08-13-depth-batch-one.md`）。
  - `examples/ch03-gemm-ladder`：默认纯 Rust 分块 GEMM 语义验证；
    `--features wgpu` 提供朴素/共享内存两级 CubeCL Kernel 与计时
    协议。本机 Metal release 实测 256/512/1024 方阵 tiled 加速
    4.71/4.22/4.44 倍；正文 ch03/05、ch03/07 第 8 节、
    OPTIONAL_PROFILES 按单机观测口径接入。
  - `examples/ch02-mini-autodiff`：约百行无依赖反向 tape（7 测试：
    数值梯度校验、扇出累加、分支、detach）；正文 ch02/05、ch02/07
    第 9 节接入，练习新增 2 题。
  - 依赖设计绕开 tracel-llvm：`ch03-gemm-ladder` 默认零 CubeCL
    依赖，wgpu 特性用 pins 同 revision 的 `cubecl`（无 `cpu`
    特性）；workspace/Makefile/Cargo.lock 同步。
- 验证：两 crate 测试（默认 3+7，wgpu 5）、clippy/fmt、
  `mdbook build/test`、`check_release.py --require-built-book`
  （`ok=true`）、`git diff --check` 均通过；wgpu 路径在本机 Metal
  实跑。
- 偏差：本机因上游资产缺失无法回归 `ch03-cubecl-kernel` 等
  cubecl-cpu 示例（见已知问题更新），CI/Linux 不受影响。
- 下一步：提交推送；有条件时在其他 GPU 平台复跑阶梯并把观测记入
  会话日志。后续深度候选：迷你 Pass 流水线（ch04）、PTQ 校准迷你
  实验（ch07）、KV cache/连续批处理模拟（需扩围决策）。

- 已完成（2026-08-13）：教科书化第一批（D023），面向读者体验的四项
  表达层修订；证据纪律与能力边界事实不变。
  - 参考文献：新增 `book/src/references.md`（九章分组约 45 条，每条
    一句导读，链接只用有把握的 arXiv/DOI/官方站）；SUMMARY 附录挂载；
    九章延伸阅读加指针；ch01/05 与 ch07/05 的 LLM 边界段补
    Orca/PagedAttention 出口。
  - 练习提示：九章约 220 条折叠提示全部题目专属化（小节链接 +
    示例观察点 + 实质方向）；顺带修正多处实质错向——ch06 两处、
    ch08 三处跨章错配、ch09 概念 8/源码 5/6 指错章与性能 5 过时
    指令（要求实现已存在的累计行为）；源码类事实均按 pins revision
    核验（详见各 chapter-sources 记录）；无套话残留。
  - 审计腔：正文「固定 X」修饰语 322→193 次；重复免责合并（重点
    ch05/06/07/08）；ch08/06 五连否改「需要/已提供/要补」分工表；
    三个小节标题去「固定」，引用同步。
  - 配图：新增 5 张 SVG 替换 ASCII——ch02 工作流、ch04 生命周期
    sync 对比、ch05 worker 通路、ch06 DDP 分层、ch08 Actor–Learner；
    qlmanage 渲染目视核验。
  - 体例：AUTHORING 四条新规、D023、九份 chapter-sources 追加记录、
    会话日志 `2026-08-13-textbook-first-batch.md`。
- 验证：`mdbook build/test book`（89 章）、`check_release.py
  --require-built-book --json`（`ok=true`、`errors=[]`、
  `warnings=[]`）、五张 SVG XML/读回/渲染核验、`git diff --check`
  均通过；无 Rust 代码改动。
- 偏差：「边界」词频 280→306，增量为新提示对小节标题的导航性链接，
  非免责句；SUMMARY 级标题未改名。外链未做在线可达性验证。
- 下一步：提交推送本批；线上抽查 `references.md`、五张新图与任一
  练习页；第二批候选见会话日志（更多配图、LLM 专章规划、外链年检）。

- 已完成（2026-08-12 夜 2）：第 3 章 matmul 逐层走查深化。
  - `ch03/04` 第 2 节从高层箭头链升级为「六个决策点」走查，每层按固定
    源码核实：API 层校验与 vec-mat 重解释、ops 层策略默认与 unwrap
    边界、kernel 准备层 broadcast-rhs 折叠与量化 binding、CubeK
    `launch_ref` 转发、Strategy 大枚举空间。
  - 教学主线：最早两处性能优化都是 kernel 存在之前的纯元数据变换。
  - `ch03/08` 源码题 3 的错配提示修正并指向走查小节；延伸阅读补
    numeric.rs / ops/tensor.rs / launch.rs / strategy.rs 具体路径；
    学习目标同步。
- 验证：`mdbook build/test book`、`check_release.py`
  （`ok=true`）、`git diff --check` 通过；本节为阅读走查，无新增示例
  代码需要编译。

- 已完成（2026-08-12 夜）：第 2 章张量字节视图深化（前几章 Burn 纵深）。
  - 示例 `ch02-tensor-basics` 新增 `inspect_tensor_bytes`：读回
    `TensorData{bytes, shape, dtype}` 并断言 `1.0f32` 小端字节、
    `-0.0` 符号位、24→48 字节的 dtype 宽度变化；共 8 项测试。
  - 正文 `ch02/02` 新增「字节视图」概念小节（含同宽度原地/跨宽度克隆
    的固定源码事实、与 Burnpack 同一 `Bytes` 的呼应）；`ch02/07` 新增
    实验第 3 节并顺延编号；学习目标同步。
- 验证：`cargo test/clippy/run -p ch02-tensor-basics --locked`、
  `mdbook build/test book`、`check_release.py`（`ok=true`）、
  `cargo fmt --all --check`、`git diff --check` 均通过。

- 已完成（2026-08-12 傍晚 4）：第 7 章 Burnpack 字节级深化。
  - 依据固定源码核验 `ModuleRecord::into_bytes` 走 `burn_pack::Writer`；
    `burn-pack/src/base.rs` 的 header/对齐/上限注释。
  - 示例新增最小 header 读取器：断言盘上 magic 为 `NRUB`（小端后果）、
    version=1、metadata 与 256 对齐数据区不重叠、截断/篡改报错；
    共 5 项测试通过。
  - 正文 `ch07/07` 新增「打开 Burnpack 字节」：格式规格、字节序细节、
    mmap 对齐动机，并用真实输出算清 12 字节参数 → 516 字节容器的对齐
    开销；学习目标与源码入口同步。
- 验证：`cargo test/clippy/run -p ch07-record-roundtrip --locked`、
  `mdbook build/test book`、`check_release.py`（`ok=true`）、
  `cargo fmt --all --check`、`git diff --check` 均通过。

- 已完成（2026-08-12 傍晚 3）：第 9 章三级通信域深化（参考原作
  cluster.md 的节点内/机柜间量级证据）。
  - 模拟器：`NetworkModel` 增加 `cross_node_multiplier`；成本按
    (rack,node) 分同节点/同机柜跨节点/跨机柜三档；TopologyAware 放置
    改为同节点→同机柜→跨机柜；`CommunicationCost` 新增
    `cross_node_bytes`。默认参数下既有断言与主程序输出不变。
  - 新测试：三档域 21/22/24us 精确成本与字节分类、node 优先放置；
    共 10 项通过。
  - 正文：`ch09/02` 新增「数量级直觉」小节（原作数字标注为写作代际的
    量级直觉、提醒 Gb/s vs GB/s）；`ch09/07` 通信成本改三档表；学习
    目标、系统结论、练习同步。
- 验证：`cargo test/clippy/run -p ch09-cluster-simulator --locked`、
  `mdbook build/test book`、`check_release.py`（`ok=true`）、
  `cargo fmt --all --check`、`git diff --check` 均通过。

- 已完成（2026-08-12 傍晚 2）：修复 SVG 中文被写成问号的事故。
  - 根因：通用文件写入工具把多字节中文字符替换为 `?`；`?` 是合法
    XML 字符，上午的 svg-assets 检查（UTF-8/控制字符/XML 解析）拦不住。
  - 修复：改用 Python 以 UTF-8 重写三张 SVG 并读回核对；严格扫描全部
    18 张（U+FFFD、`??`、控制字符、CJK 计数）确认健康。
  - 防回归：`svg-assets` 新增拒绝 U+FFFD 与连续 `??`；写非 ASCII 资产
    的纪律记入会话日志（写后读回核对、校验脚本用 `chr(0xFFFD)` 构造
    替换字符，避免通道剥离字面量造成空串假阳性）。
- 验证：`mdbook build book`、`check_release.py --require-built-book
  --json`（`ok=true`）、`git diff --check` 通过。

- 已完成（2026-08-12 傍晚）：读者语言可读性扫尾。
  - 主路径复查 CI/门禁/发布审计/D 编号/crosswalk 等维护者术语：均只在
    附录与 planning（有意，D020/D021）。
  - 修掉正文残留：`capstone.md` 裸「reference」；ch08 实验页
    「reference/回归观察值」与有歧义的「第 5 节」（改为具体小节链接）；
    ch01/06「上游」、ch01/07「本地固定上游」；四章练习页
    「固定上游中的权威入口」统一为「本书固定版本源码」。
  - `running-examples.md` 的 `make check` 段改为先声明「按章学习不必
    全仓库检查」，避免普通读者误以为必须运行贡献者检查。
  - 保留项及理由：`pins.toml`/`checkout`/`revision`（动手指导必需）、
    `workspace`（GPU 工作区内存，非 Cargo）、`host reference`（术语表
    已定义）、「镜像」（容器镜像）。
- 验证：`mdbook build book`、`check_release.py --require-built-book
  --json`（`ok=true`）、`git diff --check` 通过。

- 已完成（2026-08-12 下午）：第 8 章回放驱动学习加深（读者向）。
  - `examples/ch08-rl-rollout` 新增 `run_replay_driven` 阶段：先只收集，
    再从 replay batch 真正驱动同一 TD 规则；`capacity = 1` 的确定性对照
    显示 `initial_right_q` 精确为 0（容量截断数据分布），与在线路径的
    1.2125 形成因果对照；新增字段对齐/配置校验测试，共 9 项测试通过。
  - 正文 `ch08/07-rollout-lab.md` 改为两阶段叙事，先讲在线 vs 回放的
    数据分布差异再给代码；记录写入 replay 的 done 是合并标志的边界；
    练习页新增一道与实验闭环的对照题；章首页与系统结论同步。
- 验证：`cargo test/clippy/run -p ch08-rl-rollout --locked`、
  `mdbook build/test book`、`check_release.py --require-built-book --json`
  （`ok=true`）、`cargo fmt --all --check`、`git diff --check` 均通过。
- 下一步：与上午的内容结构批次一并提交推送；其余见下条交接。

- 已完成（2026-08-12）：内容与结构合理性强心针。
  - 实验语义：第 6 章 `final_loss` 改为最终参数重新评估；第 8 章拆分
    done/truncated，明确 replay sample 不参与在线 TD，并固定
    `initial_right_q=1.2125`；第 9 章把通信成本改为每个同步 step 累计，
    覆盖失败前与 checkpoint replay 的 step。
  - 结构：第 1–4 章补齐可点击小节列表，九章章末补前后桥接；修正第 1 章
    阅读路径与先修关系；综合实验增加分段阅读和学习者改动任务。
  - 误导修正：三张损坏 SVG 重写；第 3/4/7/8/9 章练习提示改到正确概念；
    「计算强度」统一为「算术强度」；正文 Cargo 命令补齐 `--locked`；
    `running-examples.md` 增加固定源码检出说明。
  - 防回归：`check_release.py` 新增 `cargo-command-locking` 与
    `svg-assets` 检查。
- 验证：`mdbook build book`、`mdbook test book`、受影响示例测试与
  Clippy、五个 `cargo run` 输出观察、18 张 SVG XML/控制字符检查、
  `python3 tools/check_release.py --require-built-book --json`
  （`ok=true`、`errors=[]`、`warnings=[]`）、`cargo fmt --all --check`、
  `git diff --check` 均通过。
- 偏差：完整 `make check` 在本机止于 `tracel-llvm-bundler v22.1.4-5`
  下载 `macos-x64.checksums.json` 的上游 404；失败发生在 CubeCL CPU
  构建脚本，不是本次代码或文档断言。需在资产可达环境重跑完整 gate。
- 下一步：提交推送本批修订；线上抽查 Pages；发布者再决定候选 tag/归档。

## 前次交接（2026-08-10 术语缺口）

- 已完成（2026-08-10）：读者向术语缺口处理（高/中优先项）。
  - `glossary.md` 增补：host reference、Flex/`Device::cpu()`、
    ComputeClient、fallback、autotune/特化键、fuser、Fusion stream、
    ExecutionStrategy 同名、rank、reorder buffer、Dispatch/Bridge 等。
  - 首次出现补中英定义：`ch02/02`、`ch03/03`–`04`/`06`/`07`、`ch04/03`、
    `ch05/01`、`ch06/01`、`ch09/01`；`running-examples` 同步措辞。
  - 修 `ch06/02`「host device」歧义；`docs/TERM_GLOSSARY.md` 对齐作者表。
  - 此前已改：host reference 定语义 / CpuRuntime 只验证；软化钉死/锚点。
- 验证：目视核对术语表新行与各章首次定义句；`rg` 无「钉死 / host device」
  残留。
- 偏差：低优先级单项（occupancy、HtoD、staging 等）仍靠章内上下文，
  未全部入表，避免术语表膨胀。
- 下一步：确认后提交推送；其余仍为候选 tag/归档决策。

## 前次交接（2026-08-10 语义锚点）

- 已完成（2026-08-10）：修正「默认 CPU/Runtime = 语义权威」类表述。
  - `ch03/03`：阅读顺序改为 host reference 定可观察语义，再在
    `CpuRuntime` 验证；并写明 CPU 非语义定义、不可外推 Plane/完成边界。
  - 同步软化 `ch01/04`、`ch02/01`、`ch04/01`、`ch07/01`、`ch07/04` 中
    「钉死/钉在/锚点」等易误导措辞（版本钉扎用语未改）。
- 验证：`rg` 确认正文无「钉死 / 钉在 / 锚点是 CpuRuntime」残留；目视
  核对阅读顺序与第 5 节开发顺序一致。
- 偏差：无。
- 下一步：确认后提交推送；其余仍为候选 tag/归档决策。

## 前次交接（2026-08-10 Cube/Plane）

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
- `tracel-llvm v22.1.4-5` 的 GitHub release **没有 macos-x64 资产**
  （只有 linux-AArch64/x64、macos-AArch64、windows-x64；2026-08-13
  经 API 核实）。Intel macOS 上凡依赖 `cubecl-cpu` 的构建都会在
  bundler 下载 `macos-x64.checksums.json` 时 404——这不是缓存问题。
  规避：`ch03-gemm-ladder` 的依赖设计（默认零 CubeCL、wgpu 特性不带
  `cpu`）不受影响；其余 CubeCL CPU 示例在该平台无法本地回归，以
  CI/Linux 结果为准。

## 交接模板

完成一次工作后更新本文件：

- 已完成：具体文件与内容。
- 验证：实际运行的命令和结果。
- 偏差：与计划不同之处及原因。
- 下一步：一个可以直接执行的动作。
