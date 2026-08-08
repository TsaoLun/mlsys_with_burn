# 2026-08-08：读者视角 P0 修订（导航、标题、证据标签、对照矩阵入口）

## 用户目标与选择

用户要求评估项目作为 MLSys 教材的不足与突兀之处。分析报告给出后，用户
选择先处理 P0 突兀点：标题编号体例、审计语言残留、导航顺序、对照矩阵
读者可达性。

## 分析发现（供后续会话复用）

全书主要读者体验问题：

- 第 1–6 章小节 H1 不带编号，第 7–9 章带 `7.1` 等编号，体例不一；
- 综合实验（依赖第 5–7 章）与比较卡（依赖第 3–9 章）在 SUMMARY 中位于
  第 1 章之前；
- 正文以裸代码格式引用 `planning/comparison/openmlsys-v1-crosswalk.md`
  和 `planning/chapter-sources/chNN.md`，Pages 在线读者无法到达；
- `book/src/README.md` 残留 release audit / offline gate 等审计语言；
  `capstone-p1.md` 文件名残留内部代号 P1；
- 章节着陆页与比较卡使用两套不同的证据标签措辞。

后续仍未处理的结构性不足（详见聊天报告）：全书无插图（仅 ASCII 图）、
无 GPU 可选实验路径、第 7 章无端到端部署示例、缺 LLM 推理/训练主题、
缺书内术语表与“如何运行示例”页、mdBook 中文搜索基本不可用、练习无
提示或参考答案、贯穿实验只有第 5–7 章一条。

## 已执行的修改

1. 去掉第 7–9 章 24 个小节 H1 的 `7.x`/`8.x`/`9.x` 编号前缀，与第 1–6
   章体例一致（已确认没有指向这些带编号锚点的书内链接）。
2. `git mv book/src/capstone-p1.md book/src/capstone.md`；更新
   `SUMMARY.md`、第 5–7 章实验页链接和
   `planning/chapter-sources/ch05.md`。
3. SUMMARY 重排：综合实验与比较卡移至书末新增的
   “# 贯穿实验与对照” 部分，并登记新页面 `crosswalk-guide.md`。
4. 新增 `book/src/crosswalk-guide.md`：对照矩阵的读者导读，含 GitHub
   链接、五类证据标签定义和 C/S/R/L/E 字段说明。
   注意：`tools/check_release.py` 规定 mdBook include 只能来自
   `examples/`，因此对照矩阵正文不能 include 进书，只能以导读 +
   GitHub 链接方式发布。
5. 证据标签全书统一为比较卡的五类规范措辞：`CPU 可运行验证`、
   `源码核验`、`协议/成本模型`、`可选平台实验`、`未覆盖`；9 个章节
   着陆页和综合实验页在 `## 证据状态` 下增加统一的读者化引导句
   （保留字面量 “证据状态”，`check_release.py` 依赖它）。
6. `book/src/README.md` 首页删除 release audit/offline gate 表述，
   改为指向对照矩阵导读和比较卡，保留 MathJax CDN 边界说明。
7. 第 1–8 章末节中裸引用的 `planning/...` 路径改为 GitHub blob 链接
   或指向 `crosswalk-guide.md`；D010/D011 引用改为 DECISIONS.md 链接。

## 验证

- `mdbook build book` 通过；
- `python3 tools/check_release.py --require-built-book` 通过，
  `errors=[]`、`warnings=[]`；
- `git diff --check` 通过；
- `make check`（含 upstream 校验、mdBook build/test、fmt、Clippy、
  workspace test/doctest、10 个 CPU smoke、capstone smoke、offline
  gate、release audit）：结果记录于 STATUS.md 本次交接。

## 决策

新增 D018：读者面向修订的导航与标签口径（见 `planning/DECISIONS.md`）。

## 下一步

- 确认 `make check` 全绿后可提交；推送 main 后由 Deploy Pages 发布。
- P1 候选：书内术语表页、“如何运行示例”页、中文搜索说明。
- P1 候选：WGPU 可选实验路径（独立平台 profile，不改默认 CPU gate）。

## 追加（同日第二批，用户负责提交推送）

- 新增 `book/src/running-examples.md`：环境（Rust 1.95 固定工具链、
  mdBook 0.4.51、Python 3.11+）、首次构建需网络、`--locked/--offline`
  用法、11 个示例 crate 与章节对照表、tracel-llvm 已知边界、
  MathJax CDN 边界和 `make check` 说明。
- 新增 `book/src/glossary.md`：按六个主题分组的中英术语表（定义 +
  章节链接），末尾指向证据标签导读；作者版约束仍归
  `docs/TERM_GLOSSARY.md`。
- 前言加入术语表回查与“在线搜索基于英文分词、中文检索效果有限”的
  提示；SUMMARY 增加前置页和“附录”部分；首页链接两个新页面。
- 验证：`mdbook build book`、`check_release.py --require-built-book`
  （`errors=[]`、`warnings=[]`）、`git diff --check`、完整
  `make check` 退出码 0。
- 下一步候选：WGPU 可选实验路径；GPU 拓扑/roofline/流水线的 SVG 或
  mermaid 图示；第 1/7 章补充 LLM 时代主题断层的说明与专题规划。

## 追加（同日第三批：WGPU 路径 + SVG 图示）

用户选择同时做 WGPU 可选路径与关键图示。

WGPU 路径：

- `examples/ch03-cubecl-kernel` 原本已有 `wgpu` feature 和一个
  feature 门控测试，但读者无可运行的观察入口。修改 `main.rs`：启用
  `--features wgpu` 时依次运行 CpuRuntime 与 WgpuRuntime，打印两个
  runtime 名称与输出，并断言 WGPU 结果等于 host reference。
- 本机实测：`cargo run -p ch03-cubecl-kernel --features wgpu --locked`
  输出 `runtime: wgpu<wgsl>` 且与 reference 一致；默认与 wgpu 两种
  组合的 test/Clippy 均通过。默认 CPU gate（Makefile、CI）未改。
- 文档：第 3 章实验节第 7 小节、`running-examples.md` 新增“可选
  GPU 路径”、比较卡第 3 章条目均记录命令、前提（Metal/Vulkan/DX12
  adapter）和“正确性对照不是 GPU 性能结论”的边界。
- 踩坑：`scale_reference` 导入必须随 feature 门控，否则默认构建
  Clippy `-D unused-imports` 失败——`.cargo/config` 的双文件警告
  不影响结果。

SVG 图示：

- 新增 `book/src/img/ch03-roofline.svg`（Roofline 双屋顶与拐点）、
  `ch06-pipeline-1f1b.svg`（替换 6.5 节的 ASCII 时间线，信息等价）、
  `ch09-network-topology.svg`（Spine/ToR/节点/机柜分层与超额认购
  链路）。均为自制、白底卡片式，浅色/深色主题均可读。
- 踩坑：用 Write 工具写 `.svg` 时多字节字符（中文、全角括号、`·`、
  `→`）被损坏为无效 UTF-8，expat 解析报错；Markdown 写入无此问题。
  改用 `python3` 写盘后通过 `xml.dom.minidom` 校验，再经本地
  `http.server` + 浏览器截图逐张目视核验。后续新增 SVG 必须先做
  XML 校验。
- 验证：`mdbook build book`、`check_release.py
  --require-built-book`（`errors=[]`）、完整 `make check` 退出码 0。

下一步候选：第 1/7 章 LLM 主题断层说明；更多章节配图（第 4 章编译
栈、第 5 章数据供给 F/P/G）；练习提示/难度标注。

## 追加（同日第四批：第二批配图 + LLM 边界 + 术语引用收口）

- 配图：新增 `ch01-system-layers.svg`（五层系统分层 + 横跨带）、
  `ch04-compiler-pipeline.svg`（编译器侧/运行时侧两段流水线）、
  `ch05-fpg-backpressure.svg`（F/P/G 供给与有界队列背压），替换
  对应 ASCII 块；XML 校验 + 浏览器截图核验通过。仍沿用“Python
  写盘 + expat 校验 + 截图”流程，未再用 Write 直接写 SVG。
- LLM 边界：第 1 章生命周期节新增“关于大模型时代的主题”小节，
  第 7 章推理节新增“与大模型服务的边界”；明确 KV cache、paged
  attention、continuous batching、speculative decoding、MoE、
  预训练数据管道和 RLHF 为首版未覆盖专题，并指出它们与本书第
  3/7/8/9 章机制的关系。这是编辑性声明，未新增决策记录。
- 术语引用收口：正文 7 处 `docs/TERM_GLOSSARY.md` 裸引用改为指向
  书内 `glossary.md`；`pins.toml` 等操作性仓库路径保留（运行示例
  本就要求克隆仓库）。
- 验证：`mdbook build book`、release audit `errors=[]`、完整
  `make check` 退出码 0、`git diff --check` 通过。

下一步候选：练习提示/难度标注；第 2 章计算图与第 8 章 Actor–Learner
配图；按章节加厚原理段落。

## 追加（同日第五批：全书九章深度加厚）

背景：grok 复核后判断“教材内容深度有待提升”，用户指定由 k3 执行。
grok 另做了一轮口径/事实复核（标签同步、第 7 章 LLM 边界收紧、
crosswalk-guide 字段对齐），已并入工作区。

方法：按“动机→推导→机制→源码导读→可观察→边界”配方逐节检查，
以定量推演（worked example）为主要增量；新事实一律按 pins 固定
revision 源码（用 `git show <pin>:<path>` 读取，因本地 burn 镜像
HEAD 已超前于 pin）或示例测试核验。逐章内容：

- ch01：预算公式数值实例（训练状态内存、有效吞吐）。
- ch02：05 自动微分重写（前向/反向成本分析、带数字三节点反向推演
  含共享节点梯度累加、tape 生命周期=Step 消费自身+steps.remove、
  激活内存预算与 gradient checkpointing、逐文件源码导读）；
  02/03/04 补广播梯度归约、num_params 口算、拓扑序计数与峰值内存。
- ch03：scale vs GEMM 算术强度演算（0.125 vs ≈171 FLOP/字节）、
  合并访存 1/S 带宽、tile_load_counts 推导复现 8192→1024。
- ch04：add→exp 融合流量 20N→12N；f32 浮点非法变换反例。
- ch05：队列消化 Q/(G−F)、尾批计数、shuffle 索引内存。
- ch06：straggler 利用率、1F1B 空泡公式 (p−1)/(m+p−1)、环形
  AllReduce 每设备流量 2(p−1)S/p ≈ 2S。
- ch07：PTQ 校准带数字演算（s/z 求解、量化/反量化误差）、
  f32→int8/int4 存储压缩。
- ch08：γ 有效视野、replay 新鲜度预算 C/w、rollout 瓶颈移动数值例。
- ch09：队首阻塞场景、α+βl 教学数值（小消息延迟主导 vs 大梯度字节
  主导）、Young checkpoint 间隔 C*≈√(2WM)。

验证：每章 build + release audit `errors=[]`；收尾 `make check`
退出码 0。九份 chapter-sources 均补“2026-08-08 深度加厚记录”。

下一步候选：练习提示/难度标注；第二轮段落级加厚；第 2/8 章配图。

## 追加（同日第六批：结构配图 + 练习提示）

用户确认配图与练习整体完善计划后由 grok 执行。

配图（Python 写盘 + XML 校验）：

- 新增 `ch02-expr-graph.svg`、`ch02-topo-memory.svg`、
  `ch02-autodiff-tape.svg`、`ch03-cube-hierarchy.svg`、
  `ch06-training-loop.svg`、`ch07-serving-pipeline.svg`、
  `ch08-rl-loop.svg`，替换对应结构 ASCII；
- 重画 `ch04-compiler-pipeline.svg` 为单向左→右（编译器侧→运行时侧）。

练习：

- 第 1–9 章练习页统一【基础】/【进阶】/【挑战】标签；
- 每题后 `<details><summary>提示</summary>` 指向小节/示例/固定源码，
  不提供完整答案；
- `docs/AUTHORING.md` 增加练习体例约束。

规划：九份 chapter-sources 补「练习与配图完善」记录；STATUS 本次
交接改为第六批。

## 追加（同日事实复核与口径修正）

用户要求对暂存读者修订做全面事实核验。结果：

已核实为正确：

- OpenMLSys revision `9c289782…`、工具链 Rust 1.95.0 / mdBook 0.4.51
  与 `pins.toml`/`release.toml`/`rust-toolchain.toml` 一致；
- `running-examples.md` 所列 11 个示例 crate 均存在；Makefile 为
  10 个 CPU smoke + 独立 capstone smoke；
- GitHub blob 链接与图片相对路径可解析；九章着陆页 + 综合实验均含
  `## 证据状态` 与五类规范标签；
- CubeCL 固定快照存在 `wgpu` feature、`cubecl::wgpu::WgpuRuntime` 与
  `cubecl::cpu::CpuRuntime`；`burn-flex` 不依赖 `burn-fusion`；
- `burn-onnx` Attention 测试含 `past_k`/`past_v` 图转换，但不构成
  paged attention / continuous batching 服务 runtime。

已修正：

- `docs/TERM_GLOSSARY.md` 证据分类与正文对齐（此前滞后于 D018）；
- `planning/chapter-sources/ch01,ch06–ch09`、`planning/capstone-p1.md`、
  D012/D014/D018 影响段中的旧标签措辞；
- 书内术语表 Flex 定义改为“默认实验路径不走 Fusion/CubeCL”；
- 第 7 章 LLM 边界收紧为服务 runtime，并点明 burn-onnx past KV
  图转换不等于分页 KV/连续批处理；
- `crosswalk-guide.md` 的 L 字段与状态值顺序与对照矩阵对齐。

历史 session log 中的旧标签保留为当时记录，不回溯改写。
