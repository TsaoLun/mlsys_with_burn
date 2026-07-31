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

