# 会话日志

本目录保存跨 Agent 可读的工作记录，目的是让后续会话恢复事实、决策、
操作和验证上下文，而不依赖聊天历史。

## 日志包含

- 用户目标与明确选择；
- 调研事实和固定源码证据；
- 方案取舍及可公开的推理摘要；
- 修改过的文件和执行过的关键命令；
- 验证结果、已知问题和下一步；
- 失败尝试与修正，避免后续 Agent 重复踩坑。

## 日志不包含

- 模型隐藏的内部思维链；
- 密钥、令牌、账号信息或无关环境数据；
- 大段可由 Git、CI 或源码直接恢复的原始构建输出；
- 上游源码副本和生成目录。

## 使用方式

Agent 开始工作时依次阅读：

1. `planning/STATUS.md`
2. `planning/DECISIONS.md`
3. 本目录中日期最新、且与当前任务相关的日志
4. 对应的 `planning/chapter-sources/chNN.md`

日志是审计和交接材料，不替代 Git 历史、`pins.toml` 或可运行测试。

## 索引

- [`2026-08-14-curriculum-reframe.md`](2026-08-14-curriculum-reframe.md)：
  读者主路径重编（D025）：系统课章首、产业/crate 地图、并行与服务加厚。
- [`2026-07-30-bootstrap-ch01-ch02.md`](2026-07-30-bootstrap-ch01-ch02.md)：
  项目评估、基建、远程依赖策略及第 1–2 章实现。
- [`2026-07-31-ch03-accelerator.md`](2026-07-31-ch03-accelerator.md)：
  第 3 章加速器与 CubeCL/CubeK 实验。
- [`2026-07-31-ch04-compiler-runtime.md`](2026-07-31-ch04-compiler-runtime.md)：
  第 4 章编译器、运行时与 FusionInspector。
- [`2026-07-31-backfill-ch01-ch04.md`](2026-07-31-backfill-ch01-ch04.md)：
  第 1–4 章术语/原理/实验补全与计划文档关闭。
- [`2026-08-01-ch05-data-processing.md`](2026-08-01-ch05-data-processing.md)：
  第 5 章数据处理系统来源映射、Burn DataLoader 核验和 CPU 实验。
- [`2026-08-01-ch06-training-systems.md`](2026-08-01-ch06-training-systems.md)：
  第 6 章训练系统来源映射、Burn optimizer/DDP 核验和 CPU 训练循环。
- [`2026-08-01-ch07-model-serving.md`](2026-08-01-ch07-model-serving.md)：
  第 7 章模型服务来源映射、burn-onnx/Record/Remote 核验和 CPU artifact
  round-trip。
- [`2026-08-01-ch08-reinforcement-learning.md`](2026-08-01-ch08-reinforcement-learning.md)：
  第 8 章强化学习来源映射、burn-rl/burn-train 核验和 CPU rollout/replay
  实验。
- [`2026-08-01-backfill-ch01-ch08.md`](2026-08-01-backfill-ch01-ch08.md)：
  第 1–8 章全面对照审计、原理回补、实验验证和第 9 章交接。
- [`2026-08-01-ch09-gpu-cluster.md`](2026-08-01-ch09-gpu-cluster.md)：
  第 9 章集群拓扑、调度、通信、故障边界和 CPU 模拟器。
- [`2026-08-01-p0-p1-openmlsys-comparison.md`](2026-08-01-p0-p1-openmlsys-comparison.md)：
  OpenMLSys 逐文件 crosswalk、发布门禁、CPU capstone、协议比较卡和终验收。
- [`2026-08-02-github-pages-deploy.md`](2026-08-02-github-pages-deploy.md)：
  GitHub Pages 静态书站 workflow、site-url、D016 与启用步骤。
- [`2026-08-02-p1-comparison-review.md`](2026-08-02-p1-comparison-review.md)：
  P1 贯穿实验与 OpenMLSys 比较卡的必要性、问题修正和验证。
- [`2026-08-02-comparison-card-heading.md`](2026-08-02-comparison-card-heading.md)：
  比较卡章节标题、mdBook 锚点和全书校验结果。
- [`2026-08-02-reader-terminology-cleanup.md`](2026-08-02-reader-terminology-cleanup.md)：
  读者可见项目术语清理、证据分类说明和全书校验结果。
- [`2026-08-08-reader-facing-p0-fixes.md`](2026-08-08-reader-facing-p0-fixes.md)：
  读者视角 P0–P1 修订、事实复核、九章深度加厚，以及第六批结构配图
  （8 张 SVG）与九章练习难度/提示体例。
- [`2026-08-12-content-structure-hardening.md`](2026-08-12-content-structure-hardening.md)：
  内容合理性与结构加固：实验语义/成本模型修正、章导航与桥接、SVG 修复、
  练习提示纠偏和内容发布检查扩展。

