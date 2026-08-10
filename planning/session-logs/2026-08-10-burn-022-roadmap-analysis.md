# 2026-08-10 Burn 0.21 后 PR / 0.22 目标分析

## 范围

只读分析 `tracel-ai/burn` 在 `v0.21.0` 之后的 PR 与发布说明，归纳
0.22 及后续意图；写入
`planning/upstream/burn-0.22-roadmap-from-prs.md`。未改 pins、示例或正文。

## 操作

1. 核对 releases：`v0.21.0`（2026-05-07）、`v0.22.0-pre.1`（2026-07-29）。
2. 统计 merged PR（>2026-05-07，约 250+）并按主题聚类。
3. 阅读 0.21 blog “What’s Next” 与核心 PR 正文（dispatch、remote、
   store、LoRA/reparam、fusion、量化、pliron）。
4. 记录对本项目 pin/章节的含义。

## 验证

- `gh` 对 releases / PR search / `gh pr view` 抽样。
- 发布文 What’s Next 原文与分析文档交叉引用。

## 交接

- 文档：`planning/upstream/burn-0.22-roadmap-from-prs.md`
- 下一步：若要升 pin，先写 DECISIONS，再重跑全书审计；勿把 open PR
  写成已支持能力。
