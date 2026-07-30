# 实时状态

更新日期：2026-07-30

## 当前里程碑

M1：第一条完整内容路径。

## 当前目标

完成第 1 章评审稿，并以第 2 章为下一条纵向切片。

## 进行中

- [ ] 对第 1 章进行读者视角审校，并设计第 2 章的小节与实验边界。

## 下一步

1. 将 OpenMLSys v1 的编程接口、计算图和自动微分各节映射到第 2 章。
2. 核对 Burn 0.22.0-pre.1 的 `Tensor`、`Device`、`Module` 和 Autodiff
   实现路径。
3. 为第 2 章拆出可独立验收的小节与实验，并复用第 1 章的来源模板。

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
- [x] 建立根 Git 基线提交 `e1769a5`。
- [x] 验证 `mdbook build book`、`cargo fmt --all --check`、
  `cargo clippy --workspace --all-targets -- -D warnings` 和
  `cargo test --workspace`。

## 已知问题

- `burn-onnx` 当前仓库版本为 0.22.0-pre.1，但其 manifest 仍 pin 到较早
  的 Burn commit；ONNX 章节必须按该关系单独验证，不能假定与本地 Burn
  HEAD 可互换。
- Burn 的分布式文档仍在演进，第 6、9 章不能只依赖 Burn Book。

## 交接模板

完成一次工作后更新本文件：

- 已完成：具体文件与内容。
- 验证：实际运行的命令和结果。
- 偏差：与计划不同之处及原因。
- 下一步：一个可以直接执行的动作。

