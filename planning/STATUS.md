# 实时状态

更新日期：2026-07-30

## 当前里程碑

M0：可持续写作基建。

## 当前目标

以第 2 章为首个内容纵向切片，建立节级来源映射与源码导读。

## 进行中

- [ ] 设计第 2 章的小节、实验和验收边界。

## 下一步

1. 将 OpenMLSys v1 的编程接口、计算图和自动微分各节映射到第 2 章。
2. 核对 Burn 0.22.0-pre.1 的 `Tensor`、`Device`、`Module` 和 Autodiff
   实现路径。
3. 为第 2 章拆出可独立验收的小节与实验。

## 已完成

- [x] 确定项目名为 “MLSys with Burn”。
- [x] 确定正文采用 CC BY-NC-SA 4.0，原创代码采用 MIT OR Apache-2.0。
- [x] 确定五个上游仓库保持并列、只读，并由 `pins.toml` 记录快照。
- [x] 确定以 Burn 0.22.0-pre.1 版本线展开。
- [x] 完成根 Git、许可证、Agent 规则和实时计划文档。
- [x] 完成九章 mdBook 骨架和第 2 章 CPU 张量示例。
- [x] 完成上游 pin 校验工具、Makefile 和 GitHub Actions CI。
- [x] 验证 `mdbook build book`、`cargo fmt --all --check`、
  `cargo clippy --workspace --all-targets -- -D warnings` 和
  `cargo test --workspace`。

## 已知问题

- `burn-onnx` 当前仓库版本为 0.22.0-pre.1，但其 manifest 仍 pin 到较早
  的 Burn commit；ONNX 章节必须按该关系单独验证，不能假定与本地 Burn
  HEAD 可互换。
- `cubek/cubecl/` 是未跟踪的重复 clone，不作为任何内容或构建依据。
- Burn 的分布式文档仍在演进，第 6、9 章不能只依赖 Burn Book。

## 交接模板

完成一次工作后更新本文件：

- 已完成：具体文件与内容。
- 验证：实际运行的命令和结果。
- 偏差：与计划不同之处及原因。
- 下一步：一个可以直接执行的动作。

