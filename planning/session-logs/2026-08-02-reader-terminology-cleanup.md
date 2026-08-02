# 读者可见术语清理

## 目标

清理书中只在项目内部有意义、或首次出现时不够通用的描述，保留 Burn、
CubeCL、系统论文和工程实践中的正式技术术语。

## 修改

- `P1` / `capstone` 的读者可见描述统一为“综合实验”；
- `crosswalk` 的正文描述统一为“逐文件对照矩阵”；
- 在第 1 章明确 C/S/R/L/E：
  正确性、源码、可运行性、学习路径、工程复核；
- 明确这些证据分类是本书自定义标签，不是 Burn 官方能力等级；
- 首次解释“默认 CPU 可运行路径（CPU-first）”、“发布审计
  （release audit）”、“Cargo 离线门禁（offline gate）”、
  “性能对等性（parity）”和“工作区快速验证（smoke test）”；
- 将“可选轨道”“主线”“协议卡”等项目化或隐喻性表达改为
  “可选路径”“根 workspace 的 Burn”“协议模型测试”等通用描述；
- 在 [`docs/TERM_GLOSSARY.md`](../../docs/TERM_GLOSSARY.md) 增加综合实验、
  逐文件对照矩阵、证据分类和默认 CPU 路径约束。

项目内部的 planning 标记、示例 crate 名和源码文件名没有机械重命名，
以避免破坏审计历史和代码路径。

## 验证

- `cargo fmt --all --check`
- `mdbook build book`
- `mdbook test book`
- `make check`，退出码 0
- release audit：`errors=[]`、`warnings=[]`
- IDE lint：无错误

## 结果

书内导航和示例链接仍然有效；读者不需要了解项目 P1 里程碑即可理解
综合实验和证据分类。
