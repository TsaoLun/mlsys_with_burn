# 2026-08-08 MathJax `$` 分隔符修复

## 问题

线上 GitHub Pages 中行内公式原样显示，例如第 2 章实验页：

`共有 $3\times2=6$ 个权重`

读者看到裸美元符与 TeX 命令，而不是排版后的数学符号。

## 根因

1. mdBook 0.4.51 开启 `mathjax-support` 后注入 MathJax 2
   `TeX-AMS-MML_HTMLorMML`，默认只识别 `\(...\)` / `\[...\]`，
   **不识别** `$...$`。
2. 全书正文按 `$` / `$$` 书写（`docs/AUTHORING.md`），生成 HTML 保留
   美元分隔符，因此浏览器无法 typeset。
3. 附带问题：两处行内公式跨行（ch07 PTQ 演算、ch09 Young 间隔），
   在启用 `$` 后可能导致分隔符错配。

## 决策

见 `planning/DECISIONS.md` D019：保留 `$`/`$$` 源码约定，通过
`book/theme/head.hbs` 在 MathJax 脚本前注入 `tex2jax` 配置启用美元
分隔符；不把全书改写成 `\\(`/`\\[`。

## 操作

- 新增 `book/theme/head.hbs`（`inlineMath`/`displayMath` 含 `$`/`$$`）。
- `book/book.toml` 设置 `theme = "theme"`。
- 修正 ch07/ch09 跨行行内公式；ch03 `tile $=8$` 改为更清晰表述；
  ch02 labs 公式加空格。
- 更新 `docs/AUTHORING.md`、`tools/check_release.py`（禁止跨行行内
  `$`，要求生成 HTML 含 MathJax `$` 配置）。

## 验证

- `mdbook build book`
- `python3 tools/check_release.py --require-built-book --json`
  → `errors=[]`
- Puppeteer 等待 MathJax typeset 后扫描 SUMMARY 中 42 个含 `$` 页面：
  全部 `leftoverCount=0` 且存在 `.MathJax` 节点
- 关键页浏览器目视：ch02 labs / autodiff、ch03 GEMM、ch07 量化

## 交接

合并并推送 `main` 后，由 Deploy Pages workflow 发布；确认线上
`ch02/07-labs.html` 不再显示裸 `$3 \times 2 = 6$`。
