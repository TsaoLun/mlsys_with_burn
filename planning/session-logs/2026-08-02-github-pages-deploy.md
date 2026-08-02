# GitHub Pages 静态书站发布

## 目标

将九章候选版 mdBook 以静态站点形式发布到 GitHub Pages，供在线阅读；
不改默认 CPU gate，不提交生成目录，不引入 Deno Deploy。

## 固定事实

- 仓库：`https://github.com/TsaoLun/mlsys_with_burn`
- 预期站点：`https://tsaolun.github.io/mlsys_with_burn/`
- mdBook：`0.4.51`（与 `release.toml` / CI 一致）
- 构建输出：`book/book/`（已在 `.gitignore`）
- MathJax：浏览器仍依赖 CDN（D015）

## 方案取舍

- 选定 GitHub Pages：教材已是 mdBook 静态产物，与现有 pinned CI 对齐。
- 不采用 Deno Deploy：对纯静态 HTML 无额外验收收益。
- deploy job 不重跑完整 Cargo 测试：发布门禁仍由现有 `CI` workflow
  承担，避免重复耗时。

## 已完成的修改

- 新增 `.github/workflows/deploy-pages.yml`：
  - 触发：`main` push、`workflow_dispatch`
  - `mdbook build book` → `touch book/book/.nojekyll` →
    `upload-pages-artifact` → `deploy-pages`
  - action 钉 commit SHA：
    - `actions/checkout@11d5960a326750d5838078e36cf38b85af677262` (v4.2.2)
    - `peaceiris/actions-mdbook@ee69d230fe19748b7abf22df32acaa93833fad08` (v2)
    - `actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9` (v5.0.0)
    - `actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128` (v5.0.0)
- 更新 `book/book.toml`：`site-url = "/mlsys_with_burn/"`，并设置
  `git-repository-url`。
- 新增 D016；更新 `release.toml` `[pages]`、中英文 README、`STATUS.md`。

## 验证

- 本地：`mdbook build book`；确认 `book/book/index.html` 存在，并写入
  `.nojekyll` 后检查产物边界（生成目录仍被 gitignore）。
- 不运行完整 `make check`（本次不改示例/依赖）；CI workflow 未改逻辑。

## 仓库侧一次性步骤

1. Settings → Pages → Build and deployment → Source = **GitHub Actions**
2. 推送 `main` 或手动运行 `Deploy Pages`
3. 打开 `https://tsaolun.github.io/mlsys_with_burn/`，抽查目录页与含
   公式章节

## 下一步

启用 Pages source 并确认首次部署成功；随后由发布者决定候选 tag/归档。
