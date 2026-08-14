# MLSys with Burn

中文 | [English](README.md)

《机器学习系统：基于 Burn 与 Rust 的设计和实现》是一本面向 Rust
开发者的机器学习系统开源教材。章节沿 OpenMLSys 的系统问题组织，并以
Burn → CubeCL → CubeK 为贯穿实现：从张量接口到 Kernel，从数据管道到
训练、服务与 GPU 集群。

本项目改编自 [OpenMLSys](https://github.com/openmlsys/openmlsys)，
但不是 OpenMLSys 或 Tracel 的官方项目，与二者均无隶属关系。

在线阅读：https://tsaolun.github.io/mlsys_with_burn/

## 项目状态

九章候选版，示例对齐 Burn `0.22.0-pre.1`。进度与已知限制见
[`planning/STATUS.md`](planning/STATUS.md)。

## 依赖来源

构建与 CI 从 GitHub 获取 Burn，revision 记录在 [`pins.toml`](pins.toml)。
Burn 的 manifest 决定兼容的 CubeCL 与 CubeK revision。项目 Cargo
manifest 禁止使用本地 path 依赖。`burn-onnx` 使用另一份 Burn 提交，
只作源码阅读，不进入根 workspace。

可以在项目根目录放置以下可选、只读的源码镜像：

```text
mlsys_with_burn/
├── burn/
├── burn-onnx/
├── cubecl/
├── cubek/
├── openmlsys/
├── book/
└── examples/
```

这些目录被 Git 忽略，只用于快速阅读上游源码。构建和测试不依赖它们。

## 在线阅读

静态站点由 GitHub Pages 发布：

https://tsaolun.github.io/mlsys_with_burn/

推送到 `main` 后由
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
自动重建。浏览器公式渲染依赖 MathJax CDN（见 D015 / D016）。

## 快速开始

环境要求：

- Rust 1.95
- mdBook 0.4.51
- Python 3.11 或更高版本

```bash
make check
```

`make check` 使用 `--locked`，运行默认 CPU 示例，并在获取锁定依赖后
执行 Cargo offline gate。生成的教材位于 `book/book/`，不会提交到 Git。
本地预览：

```bash
mdbook serve book
```

浏览器公式阅读仍需要 MathJax 资源。

如果本地源码镜像存在，可以检查它们是否与远程 revision 一致：

```bash
make check-local-sources
```

## 项目结构

- `book/`：mdBook 中文教材
- `examples/`：与章节对应的 Rust 示例
- `planning/`：路线图、章节映射和实时状态
- `docs/`：架构、写作与维护规范
- `tools/`：版本和内容一致性检查
- `pins.toml` / `release.toml`：源码 revision 与发布工具版本
- `.cursor/rules/`、`AGENTS.md`：Agent 协作约束

## 许可证

教材正文是 OpenMLSys 的衍生作品，采用 CC BY-NC-SA 4.0。原创 Rust
示例与工具采用 MIT OR Apache-2.0。详情见
[`LICENSE_CN.md`](LICENSE_CN.md) 和 [`NOTICE.md`](NOTICE.md)。
