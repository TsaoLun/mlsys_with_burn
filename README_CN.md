# MLSys with Burn

中文 | [English](README.md)

《机器学习系统：基于 Burn 与 Rust 的设计和实现》是一本面向 Rust
开发者的机器学习系统开源教材。项目以 Burn 为贯穿案例，并沿
Burn → CubeCL → CubeK 技术栈逐层深入张量、自动微分、编译、内核、
训练和部署。

本项目改编自 [OpenMLSys](https://github.com/openmlsys/openmlsys)，
但不是 OpenMLSys 或 Tracel 的官方项目，与二者均无隶属关系。

## 项目状态

当前是固定 `burn-0.22.0-pre.1` 源码快照的九章候选版，正在进行首个稳定
版本的发布审计。全书已有 CPU-first 可运行证据、逐文件来源 crosswalk
和明确的可选平台边界。已核验进度与剩余限制见
[`planning/STATUS.md`](planning/STATUS.md)。

## 依赖来源

构建与 CI 始终从 GitHub 获取 Burn，并使用 [`pins.toml`](pins.toml)
记录的精确 revision。Burn 自身的 manifest 会为 `0.22.0-pre.1` 写作
快照固定兼容的 CubeCL 与 CubeK revision。项目 Cargo manifest 禁止使用
本地 path 依赖。固定的 `burn-onnx` checkout 使用不同的 Burn revision，
因此只作为源码审计输入，不进入主 workspace。

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

这些目录被 Git 忽略，只用于让 Agent 快速阅读上游源码。项目构建和测试
不依赖它们，它们也不得影响 Cargo 的依赖解析。

## 快速开始

环境要求：

- Rust 1.95
- mdBook 0.4.51
- Python 3.11 或更高版本

```bash
make check
```

`make check` 统一使用 `--locked`，运行 CPU smoke suite，并在获取锁定依赖
后执行 Cargo offline gate。生成的教材位于 `book/book/`，不会提交到 Git。
浏览器公式阅读仍需要 mdBook 配置的 MathJax 资源；Cargo 离线可复现不等于
MathJax CDN 可以离线访问。

如果本地源码镜像存在，可以检查它们是否与远程快照一致：

```bash
make check-local-sources
```

## 项目结构

- `book/`：mdBook 中文教材
- `examples/`：与章节对应、可独立验证的 Rust 示例
- `planning/`：路线图、章节映射和实时状态
- `docs/`：架构、写作与维护规范
- `tools/`：版本和内容一致性检查
- `pins.toml` / `release.toml`：源码 revision 与发布工具版本
- `.cursor/rules/`、`AGENTS.md`：Agent 协作约束

## 许可证

教材正文是 OpenMLSys 的衍生作品，采用 CC BY-NC-SA 4.0。原创 Rust
示例与工具采用 MIT OR Apache-2.0。详情见
[`LICENSE_CN.md`](LICENSE_CN.md) 和 [`NOTICE.md`](NOTICE.md)。

