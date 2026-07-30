# MLSys with Burn

中文 | [English](README.md)

《机器学习系统：基于 Burn 与 Rust 的设计和实现》是一本面向 Rust
开发者的机器学习系统开源教材。项目以 Burn 为贯穿案例，并沿
Burn → CubeCL → CubeK 技术栈逐层深入张量、自动微分、编译、内核、
训练和部署。

本项目改编自 [OpenMLSys](https://github.com/openmlsys/openmlsys)，
但不是 OpenMLSys 或 Tracel 的官方项目，与二者均无隶属关系。

## 项目状态

当前处于基础设施与内容大纲阶段。实时进度、下一步任务和交接信息见
[`planning/STATUS.md`](planning/STATUS.md)。

## 本地布局

项目假设根目录下存在以下五个独立、只读的上游检出：

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

这些上游目录不会被根仓库跟踪。教材依赖的版本记录在
[`pins.toml`](pins.toml)，Agent 和贡献者不得擅自修改上游工作区。

## 快速开始

环境要求：

- Rust 1.95
- mdBook 0.4
- Python 3.11 或更高版本

```bash
make check-upstreams
make book
make test
```

生成的教材位于 `book/book/`。

## 项目结构

- `book/`：mdBook 中文教材
- `examples/`：与章节对应、可独立验证的 Rust 示例
- `planning/`：路线图、章节映射和实时状态
- `docs/`：架构、写作与维护规范
- `tools/`：版本和内容一致性检查
- `.cursor/rules/`、`AGENTS.md`：Agent 协作约束

## 许可证

教材正文是 OpenMLSys 的衍生作品，采用 CC BY-NC-SA 4.0。原创 Rust
示例与工具采用 MIT OR Apache-2.0。详情见
[`LICENSE_CN.md`](LICENSE_CN.md) 和 [`NOTICE.md`](NOTICE.md)。

