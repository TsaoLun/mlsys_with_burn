# 2026-07-30：项目基建与第 1–2 章

## 会话目标

用户计划基于 OpenMLSys 中文版，使用 Burn、CubeCL、CubeK 和 burn-onnx
重写一本面向 Rust 开发者的机器学习系统 mdBook，并更新必要的新知识。
本次会话完成项目评估、名称与许可决策、工程基建、第 1 章和第 2 章首稿。

本日志是可审计的操作与推理摘要，不包含模型隐藏思维链。

## 用户确认的关键选择

1. 项目名采用 **MLSys with Burn**，建议公开仓库名 `mlsys-with-burn`。
2. 中文书名采用《机器学习系统：基于 Burn 与 Rust 的设计和实现》。
3. 教材明确作为 OpenMLSys 衍生作品，正文使用 CC BY-NC-SA 4.0。
4. 原创 Rust 示例和工具使用 MIT OR Apache-2.0。
5. 以 Burn `0.22.0-pre.1` 开发快照为写作基线。
6. Burn 及相关 Cargo 依赖必须来自 GitHub 固定 revision。
7. 根目录下的上游 clone 是可选、只读的 Agent 快速阅读镜像，不参与构建。
8. README 与 LICENSE 默认英文，并提供 `_CN` 中文版本切换。

## 初始仓库事实

- 工作区最初不是 Git 仓库，只包含五个独立上游 clone：
  `burn/`、`cubecl/`、`cubek/`、`burn-onnx/`、`openmlsys/`。
- 上游目录曾合计约 40GB，其中大部分是各自的 Rust `target/`。
- OpenMLSys v2 中文版只有九章骨架，实际可迁移正文主要在 v1。
- OpenMLSys v1 中文正文约 105 个 Markdown，采用 CC BY-NC-SA 4.0。
- OpenMLSys v2 规划为九章；本项目采用该九章结构，以 v1 为内容来源。
- Burn、burn-onnx 版本线为 `0.22.0-pre.1`，CubeCL 为
  `0.11.0-pre.1`，CubeK 为 `0.3.0-pre.1`。

## 固定源码快照

权威值记录在 `pins.toml`：

| 项目 | revision |
|---|---|
| Burn | `976aa9c5ec1d2dd3412710f99759e3c44bdff03d` |
| CubeCL | `be278a1e76aed881e2cc6b165414ee6103ca4634` |
| CubeK | `f82a6d07ebf35a1d446893b32712458744d80f13` |
| burn-onnx | `af2dfb43af43bf363dc2d7d858d933d86e2a65a8` |
| OpenMLSys | `9c289782ccbb165ac8ad7c960ecffc12942a5560` |

Burn manifest 决定其 CubeCL 与 CubeK revision。当前 burn-onnx manifest
仍使用较早的 Burn revision
`78f10aec1ca6c6ffb1edd17a0fa131ae59ad5403`，因此不能仅凭
`0.22.0-pre.1` 版本字符串假设两仓库 API 完全同步。

## 架构与范围推理摘要

### 为什么不是机械翻译

OpenMLSys 的编程接口、计算图、编译器和运行时叙事包含较多 Python、
MindSpore、Ascend 与 C++ 扩展语境。仅替换代码语言会保留错误的系统边界。
本项目采用以下写作顺序：

```text
框架无关原理
  → Burn 用户 API
  → Burn IR/Fusion
  → CubeCL 编译与运行时
  → CubeK 高性能算子
```

Rust 示例围绕类型、所有权、trait、错误和 Device 抽象重新设计。

### 为什么固定 Git revision

各项目仍处预发布开发期，同一版本字符串下的独立仓库可能 pin 不同 commit；
在线 Burn Book 也存在 API 漂移。固定 revision、Cargo.lock 和可运行测试
共同构成教材事实来源。

### 为什么不用本地 path 依赖

本地 clone 的目录布局只存在于作者/Agent 工作区。使用 Cargo `path` 或
`[patch]` 会使公开 CI 和读者构建不可复现，也可能掩盖 GitHub 快照不兼容。
因此：

- `Cargo.toml` 使用 Burn GitHub URL 和完整 rev；
- Burn 自身解析 GitHub 上的 CubeCL/CubeK rev；
- `.gitignore` 忽略五个可选上游镜像；
- `tools/check_upstreams.py` 默认只检查提交的远程元数据和 Cargo.lock；
- `make check-local-sources` 才额外核验本地只读镜像。

### 为什么第一章不复用原图

OpenMLSys 第一章框架图包含 Python、特定硬件和原书范围暗示。首稿使用重新
设计的文本图，避免让读者把历史语境误认为 Burn 已验证能力，也降低图片迁移
和编号重构成本。

## 已建立的工程基建

### 根项目

- 初始化根 Git，分支 `main`。
- `.gitignore` 忽略上游镜像、`target/`、mdBook 输出和本地工具状态。
- `rust-toolchain.toml` 固定 Rust 1.95。
- 根 Cargo workspace 管理章节示例。
- `Makefile` 提供：
  - `make book`
  - `make check-upstreams`
  - `make check-local-sources`
  - `make fmt`
  - `make lint`
  - `make test`
  - `make check`

### 许可证

- `LICENSE.md`：默认英文许可边界。
- `LICENSE_CN.md`：中文版本。
- `LICENSES/CC-BY-NC-SA-4.0.txt`
- `LICENSES/MIT.txt`
- `LICENSES/Apache-2.0.txt`
- `NOTICE.md`：OpenMLSys 改编署名、Burn 生态引用与独立性声明。

### Agent 协作

- `AGENTS.md`
- `.cursor/rules/project.mdc`
- `.cursor/rules/book-authoring.mdc`
- `.cursor/rules/rust-examples.mdc`
- `planning/MASTER_PLAN.md`
- `planning/STATUS.md`
- `planning/CHAPTER_MATRIX.md`
- `planning/DECISIONS.md`
- `docs/ARCHITECTURE.md`
- `docs/AUTHORING.md`

规则重点：

- 上游镜像只读；
- Cargo 禁止本地 path/patch；
- 示例是正文代码片段的唯一真相；
- 只描述固定源码可核验的能力；
- 每章记录 OpenMLSys 来源与主要改动；
- 完成工作后更新 STATUS 和验证证据。

### CI

`.github/workflows/ci.yml` 在不含本地上游 clone 的环境中执行：

1. 远程依赖 metadata 校验；
2. mdBook 构建；
3. rustfmt；
4. Clippy；
5. workspace 测试。

## Git 历史

- `e1769a5 chore: bootstrap MLSys with Burn project`
- `7a35a20 implement chapter 1 of the book`

第二章改动在本日志创建时仍位于工作区，尚未记录为单独提交。

## 第 1 章实现

### 文件

- `book/src/ch01-introduction.md`
- `book/src/ch01/01-applications-and-loads.md`
- `book/src/ch01/02-design-goals.md`
- `book/src/ch01/03-system-architecture.md`
- `book/src/ch01/04-burn-stack.md`
- `book/src/ch01/05-lifecycle-and-ecosystem.md`
- `book/src/ch01/06-stack-probe-lab.md`
- `book/src/ch01/07-exercises-and-sources.md`
- `planning/chapter-sources/ch01.md`
- `examples/ch01-stack-probe/`

### 核验结论

- Burn 0.22 用户 Tensor 不再携带 Backend 泛型。
- Device 包装 DispatchDevice，在运行时选择 Flex/CUDA/WGPU 等后端。
- Flex 是教材 CPU 默认后端，不经过 CubeCL/CubeK。
- CubeCL 提供 Kernel 语言、IR 和多后端运行时。
- CubeK 提供建立在 CubeCL 上的高性能算子。
- burn-onnx 是独立仓库，需单独核验 Burn revision。
- Remote 在该快照中仍是 Beta；QAT 不应描述为已支持。

### 实验

`ch01-stack-probe` 验证：

```text
pins.toml → Device::flex → DispatchDevice::Flex
→ dtype settings → sync → Tensor 数据读回
```

实际输出：

```text
snapshot: burn-0.22.0-pre.1
device: Device<Flex(Cpu)>
default float dtype: F32
default int dtype: I32
autodiff enabled: false
observed value after sync: 7
```

## 第 2 章实现

### 文件

- `book/src/ch02-programming-and-graph.md`
- `book/src/ch02/01-interface-and-workflow.md`
- `book/src/ch02/02-tensor-device-backend.md`
- `book/src/ch02/03-module-and-state.md`
- `book/src/ch02/04-computational-graph.md`
- `book/src/ch02/05-autodiff.md`
- `book/src/ch02/06-types-ir-scheduling.md`
- `book/src/ch02/07-labs.md`
- `book/src/ch02/08-exercises-and-sources.md`
- `planning/chapter-sources/ch02.md`
- `examples/ch02-tensor-basics/`

### 关键 API 事实

- 当前公开类型是 `Tensor<const D, K>`，不是旧 Burn Book 的
  `Tensor<B, D>`。
- 秩与张量类别主要在编译期，shape、精确 dtype 和 Device 在运行时。
- Module 和 ModuleRecord 不再携带 Backend 泛型。
- `Device::flex().autodiff()` 启用一阶反模式动态自动微分。
- `require_grad` 标记叶子，`backward` 返回 Gradients，`grad` 读取叶子梯度。
- `detach()` 切断旧图但保留 require-grad 意图。
- autodiff tape、Burn IR/Fusion 和 backend graph capture 是三种不同机制。
- Flex 使用 eager 前向和 autodiff tape，不自动经过 Fusion。

### 实验与断言

`ch02-tensor-basics` 当前包含四组测试：

1. 逐元素平均：`[1,2,3]` 与 `[3,4,5]` → `[2,3,4]`。
2. 广播：`[3,1] + [1,2]` → shape `[3,2]` 和固定数值。
3. TinyModel：Linear `3→2`，参数量 8，batch `[4,3]` → `[4,2]`。
4. 自动微分：
   - product `[1,7] * [4,7]` → `[4,49]`
   - left gradient → `[4,7]`
   - right gradient → `[1,7]`

## 重要失败尝试与修正

1. 初次 rustfmt 检查只发现文件尾空行，随后执行 `cargo fmt` 修正。
2. 第一章实验文档曾把 Flex 默认整数 dtype 写成 I64；实际运行是 I32，
   已修正并增加测试断言。
3. 第一章初稿曾把 CubeK 画成 CubeCL 之后的单一路径；修正为
   burn-cubecl 同时选择通用 CubeCL Kernel 或 CubeK 算子，再进入 runtime。
4. 第二章 Module include 最初只包含 impl，看不到 derive 和字段；已移动
   anchor 覆盖结构体定义。
5. 第二章练习最初误写 `detach()` 后 Tensor 不求导；固定源码显示 detach
   保留 require-grad，已区分原 Tensor 与新 detached leaf。
6. 第二章“训练/验证状态”最初容易被理解为 Module 布尔开关；已注明当前
   快照主要通过 autodiff Device 与 inner Device 表达有效模式。
7. `cubek/cubecl/` 曾是重复、未跟踪 clone，后续已移除，不是源码真相。
8. OpenMLSys 本地仓库没有可直接读取的 `LICENSE` 文件，许可依据为其
   README 的 CC BY-NC-SA 4.0 声明和官方完整许可文本。

## 关键验证记录

本次会话多次运行并最终通过：

```bash
python3 tools/check_upstreams.py
python3 tools/check_upstreams.py --check-local
mdbook build book
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
make check
```

第二章完成时测试结果：

- `ch01-stack-probe`：2 tests passed
- `ch02-tensor-basics`：4 tests passed
- 所有 doc-tests passed
- mdBook build passed
- Clippy `-D warnings` passed
- IDE linter 无错误

本机环境持续出现以下 Cargo 警告，但不属于项目文件错误：

```text
both ~/.cargo/config and ~/.cargo/config.toml exist
```

Cargo 使用 `~/.cargo/config`。后续 Agent 不应在未获授权时修改用户级配置。

## 当前已知风险

1. Burn 及相关项目仍是预发布快照，在线文档可能与源码不同。
2. burn-onnx 使用不同 Burn commit，第 7 章必须设计兼容验证。
3. Burn 分布式文档仍在演进，第 6、9 章不能把未来能力写成事实。
4. 第二章首稿虽经事实审阅，仍需要读者视角和术语一致性审校。
5. 当前教材主要使用文本图；后续若引入图形资产，必须记录来源和修改。

## 下一步

`planning/STATUS.md` 已把当前里程碑推进到 M2 基础篇。下一项任务：

1. 映射 OpenMLSys 加速器章节；
2. 核验 CubeCL/CubeK 当前编程模型、IR 和 CPU/GPU backend；
3. 设计无需专有驱动的 CubeCL CPU 核心实验；
4. 把可选 GPU 实验与基础 CPU 路径分离；
5. 开始第 3 章“AI 加速器与编程”。

