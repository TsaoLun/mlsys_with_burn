# 如何运行本书示例

本书的代码不在正文里复制粘贴，而是全部放在仓库的 `examples/` 目录中，
正文通过 mdBook include 引用其中的片段。每个示例都是一个可独立测试的
Rust crate，默认路径只需要 CPU，不需要 GPU 驱动、不下载数据、不访问
网络服务。

## 环境准备

- **Rust 1.95**：仓库根目录的 `rust-toolchain.toml` 已固定
  `1.95.0`，用 [rustup](https://rustup.rs/) 进入仓库目录即可自动
  切换到正确工具链；
- **mdBook 0.4.51**：只在本地构建本书站点时需要（`cargo install
  mdbook --version 0.4.51 --locked`）；只运行示例则不需要；
- **Python 3.11+**：只在运行项目级校验脚本时需要，普通阅读不需要。

## 第一次构建

Burn、CubeCL 等依赖按 `pins.toml` 记录的 GitHub 固定 revision 解析，
因此**首次构建需要网络**下载和编译依赖，耗时较长：

```bash
cargo fetch --locked
cargo test -p ch01-stack-probe --locked
```

`--locked` 保证使用 `Cargo.lock` 锁定的依赖版本。首次 fetch 之后可以
离线复核：

```bash
cargo test -p ch01-stack-probe --locked --offline
```

> 已知边界：CubeCL CPU 路径依赖 `tracel-llvm` 的 LLVM 资产，个别平台
> 或缓存环境下首次构建可能受影响；若本机构建失败，以 CI 结果为参照。

## 示例与章节对照

| 示例 crate | 对应章节 | 观察什么 |
|---|---|---|
| `ch01-stack-probe` | 第 1 章 | Device → Backend → Tensor 的执行栈路径 |
| `ch02-tensor-basics` | 第 2 章 | 广播、Module 参数统计、autodiff 梯度 |
| `ch03-cubecl-kernel` | 第 3 章 | 在 CPU 上运行 CubeCL Kernel 并对照 host reference |
| `ch03-tile-loads` | 第 3 章 | tile 加载计数与算术强度模型 |
| `ch04-fusion-inspector` | 第 4 章 | Fusion 执行计划、数值等价与同步边界 |
| `ch05-data-pipeline` | 第 5 章 | 数据守恒、固定 seed、多 worker 边界 |
| `ch06-training-loop` | 第 6 章 | CPU autodiff 训练循环、loss 下降 |
| `ch07-record-roundtrip` | 第 7 章 | ModuleRecord/Burnpack 参数往返保存与恢复 |
| `ch08-rl-rollout` | 第 8 章 | 确定性环境 rollout、replay、表格 TD 更新 |
| `ch09-cluster-simulator` | 第 9 章 | 集群调度与故障的确定性虚拟时间模拟 |
| `ch05-ch07-capstone` | [综合实验](capstone.md) | Dataset → 训练 → Record → 推理的端到端路径 |

## 运行某个示例

每个示例都可以运行测试或直接运行主程序观察输出：

```bash
cargo test -p ch06-training-loop --locked
cargo run  -p ch06-training-loop --locked
```

各章“实验”小节会给出该章示例的具体命令和输出解读。示例的测试断言
语义字段（shape、数值误差、守恒性），不断言墙钟时间，因此 CPU 上的
一次运行不是性能结论。

## 可选 GPU 路径

所有默认示例都只要求 CPU。第 3 章的 `ch03-cubecl-kernel` 额外提供一个
可选 `wgpu` feature：系统存在 Metal/Vulkan/DX12 等图形驱动时，可以用
同一个 CubeCL Kernel 在 WGPU Runtime 上运行并与 host reference 对照：

```bash
cargo run  -p ch03-cubecl-kernel --features wgpu --locked
cargo test -p ch03-cubecl-kernel --features wgpu --locked
```

没有 GPU 的环境直接跳过该 feature，不影响任何默认验证；该路径的正确性
对照也不能外推为 GPU 性能结论。详见第 3 章实验小节。

## 本地构建本书站点

```bash
mdbook build book   # 产物在 book/book/，不提交 Git
mdbook serve book   # 本地预览
```

浏览器中的公式由 MathJax CDN 渲染，离线打开时公式可能不渲染；Cargo
依赖的离线可复现不代表浏览器资源离线可用。

## 整体验证（贡献者用）

```bash
make check
```

该命令依次执行上游快照校验、mdBook 构建与测试、`cargo fmt`、Clippy、
workspace 测试与 doctest、10 个 CPU 冒烟运行、综合实验冒烟、离线门禁
和发布结构审计，不需要任何本地上游源码镜像。
