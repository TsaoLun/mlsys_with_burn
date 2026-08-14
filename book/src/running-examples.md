# 如何运行本书示例

本书的代码不在正文里复制粘贴，而是全部放在仓库的 `examples/` 目录中，
正文通过 mdBook include 引用其中的片段。每个示例都是一个可独立测试的
Rust crate。没有 GPU 也能跑完九章默认路径。

## 环境准备

- **Rust 1.95**：仓库根目录的 `rust-toolchain.toml` 已固定
  `1.95.0`，用 [rustup](https://rustup.rs/) 进入仓库目录即可自动
  切换到正确工具链；
- **mdBook 0.4.51**：只在本地构建本书站点时需要（`cargo install
  mdbook --version 0.4.51 --locked`）；只运行示例则不需要；
- **Python 3.11+**：维护仓库脚本时才需要，普通阅读与跑示例不需要。

## 第一次构建

依赖按仓库锁定的 GitHub 版本解析，因此**首次构建需要网络**下载和编译，
耗时较长：

```bash
cargo fetch --locked
cargo test -p ch01-stack-probe --locked
```

`--locked` 使用 `Cargo.lock` 锁定的依赖版本。首次 fetch 之后可以离线
再跑同一条测试：

```bash
cargo test -p ch01-stack-probe --locked --offline
```

> 已知边界：CubeCL CPU 路径依赖 `tracel-llvm` 的 LLVM 资产，个别平台
> 或缓存环境下首次构建可能较慢或失败；可换机器重试，或先跑不依赖
> CubeCL 的第 1、2 章示例。

## 示例与章节对照

| 示例 crate | 对应章节 | 观察什么 |
|---|---|---|
| `ch01-stack-probe` | 第 1 章 | Device → Backend → Tensor 的执行栈路径 |
| `ch02-mini-autodiff` | 第 2 章 | 一百行反向模式 tape：拓扑序、梯度累加、分支与 detach |
| `ch02-tensor-basics` | 第 2 章 | 广播、Module 参数统计、autodiff 梯度 |
| `ch03-cubecl-kernel` | 第 3 章 | 在 CPU 上运行 CubeCL Kernel，并对照主机参考实现（host reference） |
| `ch03-gemm-ladder` | 第 3 章 | 纯 Rust 分块 GEMM 语义验证；可选 `wgpu` 实测朴素/tiled 差距 |
| `ch03-tile-loads` | 第 3 章 | tile 加载计数与算术强度模型 |
| `ch04-fusion-inspector` | 第 4 章 | Fusion 执行计划、数值等价与同步边界 |
| `ch04-mini-pass-pipeline` | 第 4 章 | 亲手写常量折叠/DCE/CSE，含浮点非法变换反例 |
| `ch05-data-pipeline` | 第 5 章 | 数据守恒、固定 seed、多 worker 边界 |
| `ch06-training-loop` | 第 6 章 | CPU autodiff 训练循环、loss 下降 |
| `ch06-parallel-strategies` | 第 6 章 | DP/TP/PP/ZeRO 的整数流量与空泡表 |
| `ch07-ptq-calibration` | 第 7 章 | PTQ 校准交易：min-max vs 百分位、per-channel、int8 GEMM |
| `ch07-record-roundtrip` | 第 7 章 | ModuleRecord/Burnpack 参数往返保存与恢复 |
| `ch07-serving-queue-sim` | 第 7 章 | 静态批 vs 连续批、TTFT/TPOT、分块 prefill 与 KV 预算 |
| `ch08-rl-rollout` | 第 8 章 | 确定性环境 rollout、replay、表格 TD 更新 |
| `ch09-cluster-simulator` | 第 9 章 | 集群调度与故障的确定性虚拟时间模拟 |
| `ch05-ch07-capstone` | [综合实验](capstone.md) | Dataset → 训练 → Record → 推理的端到端路径 |
| `ch06-parallel-strategies` + `ch07-serving-queue-sim` | [训练与服务成本](capstone-infra.md) | 切分流量 / 空泡与排队 TTFT 合读 |
| `ch02-ch04-op-anatomy` | [算子解剖](op-anatomy.md) | tanh 前向/反向/组合，以及 sum/mean 归约反向 |

## 运行某个示例

每个示例都可以运行测试或直接运行主程序观察输出：

```bash
cargo test -p ch06-training-loop --locked
cargo run  -p ch06-training-loop --locked
```

各章“实验”小节会给出该章示例的具体命令和输出解读。示例关注 shape、
数值误差、守恒性等语义字段，不把墙钟时间当成性能结论。

## 阅读上游源码

示例构建不依赖本地源码镜像。若你想打开各章练习中列出的
`burn/...`、`cubecl/...`、`openmlsys/...` 文件，请按根目录
`pins.toml` 中的 URL 和 revision 检出对应仓库，例如：

```bash
git clone https://github.com/tracel-ai/burn.git
git -C burn checkout 976aa9c5ec1d2dd3412710f99759e3c44bdff03d
```

这些 checkout 只用于阅读；不要改成 Cargo `path` 依赖。在线对照时也应
把 URL 中的分支名替换为同一 revision，避免看到已经变化的 `main`。

## 可选跑通（有环境再跑）

主线正文已同步讲解 GPU 拓扑、CubeCL 多 Runtime、部署 Device 选择与
集合通信数据面；**默认示例仍只要求 CPU**。有额外硬件或要对齐 ONNX 时，
可按下面几条路径自行尝试（更细的说明见仓库 `docs/OPTIONAL_PROFILES.md`）：

| 路径 | 何时用 | 入口命令（摘要） |
|---|---|---|
| `wgpu` | 有图形驱动，巩固第 3 章同一 Kernel | `cargo test -p ch03-cubecl-kernel --features wgpu --locked` |
| `wgpu`（GEMM 阶梯） | 想实测共享内存 tile 的差距 | `cargo run -p ch03-gemm-ladder --features wgpu --release --locked` |
| ONNX 对照 | 阅读 burn-onnx 边界（依赖版本与本书示例不同） | 独立环境；不要混进本书默认示例依赖 |
| CUDA / 集合通信 | 本机驱动与固定源码均允许时 | 先读第 3/6/9 章源码入口，再自建实验 |

`wgpu` 最小命令：

```bash
cargo run  -p ch03-cubecl-kernel --features wgpu --locked
cargo test -p ch03-cubecl-kernel --features wgpu --locked
```

没有对应环境就跳过，不影响九章主线。可选路径若只证明语义或可加载性，
请单独记下设备和软件版本，不要写成默认示例的性能结论。

## 本地构建本书站点

```bash
mdbook build book   # 产物在 book/book/，不提交 Git
mdbook serve book   # 本地预览
```

浏览器中的公式由 MathJax CDN 渲染，离线打开时公式可能不渲染；Cargo
依赖的离线可复现不代表浏览器资源离线可用。

## 想一次跑通全书默认示例时

按章学习通常不必运行全仓库检查：每个示例自己的
`cargo test -p <名称> --locked` 就够了。如果你修改了示例代码、想确认
全书仍然自洽，仓库根目录提供了 `make check`：它会构建本书、检查格式、
执行各章默认 CPU 示例，并核对依赖版本与正文一致。
