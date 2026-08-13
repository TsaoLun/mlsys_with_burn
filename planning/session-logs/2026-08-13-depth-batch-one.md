# 2026-08-13 深度批次一：GEMM 阶梯实测 + 迷你反向 tape

## 背景与决策

教科书化第一批（D023）解决表达层后，本批针对深度缺口中优先级最高的
两项：全书零真实设备测量（对照原作 CUDA 实验），以及第 2 章缺少
「亲手造一遍」的动手深度。方案 A+B：可选 wgpu 的 GEMM 优化阶梯实验、
纯 Rust 迷你反向 tape。均在 D022（可选 profile 不进默认门禁）与
既有体例内，不需要新决策条目。

## 关键发现：tracel-llvm 不再提供 Intel macOS 资产

`tracel-llvm v22.1.4-5` 的 GitHub release 只有 linux-AArch64/x64、
macos-AArch64、windows-x64 四组资产，**没有 macos-x64**。因此本机
（Intel macOS）凡依赖 `cubecl-cpu` 的构建必然失败——此前 STATUS 记录
的 404 不是缓存或网络问题，而是上游放弃了该平台预编译包。规避设计：
`ch03-gemm-ladder` 默认特性零 CubeCL 依赖；`wgpu` 特性直接声明 pins
同 revision 的 `cubecl`（`std`+`stdlib`，不带 `cpu`），wgpu/WGSL 链
不经过 LLVM。已同步 STATUS 已知问题。

## 交付

1. `examples/ch02-mini-autodiff`（新 crate，无依赖）：
   约百行标量反向 tape，7 项测试（链式法则、扇出累加、relu 截断、
   数值梯度校验、分支只记录执行路径、detach 断流、backward 重置）。
   正文接入 ch02/05（新小节）、ch02/07（新第 9 节 + include 锚点 +
   运行输出，后续小节顺延）、章首路线、练习 Rust 题 8/9、
   running-examples 表、附录范围条目。
2. `examples/ch03-gemm-ladder`（新 crate）：
   默认纯 Rust 分块 GEMM 语义验证（3 测试）；`--features wgpu` 两级
   CubeCL Kernel + 计时协议（5 测试，含 17/33/47/65 非整除形状与
   两级一致性）。本机 Metal `wgpu<wgsl>` release 实测 256/512/1024
   方阵 tiled 加速 4.71/4.22/4.44 倍——全书第一处真实设备测量，
   正文按单机单次观测口径呈现。接入 ch03/05（三类实验表）、ch03/07
   （新第 8 节）、章首、练习扩展题 4、running-examples、
   OPTIONAL_PROFILES 新 profile 段、附录范围条目。
3. 工程：workspace members、Makefile `CPU_EXAMPLES`、Cargo.lock
   （仅新增两个成员包，无新外部依赖）。

## 验证

- `cargo test -p ch02-mini-autodiff --locked --offline`：7 通过；
- `cargo test -p ch03-gemm-ladder --locked --offline`（默认）：3 通过；
- `cargo test -p ch03-gemm-ladder --features wgpu --locked --offline`：
  5 通过（本机 Metal）；
- `cargo run -p ch03-gemm-ladder --features wgpu --release`：正确性
  0 误差，计时表见正文；`cargo run -p ch02-mini-autodiff`：tape 输出
  与正文引用一致；
- 两 crate `cargo clippy --all-targets`（default 与 wgpu）无警告、
  `cargo fmt --check` 通过；
- `mdbook build/test book`、`check_release.py --require-built-book
  --json`（`ok=true`、`errors=[]`、`warnings=[]`）、
  `git diff --check` 通过。

## 边界与偏差

- 计时数字为单机单次观测，正文与 OPTIONAL_PROFILES 均标注口径；
  两级 Kernel 输出与 host reference 一致是所有比较的前置断言。
- 本机无法验证 `ch03-cubecl-kernel` 等依赖 cubecl-cpu 的既有示例
  （上游资产缺失，非本批改动引入）；干净 Linux/CI 环境不受影响。
- 阶梯第 3–5 级（thread tile、双缓冲、矩阵指令）留作练习与后续
  批次；KV cache 模拟、迷你 Pass 流水线等深度候选见 STATUS。

## 交接

推送后建议在有 GPU 的其他平台复跑
`cargo run -p ch03-gemm-ladder --features wgpu --release --locked`，
把不同设备的观测追加到会话日志（不进正文）。下一深度批次候选：
迷你编译器 Pass 流水线（ch04）、PTQ 校准迷你实验（ch07）、
KV cache/连续批处理队列模拟（需扩围决策）。
