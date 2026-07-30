# 2026-07-31：第 3 章 AI 加速器与编程

## 会话目标

在已完成第 1、2 章和远程 pin 策略的基础上继续推进计划，完成第 3 章，
形成从 Burn Tensor API 到 CubeCL Kernel/CubeK 算子库的基础篇闭环。

## 开始状态

- Git：`main` 位于 `23f5795`，工作区干净。
- 当前里程碑：M2 基础篇。
- 固定快照：
  - Burn `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`
  - CubeCL `be278a1e76aed881e2cc6b165414ee6103ca4634`
  - CubeK `f82a6d07ebf35a1d446893b32712458744d80f13`
  - OpenMLSys `9c289782ccbb165ac8ad7c960ecffc12942a5560`

## 研究与判断

### OpenMLSys 映射

逐文件检查 v1 `chapter_accelerator/` 六个文件。保留：

- 加速器通用性/能效/可编程性取舍；
- GPU 并行和存储层次；
- 库、Kernel 原语、指令三级抽象；
- GEMM、计算强度、tiling、向量化、共享内存与流水线。

降低或删除：

- Volta、Ascend、RTX 3080 等设备/版本特定数字；
- cuBLAS、WMMA、PTX、TBE/AKG 厂商路径的主线地位；
- “手写 Kernel 胜过库”的单设备结论；
- 不在固定 clone 中的 `_cn.png`、`6.4/*.png`。

OpenMLSys v2 只有 GPU/CUDA/Triton/CUTLASS TODO，没有可迁移正文。Triton
与 CUTLASS 只进入生态对照。

### CubeCL 核验

以固定源码和 `examples/gelu` 为锚点核验：

- `#[cube]`、`launch_unchecked`、`#[comptime]`；
- `CubeCount`、`CubeDim`、`ABSOLUTE_POS`；
- Runtime、ComputeClient、BufferArg；
- CPU Runtime 和 WGPU Runtime；
- 当前向量类型是 `Vector<F, N>`，不是旧 `Line`；
- `BufferArg::from_raw_parts` 与 unchecked launch 的 unsafe 合约。

选择 scale Kernel 而不是完整 GEMM 作为必做实验，原因是它能最小化暴露
macro、拓扑、buffer、launch 和读回边界；GEMM 优化阶梯在正文解释，复杂
CubeK/GPU benchmark 留为扩展。

### CubeK/Burn 边界

核验 CubeK Guide、matmul/conv/reduce/attention 源码与 burn-cubecl 调用：

- Guide 的核心是 Blueprint–Routine 架构与 Autotuner；
- 具体算子 crate 另有 Strategy 和 Launch API；
- Burn 的部分 matmul、implicit-GEMM conv、reduce、attention forward 等
  经 CubeK；
- 大量 elementwise、direct conv 与 fallback 仍是 burn-cubecl 自定义
  CubeCL 路径；
- 不宣称 FlashAttention backward 已接入 Burn；
- 不宣称 CMMA/TMA 或 autotune 结果跨设备可用。

## 实现

### 教材

将第 3 章扩展为八节：

1. 工作负载与加速器设计；
2. GPU 并行与存储模型；
3. CubeCL 编程模型；
4. CubeK 与 Burn 算子路径；
5. GEMM 与优化阶梯；
6. 算子编译、调优与生态；
7. CPU CubeCL Kernel 实验；
8. 练习、延伸阅读与来源。

更新 `book/src/SUMMARY.md` 和章导言，并建立
`planning/chapter-sources/ch03.md`。

### 示例

新增 `examples/ch03-cubecl-kernel`：

- 默认直接依赖固定 Git revision 的 CubeCL CPU feature；
- `scale_kernel` 用 `ABSOLUTE_POS` 和显式 guard；
- host 使用设备属性驱动的 `CubeDim::new` 和多 cube 覆盖；
- 空输入在 launch 前返回；
- CubeCount 使用 checked conversion；
- unsafe block 只覆盖 raw BufferArg 与 unchecked launch；
- CPU 测试与普通 Rust reference 比较；
- 可选 `wgpu` feature 在同一 Kernel 上复用正确性测试。

该 crate 有意不继承 workspace 的 `unsafe_code = "forbid"`，而在自身
manifest 中设置 `unsafe_code = "allow"`。例外被限制在有 Safety 注释的
Runtime 边界，没有全局放宽。

## 发现的问题与修复

1. 首次编译时使用 `#[comptime] scale: f32`，宏要求特化值参与 hash，
   `f32` 不满足，报 `no method named hash`。改为 `u32` comptime scale，
   Kernel 内转换为浮点。
2. 修改 API 后 `main.rs` 仍传 `2.0`，产生类型错误；改为 `2`。
3. 初版 launch 把全部元素放进一个 cube，无法扩展到设备 unit 上限，也
   没处理空输入和 `usize as u32` 截断。改为 `CubeDim::new`、多 cube、
   空输入早退和 checked conversion。
4. 初版正文把 CubeK Guide 错写为四层架构。重新读取 Guide 后修正为
   Blueprint–Routine–Autotuner，并把 Strategy/Launch 明确归于具体源码。
5. 初版实验称输入长度只改变运行参数；实际上本例让长度参与 CubeDim
   选择，而 CubeDim 属于编译键。正文已修正。
6. 来源映射初版省略 `crates/` 路径层级，已改为从各镜像根开始的精确路径。

## 验证证据

成功运行：

```text
cargo test -p ch03-cubecl-kernel
cargo run -p ch03-cubecl-kernel
cargo test -p ch03-cubecl-kernel --features wgpu
make check
make check-local-sources
git diff --check
```

观察结果：

- CPU：2 个测试通过；
- WGPU feature：CPU 与 WGPU 共 3 个测试通过；
- 示例输出 `[1,2,3,4] × 2 = [2,4,6,8]`；
- mdBook、pin 检查、fmt、Clippy `-D warnings`、workspace test 全部通过；
- IDE 未报告新增 lint。

环境持续出现用户级 Cargo 配置提示：`.cargo/config` 与
`.cargo/config.toml` 同时存在；Cargo 选择前者。它不属于仓库改动，也未
影响验证。

## 状态变化与下一步

- M2 基础篇完成；
- `planning/STATUS.md` 转入 M3 系统篇；
- 下一步：映射 OpenMLSys frontend/IR 与 backend/runtime，核验 Burn
  IR/Fusion、CubeCL IR/opt/runtime，并设计第 4 章 CPU 可观察实验。

