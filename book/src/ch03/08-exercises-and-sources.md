# 练习、延伸阅读与来源

## 小结

加速器性能来自并行执行与数据复用，而不是单一峰值数字。GPU 的 cube/unit/
plane 拓扑和寄存器/共享/全局存储层次共同约束 Kernel。CubeCL 用 Rust
风格 DSL、IR 与 Runtime 表达这套模型；raw buffer 和 unchecked launch
仍有必须显式证明的 unsafe 边界。

CubeK 以 Blueprint–Routine 架构和 Autotuner 组织算子，并在具体 crate
中提供 Strategy 与 Launch 接口。burn-cubecl 把它与纯 CubeCL Kernel、
fallback 和 LocalTuner 组合起来。
高性能策略受 shape、dtype 与设备能力约束，不是所有 Burn 算子都经过
CubeK。

## 练习

### 概念题

1. 为什么峰值 FLOP/s 不能预测逐元素加法的性能？
2. 比较 Cube、Unit 和 Plane 的协作范围。为什么 Plane 宽度不应写死？
3. Tiling 如何减少全局内存读取？它又会增加哪些资源与同步成本？
4. 为什么 checked launch 不能证明 raw BufferArg 的长度真实？
5. fallback 在可移植算子库中承担什么职责？列出三种触发原因。
6. 为什么 autotune 结果不能直接复制到另一台机器？

### Rust 与 Kernel 题

1. 将实验改为 `output[i] = input[i] * scale + bias`，先扩展 host reference，
   再扩展 Kernel 和测试。
2. 把输入长度改为 5，但将 CubeDim 向上取整到 8，验证 guard 仍使输出正确。
3. 将 `scale` 从 comptime 整数改为运行时标量。记录生成 launch API 的差异，
   并核对固定 CubeCL 源码中的 scalar argument 表示。
4. 为 raw buffer 长度不变量写一个 safe host helper，使 unsafe block 更小。
5. 增加 `Vector<f32, N>` 版本；验证长度不是 N 倍数时的尾部处理。
6. 在支持的机器上增加 WGPU feature，用相同输入比较 CPU 与 WGPU 正确性。

### 源码题

1. 在 CubeCL 中找到 `ABSOLUTE_POS` 的定义，解释它如何由 cube 和 unit
   坐标得到。
2. 找到 `BufferArg::from_raw_parts` 的 Safety 文档，列出调用者责任。
3. 沿 Burn matmul 路径找到 `cubek::matmul::launch::launch_ref`。
4. 在 CubeK matmul 中比较 Naive、CpuGemm 与一个 CMMA Strategy 的可用条件。
5. 找到 attention 因 bias、softcap 或 scale 进入 fallback 的判断。
6. 找出一个 Burn 直接实现的 CubeCL Kernel 和一个经 CubeK 实现的算子，
   比较两条路径。

### 性能实验题

1. 为不同长度的 scale Kernel 分别报告首次运行与稳态运行时间，解释差异。
2. 在 GPU 上比较连续索引和跨距索引，记录有效带宽而不只记录耗时。
3. 为三个 GEMM shape 比较 Burn 默认策略和固定策略。报告 autotune 是否
   包含在计时内，不以单一 shape 宣布普遍胜负。

## 延伸阅读

固定上游中的权威入口：

- `cubecl/README.md`
- `cubecl/examples/gelu/`
- `cubecl/crates/cubecl-core/src/frontend/topology.rs`
- `cubecl/crates/cubecl-core/src/frontend/container/slice/launch.rs`
- `cubecl/crates/cubecl-runtime/src/runtime.rs`
- `cubecl/crates/cubecl-runtime/src/config/autotune.rs`
- `cubek/GUIDE.md`
- `cubek/crates/cubek-matmul/src/`
- `cubek/crates/cubek-reduce/src/`
- `burn/crates/burn-cubecl/src/kernel/matmul/`
- `burn/crates/burn-cubecl/src/kernel/attention/`
- `burn/crates/burn-cubecl/src/ops/module.rs`

CUDA、Triton、CUTLASS、TVM 与 MLIR 文档适合做生态对照；使用时应记录版本，
不能用最新在线签名替代本项目固定源码事实。

## 来源与改编说明

本章改编并重组 OpenMLSys v1 `chapter_accelerator/`：

- `index.md`
- `accelerator_introduction.md`
- `accelerator_architecture.md`
- `accelerator_programming.md`
- `accelerator_practise.md`
- `summary.md`

保留了加速器设计、GPU 存储层次、三级编程抽象、GEMM 公式以及 tiling、
向量化、共享内存和流水线的通用思想。Volta、Ascend、cuBLAS、WMMA、PTX、
TBE/AKG 与 RTX 3080 性能结果被压缩为历史或生态边界。

本章没有复制 OpenMLSys CUDA C++ 示例、`openmlsys-cuda` 代码或缺失图片；
全部实验改为固定 CubeCL revision 上的原创 Rust 代码。新增 CubeCL Runtime、
unsafe 合约、CubeK 分层、Burn 集成、fallback 和 autotune 内容。

OpenMLSys v2 固定快照只列出 GPU/CUDA/Triton/CUTLASS TODO，没有可迁移
正文。完整逐文件与源码事实映射见 `planning/chapter-sources/ch03.md`。
OpenMLSys 原作和改编正文采用 CC BY-NC-SA 4.0，原创 Rust 示例采用
MIT OR Apache-2.0。

