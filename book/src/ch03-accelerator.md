# 第 3 章 AI 加速器与编程

第 2 章停在运行时后端边界：张量运算最终会被派发到某个设备。本章继续
向下：设备为什么快、Kernel 如何描述并行工作，以及高层 `matmul` 怎样
落到可调优的算子实现。

这是 OpenMLSys「AI 加速器」一章的对应物。产业里你会碰到 CUDA 编程模型、
Triton、CUTLASS 和各类 GEMM 库；这里用 CubeCL 作为设备无关的 Kernel
语言，用 CubeK 作为高性能算子库。机器模型（cube / unit / plane、存储
层次）与具体 Runtime 无关，应先建立。

## 本章问题

机器学习为何适合大规模并行硬件？线程和数据应该怎样映射到计算阵列与
存储层次？设备无关的 Kernel 语言如何保留性能所需的信息？高层
`Tensor::matmul` 又怎样到达 CubeK？

## 学习目标

完成本章后，你应该能够：

1. 用并行度、算术强度和数据复用解释加速器的机会与限制；
2. 区分 host、device、cube、unit 与 plane，并映射常见 GPU 术语；
3. 解释寄存器、共享内存与全局内存之间的数据移动代价；
4. 阅读 `#[cube]` Kernel，并说明 launch 拓扑、边界检查和 raw buffer
   的安全责任；
5. 区分 CubeCL 语言/运行时、CubeK 算子库与 `burn-cubecl` 桥，并能沿
   `Tensor::matmul` 走到 Strategy；
6. 用 GEMM 解释 tiling、向量化、共享内存和流水线；
7. 说明 autotune 为何依赖 shape、dtype、设备与运行时状态；
8. 在 CPU runtime 上运行一个 CubeCL Kernel，并与 host 参考实现对照；
   有图形驱动时，可用可选实验看共享内存 tile 的收益。

## 先修知识

建议先完成第 2 章，了解矩阵乘、缓存局部性和 Rust `unsafe`。不要求 CUDA
经验；CUDA 名词只用于对照。

## 本章路线

先按 GPU 机器模型建立并行与存储层次，再进入 CubeCL / CubeK。正文会对照
`CpuRuntime`、`WgpuRuntime`、`CudaRuntime`、`HipRuntime`：同一套 Kernel
IR 如何接到不同设备。默认实验在 CPU 上核对语义；有图形驱动时可以加上
`--features wgpu`。更深的融合、JIT 与 stream 留到第 4 章。

## 小节

1. [工作负载与加速器设计](ch03/01-workloads-and-design.md)
2. [GPU 并行与存储模型](ch03/02-gpu-machine-model.md)
3. [CubeCL 编程模型](ch03/03-cubecl-programming.md)
4. [CubeK 与 Burn 算子路径](ch03/04-cubek-and-burn.md)
5. [GEMM 与优化阶梯](ch03/05-gemm-optimization.md)
6. [算子编译、调优与生态](ch03/06-compilation-and-tuning.md)
7. [实验：CPU 上运行 CubeCL Kernel](ch03/07-cpu-kernel-lab.md)
8. [练习、延伸阅读与来源](ch03/08-exercises-and-sources.md)

读完 Kernel 与 GEMM 后，第 4 章把「单个正确 Kernel」推进到一串操作的
融合、lowering 与运行时资源管理。
