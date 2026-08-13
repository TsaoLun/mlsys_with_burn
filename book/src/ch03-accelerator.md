# 第 3 章 AI 加速器与编程

第 2 章停在运行时 Backend 边界：Tensor 运算最终会被派发到某个设备。
本章继续向下，研究设备为什么快、Kernel 如何描述并行工作，以及 Burn 如何
在手写 CubeCL Kernel 与 CubeK 高性能算子之间选择。

## 本章问题

机器学习为何适合大规模并行硬件？线程和数据应该怎样映射到计算阵列与存储
层次？设备无关的 Kernel 语言如何保留性能所需的硬件信息？高层
`Tensor::matmul` 又怎样到达可调优的 CubeK 实现？

## 学习目标

完成本章后，你应该能够：

1. 用并行度、算术强度和数据复用解释加速器的机会与限制；
2. 区分 host、device、cube、unit 与 plane，并映射常见 GPU 术语；
3. 解释寄存器、共享内存与全局内存之间的数据移动代价；
4. 阅读 `#[cube]` Kernel，并说明 launch 拓扑、边界检查和 raw buffer
   的安全责任；
5. 区分 CubeCL 编程语言/运行时、CubeK 算子库与 `burn-cubecl` bridge，
   并能沿六个决策点走查一次 `Tensor::matmul` 到 CubeK Strategy 的路径；
6. 用 GEMM 解释 tiling、向量化、共享内存和流水线；
7. 说明 autotune 为何依赖 shape、dtype、设备与运行时状态；
8. 在 CPU runtime 上运行并测试一个真实 CubeCL Kernel，用 host 加载
   模型理解 tiling 为何减少全局读；有图形驱动时，再用可选 GEMM
   阶梯实验实测共享内存 tile 的收益。

## 先修知识

建议先完成第 2 章，并了解矩阵乘法、缓存局部性和 Rust `unsafe` 的基本含义。
本章不要求 CUDA 经验；CUDA 名词只用于帮助读者对照已有资料。

## 本章路线

我们先按 **GPU 机器模型**建立并行与存储层次，再进入 CubeCL / CubeK。
正文同步对照 `CpuRuntime`、`WgpuRuntime`、`CudaRuntime`、`HipRuntime`
等源码入口：同一套 Kernel IR 如何接到不同设备。默认实验仍在 CPU（可选
`--features wgpu`）上验证语义，不要求本机安装 CUDA；有图形驱动的读者
还可以用可选的 GEMM 阶梯实验，把「朴素 → 共享内存 tile」的差距在
自己机器上实测出来。GEMM 把硬件、Kernel 与算子库连起来；更深的
Fusion / JIT / stream 留到第 4 章。

## 小节

1. [工作负载与加速器设计](ch03/01-workloads-and-design.md)
2. [GPU 并行与存储模型](ch03/02-gpu-machine-model.md)
3. [CubeCL 编程模型](ch03/03-cubecl-programming.md)
4. [CubeK 与 Burn 算子路径](ch03/04-cubek-and-burn.md)
5. [GEMM 与优化阶梯](ch03/05-gemm-optimization.md)
6. [算子编译、调优与生态](ch03/06-compilation-and-tuning.md)
7. [实验：CPU 上运行 CubeCL Kernel](ch03/07-cpu-kernel-lab.md)
8. [练习、延伸阅读与来源](ch03/08-exercises-and-sources.md)

读完 Kernel 与 GEMM 后，第 4 章会把“单个正确 Kernel”推进到一串
Tensor 操作的融合、lowering、JIT 与运行时资源管理。
