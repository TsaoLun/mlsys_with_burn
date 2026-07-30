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

1. 用并行度、计算强度和数据复用解释加速器的机会与限制；
2. 区分 host、device、cube、unit 与 plane，并映射常见 GPU 术语；
3. 解释寄存器、共享内存与全局内存之间的数据移动代价；
4. 阅读 `#[cube]` Kernel，并说明 launch 拓扑、边界检查和 raw buffer
   的安全责任；
5. 区分 CubeCL 编程语言/运行时、CubeK 算子库与 `burn-cubecl` bridge；
6. 用 GEMM 解释 tiling、向量化、共享内存和流水线；
7. 说明 autotune 为何依赖 shape、dtype、设备与运行时状态；
8. 在 CPU runtime 上运行并测试一个真实 CubeCL Kernel。

## 先修知识

建议先完成第 2 章，并了解矩阵乘法、缓存局部性和 Rust `unsafe` 的基本含义。
本章不要求 CUDA 经验；CUDA 名词只用于帮助读者对照已有资料。

## 本章路线

我们先建立框架无关的加速器和 GPU 机器模型，再进入 CubeCL 与 CubeK。
GEMM 把硬件、Kernel 和算子库三层连接起来。最后在 CPU 上运行同一套
CubeCL IR；这能验证语义和边界，但不被当作 GPU 性能实验。更深入的 IR、
融合和运行时系统留到第 4 章。

