# 第 3 章 AI 加速器与编程

## 本章问题

GPU 为什么能加速机器学习负载，设备无关的 Kernel 语言又如何映射到不同
硬件后端？

## 计划内容

- GPU 执行与存储层次
- 并行分解、访存合并与同步
- CubeCL 编程模型和宏展开
- CubeCL 后端：CUDA、ROCm、WGPU、CPU
- CubeK 的矩阵乘、卷积和注意力算子
- 自动调优与性能测量

## 实验

从 CPU 可验证的 Kernel 开始，再按本机能力进入 WGPU 或 CUDA。

## 来源与改编说明

计划参考 OpenMLSys v1 `chapter_accelerator/`。保留硬件基础，使用
CubeCL/CubeK 重写框架和厂商专用算子示例。

