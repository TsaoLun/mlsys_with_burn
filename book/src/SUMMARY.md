# Summary

[首页](README.md)
[前言](preface.md)
[许可、来源与独立性声明](attribution.md)

# 基础篇

- [第 1 章 导论](ch01-introduction.md)
  - [机器学习应用与系统负载](ch01/01-applications-and-loads.md)
  - [机器学习系统的设计目标](ch01/02-design-goals.md)
  - [从编程接口到硬件](ch01/03-system-architecture.md)
  - [Burn 技术栈](ch01/04-burn-stack.md)
  - [生命周期、生态与阅读路径](ch01/05-lifecycle-and-ecosystem.md)
  - [实验：探测执行栈](ch01/06-stack-probe-lab.md)
  - [练习、延伸阅读与来源](ch01/07-exercises-and-sources.md)
- [第 2 章 编程接口与计算图](ch02-programming-and-graph.md)
  - [从工作流到编程接口](ch02/01-interface-and-workflow.md)
  - [Tensor、Device 与运行时后端](ch02/02-tensor-device-backend.md)
  - [Module、参数与模型状态](ch02/03-module-and-state.md)
  - [计算图的构成与生成](ch02/04-computational-graph.md)
  - [自动微分](ch02/05-autodiff.md)
  - [类型、IR 与调度边界](ch02/06-types-ir-scheduling.md)
  - [实验：张量、Module 与梯度](ch02/07-labs.md)
  - [练习、延伸阅读与来源](ch02/08-exercises-and-sources.md)
- [第 3 章 AI 加速器与编程](ch03-accelerator.md)
  - [工作负载与加速器设计](ch03/01-workloads-and-design.md)
  - [GPU 并行与存储模型](ch03/02-gpu-machine-model.md)
  - [CubeCL 编程模型](ch03/03-cubecl-programming.md)
  - [CubeK 与 Burn 算子路径](ch03/04-cubek-and-burn.md)
  - [GEMM 与优化阶梯](ch03/05-gemm-optimization.md)
  - [算子编译、调优与生态](ch03/06-compilation-and-tuning.md)
  - [实验：CPU 上运行 CubeCL Kernel](ch03/07-cpu-kernel-lab.md)
  - [练习、延伸阅读与来源](ch03/08-exercises-and-sources.md)

# 系统篇

- [第 4 章 AI 编译器与运行时系统](ch04-compiler-and-runtime.md)
- [第 5 章 数据处理系统](ch05-data-processing.md)
- [第 6 章 训练系统](ch06-training-systems.md)

# 应用与扩展篇

- [第 7 章 模型服务](ch07-model-serving.md)
- [第 8 章 强化学习系统](ch08-rl-systems.md)
- [第 9 章 大规模 GPU 集群管理](ch09-gpu-cluster.md)

