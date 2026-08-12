# 第 4 章 AI 编译器与运行时系统

第 3 章从单个 Kernel 和算子库观察设备执行。本章转向 Kernel 之上的系统：
怎样暂存一串 Tensor 操作，寻找可融合子图，再把 CubeCL IR 编译并提交到
设备。编译器负责保持语义的变换，运行时负责把计划落实到具体资源和时序。

## 本章问题

系统如何在不改变模型语义的前提下变换计算、生成 Kernel，并管理设备上的
资源与执行？动态图、自动微分 tape、Fusion IR、CubeCL IR 和设备 graph
为什么必须分层理解？

## 学习目标

完成本章后，你应该能够：

1. 解释 IR、Pass、lowering、code generation 与 Runtime 的职责；
2. 比较线性 IR、图 IR 和混合表示，并说明机器学习 IR 需要携带的信息；
3. 区分 Rust 类型、运行时 Tensor 元数据与编译器分析结果；
4. 描述 Tensor 操作如何注册为 Burn OperationIr 并进入 Fusion stream；
5. 解释融合为何减少中间读写，以及同步如何切断延迟片段；
6. 沿 CubeCL Scope、KernelDefinition、Compiler、JIT 和缓存追踪 Kernel；
7. 用生命周期、TensorStatus 和 HandleContainer 解释安全复用条件；
8. 使用测试观测 API验证融合计划，并避免把它误作硬件 launch 计数。

## 先修知识

建议先完成第 2 章的计算图/自动微分和第 3 章的 CubeCL Kernel。了解基本
数据流图、所有权与缓存局部性即可；不要求先学 LLVM。

## 本章路线

我们先建立框架无关的 IR 与 Pass 模型，再沿第 1 章系统分层下行：

```text
Tensor 操作
  → burn-ir / burn-fusion（Burn IR / Fusion 计划）
  → burn-cubecl-fusion / CubeK 或回退
  → CubeCL Scope → KernelDefinition → Compiler
  → 设备 Runtime（allocate / schedule / launch → read/sync）
```

这与第 2 章的三种表示一致：本章处理 Fusion 计划与 CubeCL IR，不把它们
写成 autodiff tape 或 device graph capture。相对第 3 章多出来的一层是：
**同一 IR 如何经选择、内存与 JIT 落到不同 Runtime**（含 GPU/图形栈）。
随后讨论 Kernel 选择、stream 与异步边界。默认实验用 FusionInspector
在 CPU Fusion 路径上看计划切分；有 GPU 环境时，阅读重点转向 launch/
同步代价，而不是把 CPU 计划数当成硬件计数器。

## 小节

1. [编译栈与中间表示](ch04/01-stack-and-ir.md)
2. [静态信息、Pass 与自动微分边界](ch04/02-static-analysis-and-passes.md)
3. [Burn IR 与运行时融合](ch04/03-burn-ir-and-fusion.md)
4. [图优化、Kernel 选择与回退](ch04/04-graph-and-kernel-selection.md)
5. [CubeCL Lowering、JIT 与缓存](ch04/05-cubecl-lowering-and-jit.md)
6. [内存、Stream 与异步执行](ch04/06-memory-streams-execution.md)
7. [实验：观察 Fusion 执行计划](ch04/07-fusion-inspector-lab.md)
8. [练习、延伸阅读与来源](ch04/08-exercises-and-sources.md)

第 5 章会把问题移回设备之前：样本读取、变换与 batching 不进入本章的
Fusion 图，却会通过吞吐率与缓冲决定设备是否持续有工作可做。
