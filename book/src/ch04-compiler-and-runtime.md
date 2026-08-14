# 第 4 章 AI 编译器与运行时系统

第 3 章从单个 Kernel 观察设备执行。本章转向 Kernel 之上：怎样暂存一串
张量操作、寻找可融合子图，再把 IR 编译并提交到设备。编译器做保持语义
的变换；运行时把计划落实到内存、流和时序。

这是 OpenMLSys「前端与 IR」「后端与运行时」两章的合并。产业里对应
XLA、TVM、nvFuser、PyTorch 2 compile 那一层。autodiff tape（第 2 章）
记录的是反向依赖，和这里的 Fusion IR、CubeCL IR 不是同一张图。

## 本章问题

系统如何在不改变模型语义的前提下变换计算、生成 Kernel，并管理设备上的
资源与执行？动态图、自动微分 tape、Fusion IR、CubeCL IR 和设备 graph
为什么必须分层理解？

## 学习目标

完成本章后，你应该能够：

1. 解释 IR、Pass、lowering、code generation 与 Runtime 的职责；
2. 比较线性 IR、图 IR 和混合表示，并说明机器学习 IR 需要携带的信息；
3. 区分 Rust 类型、运行时张量元数据与编译器分析结果；
4. 描述张量操作如何注册为 OperationIr 并进入 Fusion stream；
5. 解释融合为何减少中间读写，以及同步如何切断延迟片段；
6. 沿 CubeCL Scope、KernelDefinition、Compiler、JIT 和缓存追踪 Kernel；
7. 用生命周期、TensorStatus 和 HandleContainer 解释安全复用条件；
8. 观察融合计划的切分，而不把它当成硬件 launch 计数。

想改融合规则或 JIT 缓存键时，见
[一次调用会经过哪些层](crate-map.md)。

## 先修知识

建议先完成第 2 章的计算图/自动微分和第 3 章的 CubeCL Kernel。了解基本
数据流图与缓存局部性即可。

## 本章路线

先建立框架无关的 IR 与 Pass 模型，再沿第 1 章分层下行：

```text
Tensor 操作
  → burn-ir / burn-fusion
  → burn-cubecl-fusion / CubeK 或回退
  → CubeCL Scope → KernelDefinition → Compiler
  → 设备 Runtime（allocate / schedule / launch → read/sync）
```

默认实验用 FusionInspector 看 CPU Fusion 路径上的计划切分；有 GPU 时，
阅读重点转向 launch 与同步代价。迷你 Pass 实验让你亲手写常量折叠 /
DCE / CSE，并看到非法 fast-math 会破坏什么。

## 小节

1. [编译栈与中间表示](ch04/01-stack-and-ir.md)
2. [静态信息、Pass 与自动微分边界](ch04/02-static-analysis-and-passes.md)
3. [Burn IR 与运行时融合](ch04/03-burn-ir-and-fusion.md)
4. [图优化、Kernel 选择与回退](ch04/04-graph-and-kernel-selection.md)
5. [CubeCL Lowering、JIT 与缓存](ch04/05-cubecl-lowering-and-jit.md)
6. [内存、Stream 与异步执行](ch04/06-memory-streams-execution.md)
7. [实验：观察 Fusion 执行计划](ch04/07-fusion-inspector-lab.md)
8. [练习、延伸阅读与来源](ch04/08-exercises-and-sources.md)

第 5 章回到设备之前：样本读取与 batching 不进入本章的 Fusion 图，却
通过吞吐和缓冲决定设备是否一直有工作可做。
