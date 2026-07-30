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

我们先建立框架无关的 IR 与 Pass 模型，再沿两层实现下行：

```text
Tensor 操作 → burn-ir / burn-fusion → 优化执行块
          → burn-cubecl-fusion / CubeK 或回退
          → CubeCL Scope → KernelDefinition → Compiler → Runtime
```

随后讨论 Kernel 选择、内存生命周期、stream 和异步边界。实验使用
FusionInspector 观察计划结构和同步切分；它是固定快照中的测试 API，不是
生产诊断接口。AOT、设备 graph 与性能测量只建立边界，不超出已核验能力。

