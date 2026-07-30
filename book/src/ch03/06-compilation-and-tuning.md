# 算子编译、调优与生态

同一算法面对不同 shape、dtype 和设备时，最佳 tile、向量宽度和流水线可能
不同。手工为每个组合选择实现不可扩展，算子编译和自动调优由此产生。

## 1. 编译期特化

CubeCL 的 `#[comptime]` 与 CubeK Blueprint 可以把策略决策变成编译期
常量，消除无效分支并生成设备指令。代价是更多 Kernel 变体、编译延迟和
缓存占用。若把运行时 shape 的每个细节都纳入特化键，会形成 Kernel
explosion；若特化太少，又可能保留低效的动态逻辑。

第 4 章会进一步区分：

- 用户 Tensor 程序；
- Burn Fusion IR；
- CubeCL Kernel IR；
- Runtime 编译产物与设备 graph。

本章只需记住：Kernel 不是直接把 Rust 源码交给 GPU，而是经过宏展开、IR
构建、Runtime 编译和 launch。

## 2. Autotune 不是静态规则表

固定快照中，CubeCL 提供全局 autotune level 与缓存配置；burn-cubecl
使用 `LocalTuner` 为 matmul、conv、reduce 和 attention 等注册候选。
典型流程是：

```text
输入 shape/layout/dtype + Device 能力
    ↓ 过滤不支持的策略
候选 Kernel 集合
    ↓ 在当前设备测量
选择较优候选
    ↓ 按 tune key 缓存
后续相似调用复用
```

测量结果依赖设备、驱动、温度、系统负载和 Runtime 版本。持久缓存可以减少
首次成本，却不能让结果跨机器天然有效。benchmark 也必须在真实同步后停止
计时。

Autotune level 控制搜索成本和候选广度，不等于“越高永远越快”。短生命周期
进程可能还没收回搜索成本；服务系统则可能通过预热摊销。

## 3. 从 TVM/MLIR 到 CubeCL

OpenMLSys 介绍 TVM、Ansor、MLIR、TBE 和 AKG，核心问题仍然成立：

- 如何表示高层算子与低层循环/并行结构；
- 如何把算法与 schedule 分离；
- 如何搜索 tile、向量化、内存层次与并行映射；
- 如何在多种硬件上 lowering 并验证正确性。

本项目选择 CubeCL/CubeK 作为连续实现栈，不表示其他系统已过时。TVM 与
MLIR 的深入比较放到第 4 章；厂商专用 TBE/AKG 只作为生态历史边界。

## 4. CUDA、Triton 与 CUTLASS 对照

OpenMLSys v2 将 CUDA、Triton 和 CUTLASS 列入第 3 章，但固定 v2 快照没有
正文。本书只建立概念坐标：

- **CUDA C++** 直接暴露 NVIDIA 线程、内存和指令生态；
- **Triton** 用 Python DSL 以 tile/block program 表达 Kernel；
- **CUTLASS** 用 C++ 模板和组件组织 NVIDIA GEMM/conv；
- **CubeCL** 用 Rust 风格 DSL 和 Runtime 面向多个后端；
- **CubeK** 在 CubeCL 上组织高性能算子与策略。

它们解决的问题有重叠，但类型系统、后端范围、成熟度和性能路径不同。不能
仅凭 API 相似就宣称语义或性能等价。本书不引入前三者为构建依赖。

## 5. 测试、Benchmark 与性能声明

CubeK 使用 host reference、近似数值断言、CPU 测试和可选 GPU benchmark。
CPU CI 能发现大量索引、布局和数值错误，却不能覆盖 CMMA/TMA 等硬件路径。
一个可信性能声明还应满足：

1. correctness test 与 benchmark 分离；
2. 结果包含环境和固定 commit；
3. 首次编译/autotune 与稳态延迟分别报告；
4. 多个代表 shape，而不是只挑一个获胜输入；
5. 不支持或 fallback 情况被显式记录。

