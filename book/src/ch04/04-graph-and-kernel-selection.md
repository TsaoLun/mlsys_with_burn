# 图优化、Kernel 选择与回退

Fusion 找到等价执行块之后，仍要回答“用哪个 Kernel 执行”。图优化和
Kernel 选择相关，却不是同一步。

## 1. 子图替换

图优化可把满足条件的子图替换为语义等价实现：

```text
多个 Tensor 操作
    ↓ pattern + legality
融合 Kernel、库算子或新的操作序列
```

合法性至少依赖：

- 操作的数据依赖和副作用；
- shape/broadcast；
- dtype 与精度策略；
- layout/stride；
- 中间值是否还有外部使用者；
- 目标 Backend 的实现能力。

OpenMLSys 以 Conv+ReLU、Graph Kernel 和 layout 转换消除说明减少访存的
价值。Burn 对应案例包括元素级 trace、matmul/reduce 前后融合和 NHWC
relayout fuser，但二者实现不能逐项机械对应。

## 2. Layout 是语义和成本的交界

同一逻辑 Tensor 可以有不同物理布局。以四维图像为例，NCHW 与 NHWC 的
线性偏移不同；相邻逻辑维度是否连续会影响向量化与合并访存。

编译器必须区分：

- 只改变 view/stride 的零拷贝重解释；
- 必须搬运数据的真实 relayout；
- Kernel 原生支持的布局；
- 为使用某个库算子而插入的转换。

若一次高性能 Kernel 节省的时间小于前后 layout 转换，整体计划反而更慢。
因此 Kernel 选择不能只看算子本体 benchmark。

## 3. Dtype 与数值语义

低精度可减少存储和提高矩阵吞吐，但可能改变：

- 舍入误差；
- 累加精度；
- 溢出/下溢范围；
- 哪些硬件指令可用；
- Kernel 输入输出转换成本。

候选选择必须检查 dtype 支持，不能静默把 f32 降为 f16。TF32、混合精度或
量化路径应明确累加和输出类型。第 6 章会从训练稳定性继续讨论。

## 4. 候选过滤、选择与调优

一个健壮的流程通常是：

1. 根据 op、shape、layout、dtype 生成候选；
2. 用 Runtime/device properties 过滤不可能的策略；
3. 应用静态启发式或历史缓存；
4. 必要时运行 autotune；
5. 缓存 tune key 到选择结果；
6. 无候选时执行正确的 fallback。

第 3 章已经看到 CubeK Strategy。Fusion 在更高层决定哪些操作进入同一
优化；CubeK/CubeCL 在更低层决定 Kernel 结构与 launch。两个搜索空间相关，
但不要把 Fusion block 数直接当作 CubeK 候选数。

## 5. 回退与组合计划

回退保证能力不完整时仍可执行。它可能发生在：

- fuser 无法接收某个操作；
- Kernel 对当前 shape/dtype 不可用；
- autotune 候选失败；
- 动态条件超出已缓存计划；
- 设备资源不足。

Burn 可以把 fused optimization 与 unfused 操作组合为一个执行策略。
性能报告必须说明是否发生 fallback；只验证最终数值，无法证明预期优化路径
真的被使用。本章实验使用 Inspector 补上这一层结构证据。

## 6. 从 Fusion 计划进入 CubeCL

计划执行时，burn-cubecl 的 `CubeOptimization` 分派到具体 optimization：
ElementWise 路径把 trace 生成为 CubeCL Kernel；Matmul/Reduce 路径组合
对应的 CubeK launch 与前后 trace；无法纳入的位置通过 FallbackOperation
调用底层操作。由此产生的 CubeCL KernelDefinition 才进入下一节的优化、
lowering 和 Runtime launch。Fusion block 是上层计划，不能直接等同于一份
KernelDefinition。

