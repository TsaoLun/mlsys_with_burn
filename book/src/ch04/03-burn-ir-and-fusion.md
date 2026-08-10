# Burn IR 与运行时融合

## 1. 从即时执行到注册操作

Fusion Backend 下的 Tensor 操作不会立刻调用底层 Backend。以浮点加法为
例，burn-fusion 创建 `BinaryOpIr`，把它包装为 `OperationIr`，然后向当前
Fusion 流标识（`StreamId`：延迟队列的隔离键；不是 CUDA stream，也不是
集群作业队列）对应的 Fusion client 注册。

`OperationIr` 是覆盖多种操作族的大枚举。每个 Tensor 通过 `TensorIr`
携带：

- `TensorId`：Fusion 资源标识；
- shape；
- dtype；
- `TensorStatus`：输出未初始化、只读或可读写等状态。

这些字段是执行和生命周期快照，不是用户 Tensor 的完整 Rust 类型。

## 2. Client、Server 与 Stream

高层链路可简化为：

```text
Tensor op
  → GlobalFusionClient::register
  → FusionServer
  → MultiStream / OperationQueue
  → 搜索可行 Block 与 ExecutionStrategy
  → fused optimization 或 UnfusedOp
  → HandleContainer 注册输出
```

这里的 `ExecutionStrategy` 是 **Fusion 块内**如何执行/融合的搜索结果，
与第 6 章 `burn-train` 的 `ExecutionStrategy`（MultiDevice/DDP 装配）
只是同名，不要当成同一个类型。

Fusion 按 stream 保存队列。固定实现中的 `StreamId::current()` 与注册操作
的线程/任务上下文相关。跨 stream 共享 Tensor 时，系统必须建立别名和顺序
关系，必要时先 drain 来源 stream；否则重排可能在值产生前读取或在使用前
释放。

## 3. 搜索的对象是可执行块

Fusion 搜索跟踪每个 block 的：

- `produced`：在块内产生的资源；
- `read`：块外输入；
- `freed`：最后使用后可释放的资源；
- 操作顺序与候选融合器（fuser：决定一组操作能否合成一块）状态。

源码把相关配置命名为 `BeamSearchConfig`，并限制同时探索的候选 block
数量；这不是对某种通用 beam-search 评分算法的承诺。探索上限和日志级别
由 FusionConfig 控制。运行时可以命中已有计划，也可以探索新计划，或者
逐操作回退。动态系统的优化因此包含状态与成本，不是一次静态图重写。

## 4. CubeCL Backend 的 fuser

固定 burn-cubecl 快照按顺序注册：

- `ElementWiseFuser`；
- `MatmulFuser`；
- `ReduceFuser`；
- `ReduceBroadcastedFuser`；
- `NHWCRelayoutFuser`。

这表示“存在这些优化入口”，不表示任意同类表达式都能完整融合。shape、
broadcast、layout、dtype、设备能力和 fuser 接受状态都会关闭候选。
Matmul/Reduce 优化还可以与不能融合的操作组成 `Composed` 策略并执行
fallback。

## 5. 融合为什么有价值

考虑：

```text
t0 = a + b
out = exp(t0)
```

逐操作执行通常要把 `t0` 写到全局内存，再由第二个 Kernel 读回。对 $N$
个 `f32` 元素定量比较（只数全局内存流量）：

- **不融合**：`add` 读 $8N$ 字节、写 $4N$；`exp` 读 $4N$、写 $4N$；
  合计 $20N$ 字节、2 次 launch、1 份中间 allocation；
- **融合**：单个 Kernel 读 $8N$、写 $4N$，中间值留在寄存器；合计
  $12N$ 字节、1 次 launch、0 份中间 allocation。

流量下降 $40\%$，launch 和分配各少一次。第 3 章已经算过这类逐元素
算子的算术强度只有约 $0.125\ \text{FLOP/字节}$，完全受带宽限制——
因此对它们来说，**减少流量几乎等于减少时间**，这正是元素级融合收益
直接的量化解释。本章 FusionInspector 实验观察的就是这条链上计划
结构的变化。

融合也可能增加寄存器压力、编译时间或产生过大的 Kernel。包含复杂归约或
不兼容布局时，拆分可能更合理。系统需要候选与测量，而不是“融合越多越好”。

## 6. 物化与同步边界

以下操作会要求队列中的值变成真实底层资源：

- `to_data()` / `into_data()` 读回；
- `Device::sync()`；
- 某些跨 stream 共享；
- 后续操作无法继续延迟或优化策略决定执行。

同步位于 `a + b` 与 `exp` 之间时，前一段必须先执行，后来的 `exp` 不能跨
边界与加法形成同一融合块。本章实验直接验证这一点。

## 7. Flex 不是 Fusion CPU

`Device::flex()` 走 burn-flex eager 路径，不生成上述 OperationIr。
`Device::cpu()` 在启用 `cpu` 与 `fusion` feature 后使用 CubeCL CPU 的
Fusion 包装。二者都是“CPU 上能跑”，但默认实验与 Fusion 观察不在同一条
路径上。本章实验必须选择后者；把 Flex 输出正确当作 Fusion 已执行
是错误证据。

