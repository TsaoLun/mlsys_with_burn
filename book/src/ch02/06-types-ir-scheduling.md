# 类型、IR 与调度边界

## 编译期与运行时信息

Rust 类型系统能表达一部分张量约束，但不应把所有 shape 都强行放进类型。
固定快照采用如下分工：

| 信息 | 主要确定时机 | 示例 |
|---|---|---|
| 秩 | 编译期 | `Tensor<2>` |
| 张量类别 | 编译期 | `Tensor<2, Int>` |
| 每维尺寸 | 运行时 | `[batch, features]` |
| 精确 dtype | 运行时 | F32、BF16、I32 |
| 后端与硬件 | 运行时 Device | Flex、CUDA、WGPU |
| 可用后端集合 | Cargo 编译期 | `flex`、`cuda` features |

这种设计让同一模型二进制能够在已编译进来的多个后端间选择，同时保留
秩相关 API 的静态检查。

## 所有权与并发约束

Tensor 作为 Rust 值参与移动和 clone。公开后端契约要求适合跨线程使用的
类型边界，具体设备资源则封装在安全抽象中。所有权帮助系统判断何时可原地
复用 buffer，但不能单独解决异步设备错误或数据竞争；后端实现仍需维护
正确同步。

对教学代码而言，显式 clone 也暴露了计算图中的分支。每个 clone 不是图中
的新数值节点，而是对同一 Tensor primitive 的另一个句柄；真正的算子调用
才产生新结果和依赖。

## 计算图与 IR 的区别

计算图泛指操作与依赖；IR 则是为了分析、转换或执行而定义的具体数据结构。
一个 IR 必须规定：

- 操作和 Tensor 如何编号；
- shape、dtype、Device 等属性如何表示；
- 控制流与副作用是否进入表示；
- 哪些变换保持语义；
- 如何 lowering 到 Kernel 或远程协议。

Burn 的 `OperationIr`、Fusion 搜索图和 CubeCL IR 处于不同层。第二章只
使用 autodiff tape 理解反向依赖，第 4 章再研究这些 IR 的结构和转换。

## 调度直觉

如果两个节点互不依赖，它们理论上可以并行；如果多个合法执行顺序存在，
调度器还要考虑：

- 设备可用的流或队列；
- 中间 Tensor 的内存峰值；
- Kernel 启动和同步成本；
- 跨设备数据移动；
- 后续融合机会。

本章只建立直觉。固定快照中的分工是：

| 路径 | 调度窗口 | 本章是否展开 |
|---|---|---|
| Flex eager | 单次 Tensor op 尽快执行 | 用于语义与 autodiff tape |
| Burn Fusion | 按 StreamId 延迟注册，再搜索执行块 | 第 4 章 |
| CubeCL Runtime | launch 入队；read/`Device::sync` 才是完成边界 | 第 3–4 章 |

因此：拓扑序与控制流影响“有哪些依赖”；Fusion stream 决定“何时物化成
Kernel”；二者不要混称。数据加载与训练执行的流水线属于更高层调度，第 5、
6 章展开；跨节点放置与拓扑感知调度属于第 6、9 章。

## 本章与第 4 章的边界

本章需要知道“记录依赖有多种目的”，并停在 autodiff tape 与调度直觉。
第 4 章将继续：

- Burn IR 中的 OperationIr 与 TensorId；
- Fusion 如何按 stream 搜索可融合块，以及 sync 如何切断窗口；
- CubeCL 如何从 Scope 生成 KernelDefinition 并 JIT；
- 运行时如何管理内存生命周期、缓存、调优与重放。

