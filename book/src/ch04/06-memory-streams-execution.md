# 内存、Stream 与异步执行

优化计划只有在正确管理资源和时序时才成立。运行时既要减少 allocation 和
等待，又不能让重排破坏生命周期。

## 1. Tensor 大小只是起点

连续 Tensor 的数据字节数可粗略写为：

$$
\text{bytes}=\prod\_i \text{shape}\_i\times \text{sizeof(dtype)}。
$$

实际 allocation 还受对齐、量化 metadata、padding、layout、内存池粒度和
Runtime 约束影响。view 可能共享 allocation；非连续 Tensor 也不能只用
逻辑元素数描述地址范围。

## 2. 生命周期与复用

若 Tensor A 的最后一次读取发生在 Tensor B 创建前，两者可能复用同一块
设备内存。一般图中的最佳静态内存规划很难，动态 shape 和异步 stream 又
让精确生命周期更复杂，所以运行时常使用池化和启发式。

用条带图表示时间（横轴）与 allocation（纵轴）。连续 `add → exp` 且中间
结果不被同步读回时，中间值的寿命可以很短，甚至落入同一融合块内部：

```text
时间 →
t0   t1        t2
left ████
right ████
 mid      ██                 （仅块内临时，可能不单独暴露）
 out         ████████
```

若在 `add` 后 `Device::sync()`（本章 FusionInspector 实验的切分路径），
中间结果必须在同步点可观察，寿命被拉长，复用窗口也切开：

```text
时间 →
t0   sync    t1
left ████
right ████
 mid      ████████           （同步边界强制物化）
 out              ████████
```

Burn IR 的 `TensorStatus` 提供局部依据：

- `NotInit`：输出尚无现有 handle；
- `ReadOnly`：输入还可能共享，读取时保留 handle；
- `ReadWrite`：最后使用者可取得所有权，具备原地/复用机会。

`HandleContainer` 以 TensorId 管理底层 handle。`ReadWrite` / in-place
只是必要条件之一，不是充分条件：shape、dtype、别名、Kernel 语义以及
是否仍有其他读者，都必须同时满足。不能把状态枚举读成“运行时总会原地
覆盖”。

## 3. CubeCL 内存池

固定 CubeCL Runtime 包含 sliced、exclusive、persistent 等内存池策略。
池化减少频繁向驱动申请/释放的成本，也会带来碎片、缓存上限和生命周期
管理问题。

graph capture 等场景可能要求 capture 窗口内不再动态分配，因此使用
persistent pool。主设备池、staging/pinned buffer 和 metadata buffer
不一定由同一配置控制，不能把“配置了内存池”理解为所有内存都走同一算法。

## 4. Launch 通常是异步的

ComputeClient 将 Kernel 提交到 stream；host 返回不表示设备已完成。
提交与等待接口包括：

- `read`/`read_one`：必须等值可读；
- `sync`：等待对应 stream；
- `flush`：主要把待处理命令提交/推进，不应跨 Runtime 假设设备已经完成；
- 有依赖的后续命令：由 stream 顺序或事件保证。

只测 host 提交时间会严重低估执行时间。错误也可能延迟到 read/sync 才被
报告。CPU 的 flush 实现可能表现为等待，但这不是所有 GPU Runtime 的统一
完成语义。

在 GPU/图形 Runtime 上阅读时，把下列问题当作检查清单：

1. 这次 `read`/`sync` 等到的是哪条 stream？
2. HtoD / DtoH 是否与计算重叠，还是被多余同步切开？
3. Fusion 计划减少的是中间读写，还是只在 CPU Fusion 路径上可见？
4. 设备 graph capture（若该 Runtime 支持）复用的是命令序列，不是 autodiff tape。

默认实验仍在 CPU Fusion 上观察计划切分；有 WGPU/CUDA 环境时，用同一清单
核对完成边界，不要把 CPU 上的 flush 习惯直接抄到 GPU。

## 5. 多 Stream 与依赖

同一 stream 通常保持提交顺序，不同 stream 可能并行。跨 stream 使用同一
Tensor 时必须建立依赖：

```text
stream A: produce x ──event/同步──▶ stream B: consume x
```

Burn Fusion 按 StreamId 管理延迟队列。跨线程/stream 共享 view 可能要求
先 drain 来源队列再创建别名，以免旧计划重排导致 use-after-free。

多 stream 也会增加内存峰值：并发块的生命周期重叠，原本可复用的 allocation
可能必须同时存在。并行度与内存不是独立目标。

## 6. Eager、Fusion 与设备 graph

- **Flex eager**：每个 Tensor op 直接执行，仍可能使用底层异步设备操作；
- **Burn Fusion**：延迟注册 Tensor op，生成融合/回退执行计划；
- **CubeCL stream**：提交已编译 Kernel 和内存操作；
- **设备 graph capture**：在底层 Runtime 支持时记录命令用于重放，受
  allocation 和 stream 约束；CPU 路径没有对应实现。

在支持 graph capture 的底层 Runtime 上，Fusion block 可以进入捕获命令，
但二者不是同一抽象。前者改变 Tensor 操作如何组合，后者复用命令提交序列。

## 7. 同步是一种语义与性能操作

同步提供确定的可观察点，便于读回、调试和错误定位；它也会缩小融合窗口、
阻止 host/device 重叠，并可能增加等待。本章实验把同步放在两个元素级操作
之间，故意展示计划切分。生产代码不应为了“保险”在每一步同步，而应由真实
数据依赖和观测需求决定。

