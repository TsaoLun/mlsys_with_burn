# CubeCL Lowering、JIT 与缓存

Burn Fusion 决定一个可执行优化块后，CubeCL 还要把 Kernel 描述变成目标
Runtime 可执行的产物。本节沿源码追踪这条链，不假设所有后端共享同一
最终 IR。

## 1. 从 `#[cube]` 到 Scope

`#[cube]` 过程宏生成 host 侧模块和 `expand` 函数。expand 不执行数值计算，
而是向 `cubecl_ir::Scope` 注册类型、值和指令。带 `launch` 的 Kernel
还会生成 launch glue 和 `CubeKernel` 实现。

`KernelBuilder` 结合：

- Scope body；
- buffer、Tensor 与 scalar 参数；
- `CubeDim`；
- address type、检查模式等设置；
- Runtime/device properties；

生成 `KernelDefinition`。Scope 更接近 Kernel 主体 IR，KernelDefinition
则补齐可编译/launch 所需接口信息。

## 2. 优化管线并不唯一

SPIR-V、CPU MLIR 和 CPP 类后端不会简单调用完全相同的一组 Pass。
源码中可以观察到两类路径：

- `cubecl-opt::Optimizer` 把 Scope 转为 CFG/SSA 风格表示并执行分析/优化；
- CPP 编译器还使用 shared-memory 分析、visitor 与 Scope post-processing。

因此“CubeCL 优化后 IR”必须注明具体 Compiler。打印 `cubecl-opt`
Optimizer 结果能解释 CFG/SSA 变换，却不能被当作 CUDA CPP 路径的逐字
最终 IR。

## 3. Lowering 与代码生成

Compiler 读取 KernelDefinition，经过目标相关 lowering，产生 MLIR/LLVM、
SPIR-V、CPP/设备源码或其他后端表示。lowering 可能处理：

- Cube/Unit/Plane 拓扑内建量；
- Vector、共享内存和同步；
- checked IO；
- dtype 与矩阵指令；
- address width；
- Runtime feature 差异。

第 3 章的 CubeCL 源码看起来设备无关，但性能与合法性信息会在这里落到
具体目标。某个 Runtime 不支持的操作必须在编译前过滤、lowering 时报错，
或由上层选择其他策略。

## 4. JIT 的首次成本

CubeCL 只为实际使用的 Kernel 变体编译，典型过程是：

```text
KernelId / 编译设置
    ↓ 查当前后端可用缓存
cache miss
    ↓ define + optimize + lower + target compile
加载 module/pipeline
    ↓
launch
```

这解释了为什么首次调用、稳态调用和新 shape/新特化参数的延迟不同。可靠
benchmark 要分开报告编译/autotune 与稳态执行，不能只选第二次运行并省略
预热事实。

CPU Runtime 使用进程内编译缓存，部分 CUDA/HIP/Metal/WGPU-SPIR-V
路径还可按配置使用持久化缓存。CubeCL 仍以 JIT 为主；本书不把这些有条件
的能力扩张为跨所有后端统一、可离线部署的完整 AOT 工具链。

### 4.1 选择、编译、缓存和执行是一条因果链

一次 Tensor 调用可以按下面的顺序追踪：

```text
op + shape/layout/dtype/device
  → Fusion block / fallback 计划
  → Strategy 候选与能力过滤
  → tune key 选择实现
  → KernelDefinition + 编译 key
  → cache hit 或 define/optimize/lower/compile
  → module/pipeline 加载
  → ComputeClient launch 入队
  → read/sync 物化并报告错误
```

每个箭头都可能改变成本。Fusion block 变了，候选和编译输入就可能变；
shape 或 comptime 参数变了，可能产生新的 tune key 和 KernelDefinition；
cache 命中只表示某一层结果可复用，不表示设备 module 已加载，也不表示
本次 launch 已完成。只有在 read 或明确同步之后，host 才能把设备结果、
执行错误和端到端耗时当作已观察事实。

## 5. 编译缓存与调优缓存不同

- **编译缓存**：KernelId 到目标编译产物；
- **autotune cache**：问题 tune key 到候选选择；
- **pipeline/module cache**：运行时已加载对象；
- **metadata cache**：shape/stride 等辅助设备数据。

它们的失效条件不同。源码变更、Compiler 版本、设备能力或配置改变时，
旧结果可能不可用。调优结果命中也不保证编译产物已加载。

## 6. 可以观察什么

CubeCL API 允许在 host 上：

- 调用宏生成的 expand 构造 Scope；
- 格式化 Scope；
- 运行 `cubecl-opt` 并打印优化表示；
- 借助 CPU Runtime 取得 device properties 和编译 Kernel。

这些接口比 Tensor API 更低层，也更易随版本变化。本章必做实验选择更稳定
的 Burn FusionInspector；Scope/Optimizer 打印作为源码练习，避免把内部
调试 API 伪装成面向用户的长期承诺。

