# CubeCL Lowering、JIT 与缓存

Burn Fusion 决定一个可执行优化块后，CubeCL 还要把 Kernel 描述变成目标
Runtime 可执行的产物。本节沿源码追踪这条链。本版的关键变化是：Kernel
主体不再是自维护的指令图，而是 **Pliron 方言上的操作**；各 Compiler 在
同一套 IR 上跑 Pass，再降到 LLVM、SPIR-V、CPP 或 WGSL。

## 1. 从 `#[cube]` 到 Scope，再到 KernelDefinition

`#[cube]` 过程宏生成 host 侧模块和 `expand` 函数。expand 不执行数值计算，
而是向 `cubecl_ir::Scope` 登记类型、值和 Pliron 操作。带 `launch` 的
Kernel 还会生成 launch glue 和 `CubeKernel` 实现。

`KernelBuilder`（`cubecl-core/src/compute/builder.rs`）结合：

- Scope body（内部是 Pliron `Context` + `ModuleOp`）；
- buffer、Tensor 与 scalar 参数；
- `CubeDim` 与 `KernelSettings`；
- address type、检查模式；
- Runtime / device properties；

调用 `build()` 得到 `KernelDefinition`：

```rust,ignore
pub struct KernelDefinition {
    pub body: Scope,
    pub info: Info,              // 参数与 metadata 布局
    pub settings: KernelSettings,
}
```

定义在 `cubecl-runtime/src/kernel.rs`。Scope 更接近 Kernel 主体 IR，
KernelDefinition 补齐 Compiler 与 launch 所需的接口信息。

## 2. Pliron 方言与共享 Pass

CubeCL 把 Kernel 操作拆进方言，而不是一个无所不包的 opcode 枚举。源码
目录 `cubecl-ir/src/dialect/` 里可以读到：

| 方言（示意） | 解决什么 |
|---|---|
| `math` / `cmp` / `bitwise` | 算术、比较、位运算 |
| `memory` / `vector` | 分配、load/store、向量拼拆 |
| `scf` / `branch` | 结构化控制流与基本块分支；`scf` 注释写明：提升回 CPP/WGSL 太难，它服务 mem2reg 等需要 SSA 的变换 |
| `plane` / `barrier` / `synchronization` | 组内协同与同步 |
| `matrix` / `ssa_matrix` / `tma` | 矩阵指令与异步拷贝 |
| `atomic` / `asm` / `spirv` | 原子、内联汇编、SPIR-V 专用 |

`cubecl-opt` 提供可在后端之间共享的自定义 Pass：`mem2reg`、`sroa`
（标量替换聚合）、`simple_cse`、共享内存分配、buffer 可见性标注。
它们实现的是 Pliron 的 `Pass` / rewrite 接口，不再把 Scope 转成一份
独立的 petgraph CFG 再优化。

**优化管线并不唯一。** CPU 的 `PlironCompiler`（`cubecl-llvm`）会按函数
跑一组明确的 Pass，例如（顺序以源码为准）：SROA、SCCP、SimpleCSE、
化简、minifloat/bitwise lowering、DCE、`BranchToSCF`、`SCFToLlvmCf`、
入口 ABI、`CubeToLLVM`、SimplifyCFG、Mem2Reg，最后
`builtin_to_llvm_pass()`。SPIR-V 与 CPP（CUDA/HIP/Metal）走各自的
Compiler：它们消费同一份 `KernelDefinition`，但 conversion 与代码生成
不同。因此“CubeCL 优化后 IR”必须注明具体 Compiler；打开
`CUBECL_DEBUG_PLIRON` 打印的是 Pliron dump，不是 CUDA 源码。

## 3. Lowering 与代码生成

Compiler 读取 KernelDefinition，经过目标相关 lowering，产生：

| Compiler / crate | 典型产物 | 典型 Runtime |
|---|---|---|
| `cubecl-llvm` 的 `PlironCompiler` | LLVM IR → 本机代码 | `CpuRuntime` |
| `cubecl-spirv` | SPIR-V | Vulkan / 部分 wgpu |
| `cubecl-cpp` | CUDA / HIP / Metal 源码 | `CudaRuntime`、`HipRuntime`、原生 Metal |
| wgpu 的 WGSL 路径 | WGSL | `WgpuRuntime` |

lowering 可能处理：

- Cube / Unit / Plane 拓扑内建量；
- Vector、共享内存和同步；
- checked IO；
- dtype 与矩阵指令；
- address width；
- Runtime feature 差异；
- 硬件没有的 minifloat / 复数等到软件路径（polyfill）。

第 3 章的 CubeCL 源码看起来设备无关，但性能与合法性信息会在这里落到
具体目标。某个 Runtime 不支持的操作必须在编译前过滤、lowering 时报错，
或由上层选择其他策略。

CPU 路径曾经常被写成“经 MLIR 编译”。本版事实是：**IR 层是 Pliron
（MLIR 风格），代码生成走 `pliron-llvm` / LLVM**。`cubecl-cpu` 依赖
`cubecl-llvm`，`CpuCompiler` 就是 `PlironCompiler`。

## 4. JIT 的首次成本

CubeCL 只为实际使用的 Kernel 变体编译，典型过程是：

```text
KernelId / 编译设置
    ↓ 查当前后端可用缓存
cache miss
    ↓ define + Pliron Pass + lower + target compile
加载 module/pipeline
    ↓
launch
```

这解释了为什么首次调用、稳态调用和新 shape/新特化参数的延迟不同。可靠
benchmark 要分开报告编译/autotune 与稳态执行，不能只选第二次运行并省略
预热事实。

CPU Runtime 使用进程内编译缓存，部分 CUDA/HIP/Metal/WGPU-SPIR-V
路径还可按配置使用持久化缓存。`cubecl-environment` 把缓存、stream 策略
和环境 bundle（预热的 autotune / 编译结果）收成跨 std / wasm / no_std
的一层。CubeCL 仍以 JIT 为主；本书不把这些有条件的能力扩张为跨所有
后端统一、可离线部署的完整 AOT 工具链。

### 4.1 选择、编译、缓存和执行是一条因果链

一次 Tensor 调用可以按下面的顺序追踪：

```text
op + shape/layout/dtype/device
  → Fusion block / fallback 计划
  → Strategy 候选与能力过滤
  → tune key 选择实现
  → KernelDefinition + 编译 key
  → cache hit 或 define / Pliron Pass / lower / compile
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
- **metadata cache**：shape/stride 等辅助设备数据；
- **environment bundle**：可随二进制分发的预热缓存集合。

它们的失效条件不同。源码变更、Compiler 版本、设备能力或配置改变时，
旧结果可能不可用。调优结果命中也不保证编译产物已加载。Pliron 迁移
改变的是 **miss 时怎么编译**，不是“缓存可以跨 Compiler 或跨设备混用”。

## 6. 可以观察什么

CubeCL API 允许在 host 上：

- 调用宏生成的 expand 构造 Scope；
- 走 `CpuRuntime` 编译并对照 host reference；
- 打开 `pliron-dump` / `CUBECL_DEBUG_PLIRON` 看 Pass 之后的 IR（调试用）；
- 借助 CPU Runtime 取得 device properties 和编译 Kernel。

这些接口比 Tensor API 更低层，也更易随版本变化。本章必做实验选择更稳定
的 Burn FusionInspector；Pliron dump 作为源码练习，避免把内部调试 API
伪装成面向用户的长期承诺。
