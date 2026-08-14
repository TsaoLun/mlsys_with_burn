# CubeCL 编程模型

CubeCL 同时包含 Kernel 语言、IR、编译器和 Runtime。本节聚焦程序员可见的
边界；第 4 章再讨论 IR、融合和运行时系统。

## 1. `#[cube]` 不是普通 Rust 函数

`#[cube]` 过程宏读取 Rust 风格语法，并为受支持的表达式构造 CubeCL IR。
因此它能复用泛型、trait 和编译期分支等 Rust 抽象，但不能把任意 host Rust
代码原样放到设备执行。

```rust,ignore
#[cube]
fn inner<F: Float>(value: F) -> F {
    value * value
}
```

带 `launch` 或 `launch_unchecked` 参数时，宏还会生成 host 侧 launch
入口。普通 `#[cube]` 函数可被其他 Kernel 组合，却不能独立提交。

`#[comptime]` 参数在 Kernel 构造期已知，可用于选择算法、展开循环或固定
tile。这样能产生特化代码，也可能增加编译数量；它不是普通运行时标量。

### 宏在替你写什么

对本章 GEMM 阶梯实验的 `gemm_naive_kernel` 运行 `cargo expand`，可以
看到 `#[cube(launch_unchecked)]` 生成了三样东西（本书所用版本，
节选并简化）：

**第一，展开函数**：与原函数同名的模块里出现一个 `expand`，签名中
每个类型都换成了它的 `ExpandType`，函数体不再计算数值，而是向
`Scope` 里**登记 IR**——`row * k + i` 变成一串
`__expand_mul_method`/`__expand_add_method` 调用：

```rust,ignore
mod gemm_naive_kernel {
    pub fn expand<F: Float>(
        scope: &cubecl::prelude::Scope,
        a: &<[F] as CubeType>::ExpandType,
        // …每个参数都是 ExpandType…
    ) { /* __expand_* 调用链，把表达式写进 scope */ }
}
```

这解释了第 1 节的论断：`#[cube]` 函数在 host 上"执行"一次的产物是
IR，不是结果。也解释了为什么循环里的 Rust `if`/泛型/`#[comptime]`
能自然工作——它们在展开期就是普通 Rust 控制流，走过哪条分支，哪条
分支的 IR 才被登记（与第 2 章 tape 只记录执行路径同构）。

**第二，Kernel 类型**：`struct GemmNaiveKernel<F, R>` 实现
`CubeKernel`，它的 `define()` 创建 `KernelBuilder`、注入 Runtime 与
设备属性、调用上面的 `expand`，产出 `KernelDefinition`：

```rust,ignore
impl<F: Float, __R: Runtime> CubeKernel for GemmNaiveKernel<F, __R> {
    fn define(&self) -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(__R::target_properties());
        builder.device_properties(self.client.properties());
        // …注册类型与参数，运行 expand，收束成 KernelDefinition
    }
}
```

第 4 章讲的 JIT 编译缓存，键的来源就在这里：设备属性、类型注册与
comptime 值都参与 `define()`，任何一项变化都是一个新的特化。

**第三，launch 入口**：`pub unsafe fn launch_unchecked(client,
cube_count, cube_dim, 参数…)` 把 `RuntimeArg` 逐个注册进
`KernelLauncher` 再提交。实验里调用的那个"函数"，其实是宏写好的
装配代码。

三件产物对应三个时刻：expand 在 **Kernel 构造期**运行（host 普通
Rust），define 在**首次编译期**运行（产出 JIT 输入），launch 入口
在**每次提交时**运行。想亲眼看全文，运行
`cargo expand -p ch03-gemm-ladder --lib --features wgpu`。

## 2. Runtime 与 ComputeClient

`Runtime` 关联设备、编译器和计算服务。host 通过
`R::client(device)` 获得计算客户端（`ComputeClient<R>`）——向该 Runtime
创建 buffer、launch Kernel 并读回结果的入口——再完成：

1. 创建输入和输出 buffer；
2. 选择 `CubeCount` 与 `CubeDim`；
3. 把 buffer 描述为 Kernel 参数；
4. launch Kernel；
5. 在验证时读回结果。

CPU、WGPU、CUDA 和 HIP/ROCm 路径是否可用取决于 feature 与平台。相同
Kernel IR 可以由不同 Runtime 编译，并不意味着各 Runtime 的能力集合相同。
代码应查询 device properties，而不是把高性能 GPU 特性当成基础语言语义。

### 2.1 多 Runtime：同一 IR，不同完成边界

按本书所用版本，CubeCL 至少暴露这些 Runtime 类型（路径相对 `cubecl/`）：

| Runtime | 源码位置（示意） | 典型目标 | 本书默认 |
|---|---|---|---|
| `CpuRuntime` | `crates/cubecl-cpu/src/runtime.rs` | LLVM/MLIR CPU | 默认实验 |
| `WgpuRuntime` | `crates/cubecl-wgpu/src/runtime.rs` | WGSL 等图形栈 | 可选 `--features wgpu` |
| `CudaRuntime` | `crates/cubecl-cuda/src/runtime.rs` | NVIDIA GPU | 源码导读；非默认示例 |
| `HipRuntime` | `crates/cubecl-hip/src/runtime.rs` | AMD GPU | 源码导读；非默认示例 |

阅读顺序建议：

1. 先用主机参考实现（host reference：普通 host 代码写出的可观察正确结果）
   写清语义，再在 `CpuRuntime` 上验证（本章实验；见实验节
   `scale_reference`）；
2. 有图形驱动时，用同一 Kernel 走 `WgpuRuntime`，对照同一 host reference；
3. 再打开 `CudaRuntime` / `HipRuntime` 源码，看 client、编译与同步入口
   如何对应 GPU 完成边界——不必本机装齐驱动也能读懂契约。

`CpuRuntime` 是默认可跑路径，不是语义定义本身；Plane 大小、共享内存与
完成边界仍因 Runtime 而异，不能从 CPU 正确性直接外推。

Burn 侧通过 `burn-cubecl` / `burn-wgpu` / `burn-cuda` 把 `Device` 接到这些
Runtime；Flex **不**走这条桥。因此“Tensor API 统一”不等于“默认已经测过
所有 GPU 后端”。

## 3. 边界检查与安全责任

Kernel 常用以下保护：

```text
if ABSOLUTE_POS < input.len() {
    output[ABSOLUTE_POS] = ...
}
```

`#[cube(launch)]` 使用 checked execution mode；`launch_unchecked`
允许跳过一部分检查以减少开销。二者都不能证明 host 提供的 raw buffer
长度真实。本版中的 `BufferArg::from_raw_parts` 是 `unsafe fn`：
调用者若声明错误长度，Kernel 可能越界访问。

因此本章示例把 `unsafe` 限定在一个小块内，并写出两项不变量：

- handle 对应的 allocation 确实容纳 `len` 个 `f32`；
- Kernel 在使用全局索引前执行边界保护。

这不是“用 Rust 就自动没有设备内存错误”。Rust 类型系统保护 host 代码的
普通引用；跨 Runtime 的 raw allocation 描述仍需要人工证明。教学 crate
也因此对 `unsafe_code = "forbid"` 做了局部、有注释可复核的例外。

## 4. Slice、Vector 与 Tensor 参数

- `&[F]` / `&mut [F]` 表达一维 buffer；
- `Vector<F, N>` 表达连续的 N 个元素，供向量化使用；
- CubeCL Tensor 参数还携带 shape 与 stride，适合多维寻址。

当前版本使用 `Vector`，不是旧资料中的 `Line` 类型名。向量宽度必须与
buffer 布局、元素总数和 Runtime 能力一致。多维 Tensor 的 raw shape 和
stride 同样属于 unsafe 合约的一部分。

## 5. 从正确 Kernel 到高性能 Kernel

建议按以下顺序开发：

1. 用 host reference 定义可观察语义（对照标准，不是某个 Runtime）；
2. 实现带显式边界的标量 Kernel；
3. 在 CPU 或 checked mode 中验证多个 shape；
4. 再引入 Vector、plane、共享内存和矩阵指令；
5. 在真实目标设备上测量，并保留回退路径（fallback）。

这种顺序将“算法错了”和“优化只在某设备不成立”分开。CubeCL 提供可移植
表达方式，但性能可移植性仍需要多种策略和测量，这正是 CubeK 的职责之一。

