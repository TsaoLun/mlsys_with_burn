# CubeCL 编程模型

CubeCL 同时包含 Kernel 语言、IR、编译器和 Runtime。本节聚焦程序员可见的
边界；第 4 章再讨论 IR、融合和运行时系统。

## 1. `#[cube]` 不是普通 Rust 函数

`#[cube]` 过程宏读取 Rust 风格语法，并为受支持的表达式构造 CubeCL IR。
因此它能复用泛型、trait 和编译期分支等 Rust 抽象，但不能把任意 host Rust
代码原样放到设备执行。

```rust
#[cube]
fn inner<F: Float>(value: F) -> F {
    value * value
}
```

带 `launch` 或 `launch_unchecked` 参数时，宏还会生成 host 侧 launch
入口。普通 `#[cube]` 函数可被其他 Kernel 组合，却不能独立提交。

`#[comptime]` 参数在 Kernel 构造期已知，可用于选择算法、展开循环或固定
tile。这样能产生特化代码，也可能增加编译数量；它不是普通运行时标量。

## 2. Runtime 与 ComputeClient

`Runtime` 关联设备、编译器和计算服务。host 通过
`R::client(device)` 获得 `ComputeClient<R>`，再完成：

1. 创建输入和输出 buffer；
2. 选择 `CubeCount` 与 `CubeDim`；
3. 把 buffer 描述为 Kernel 参数；
4. launch Kernel；
5. 在验证时读回结果。

CPU、WGPU、CUDA 和 HIP/ROCm 路径是否可用取决于 feature 与平台。相同
Kernel IR 可以由不同 Runtime 编译，并不意味着各 Runtime 的能力集合相同。
代码应查询 device properties，而不是把高性能 GPU 特性当成基础语言语义。

## 3. 边界检查与安全责任

Kernel 常用以下保护：

```text
if ABSOLUTE_POS < input.len() {
    output[ABSOLUTE_POS] = ...
}
```

`#[cube(launch)]` 使用 checked execution mode；`launch_unchecked`
允许跳过一部分检查以减少开销。二者都不能证明 host 提供的 raw buffer
长度真实。固定快照中的 `BufferArg::from_raw_parts` 是 `unsafe fn`：
调用者若声明错误长度，Kernel 可能越界访问。

因此本章示例把 `unsafe` 限定在一个小块内，并写出两项不变量：

- handle 对应的 allocation 确实容纳 `len` 个 `f32`；
- Kernel 在使用全局索引前执行边界保护。

这不是“用 Rust 就自动没有设备内存错误”。Rust 类型系统保护 host 代码的
普通引用；跨 Runtime 的 raw allocation 描述仍需要人工证明。教学 crate
也因此对 `unsafe_code = "forbid"` 做了局部、可审计的例外。

## 4. Slice、Vector 与 Tensor 参数

- `&[F]` / `&mut [F]` 表达一维 buffer；
- `Vector<F, N>` 表达连续的 N 个元素，供向量化使用；
- CubeCL Tensor 参数还携带 shape 与 stride，适合多维寻址。

当前固定快照使用 `Vector`，不是旧资料中的 `Line` 类型名。向量宽度必须与
buffer 布局、元素总数和 Runtime 能力一致。多维 Tensor 的 raw shape 和
stride 同样属于 unsafe 合约的一部分。

## 5. 从正确 Kernel 到高性能 Kernel

建议按以下顺序开发：

1. 用 host reference 定义可观察语义；
2. 实现带显式边界的标量 Kernel；
3. 在 CPU 或 checked mode 中验证多个 shape；
4. 再引入 Vector、plane、共享内存和矩阵指令；
5. 在真实目标设备上测量，并保留 fallback。

这种顺序将“算法错了”和“优化只在某设备不成立”分开。CubeCL 提供可移植
表达方式，但性能可移植性仍需要多种策略和测量，这正是 CubeK 的职责之一。

