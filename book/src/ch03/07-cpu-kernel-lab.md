# 实验：CPU 上运行 CubeCL Kernel

实验位于 `examples/ch03-cubecl-kernel`。它直接依赖 `pins.toml` 中的
CubeCL Git revision，并启用 `cpu`、`std` 和 `stdlib` feature。CPU Runtime
通过 LLVM/MLIR 路径编译 Kernel；首次构建明显慢于前两章，增量运行会快得多。

本实验验证编程模型和正确性，不把 CPU 时间当作 GPU 性能数据。

## 1. Host reference

`scale_reference` 用普通 Rust 定义语义：每个输入乘以整数 scale。测试把
Kernel 输出与它比较，而不是只观察打印内容。

scale 使用 `u32`，因为 `#[comptime]` 值会参与 Kernel 特化键，必须满足
固定宏实现所需的可哈希约束；`f32` 不能直接作为这个特化键。Kernel 内再将
它转换为元素类型。若 scale 需要频繁变化且不应触发特化，更合理的设计是
把它作为运行时标量参数。

## 2. Kernel

```rust
{{#include ../../../examples/ch03-cubecl-kernel/src/lib.rs:kernel}}
```

`F: Float` 让同一 Kernel 可为不同浮点类型构造 IR。本实验 host 只实例化
`f32`。`ABSOLUTE_POS` 将每个 unit 映射到一个元素；显式 guard 允许 launch
的 unit 数向硬件友好尺寸取整，而不会访问尾部之外。

`launch_unchecked` 表示调用者承担 launch 与参数合约。它不等于 Kernel
内部完全没有保护：本例仍保留长度检查。

## 3. Host launch

```rust
{{#include ../../../examples/ch03-cubecl-kernel/src/lib.rs:host}}
```

流程依次是：

1. 从 Runtime 取得 ComputeClient；
2. 上传输入，并分配等字节数输出；
3. 用设备属性选择合法 CubeDim，再用多个 cube 覆盖全部元素；
4. 构造两个 raw BufferArg 并 launch；
5. 读回字节并解释为 `f32`。

空输入会在分配和 launch 前返回，避免构造零尺寸 CubeDim。cube 数使用
checked conversion 转为 `u32`，不会静默截断超大 `usize`。

unsafe block 只覆盖必须证明的 raw 边界。`input_handle` 由同一个 input
创建，`output_handle` 分配了 `size_of_val(input)` 字节，两个 BufferArg
都声明 `input.len()` 个 `f32`，因此长度一致。若只修改其中一个数字，这个
证明就会失效。

根 workspace 默认禁止 unsafe。本 crate 没有继承该 lint，而是局部设置
`unsafe_code = "allow"`；这是一处有注释、有测试的 FFI/Runtime 边界，
不是全项目放宽。

## 4. 运行与测试

```bash
cargo run -p ch03-cubecl-kernel
```

预期输出：

```text
runtime: cpu
input:   [1.0, 2.0, 3.0, 4.0]
output:  [2.0, 4.0, 6.0, 8.0]
```

运行测试：

```bash
cargo test -p ch03-cubecl-kernel
```

测试使用包含负数、零和小数的输入，并与 host reference 做精确比较。整数
缩放对这些 `f32` 值可精确表示；换成近似函数或累加时应使用容差断言。

## 5. 观察编译边界

修改 `scale` 会产生不同的 comptime 特化。input 长度本身是 buffer 的运行时
元数据，但本例还把它交给 `CubeDim::new` 选择 launch 拓扑；CubeDim 属于
编译键，因此某些长度变化也可能触发不同编译配置。可以设置 CubeCL 日志
观察编译与缓存，但配置接口随快照演进；实验命令以源码中的
`CubeClRuntimeConfig` 为准。

不要用删掉 guard 的方式“观察越界”。unchecked raw buffer 错误可能不是
普通 Rust panic，而是 Runtime 错误甚至进程崩溃。

## 6. 可选 GPU 路径

本章没有默认启用 WGPU，以确保基础实验不要求图形 API；crate 提供可选
`wgpu` feature，并复用同一个 Kernel 与 reference：

```bash
cargo test -p ch03-cubecl-kernel --features wgpu
```

命令只有在系统存在 CubeCL/WGPU 可用 adapter 时才能通过。继续扩展 CUDA
等路径时应：

1. 为 crate 增加独立 feature 和对应 CubeCL Runtime；
2. 保留同一 host reference；
3. 根据 device properties 选择合法 CubeDim；
4. 在读回前同步；
5. 把设备、驱动、warm-up 和首次编译成本写入报告。

GPU 实验可以进一步加入 `Vector`、共享 tile 或 CubeK matmul 对照，但不能
把 CPU 测试通过当成这些硬件路径已经验证。

