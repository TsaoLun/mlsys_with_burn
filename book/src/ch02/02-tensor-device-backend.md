# Tensor、Device 与运行时后端

## `Tensor<D, K>` 表达什么

本版中的公开张量定义可以简化为：

```rust,ignore
pub struct Tensor<const D: usize, K = Float>
where
    K: Basic,
{
    // 运行时 primitive
}
```

`D` 是编译期已知的**秩（rank）**，即维度数量；`K` 是 Float、Int、Bool
等张量类别。二者不等于完整 shape 和 dtype：

- `Tensor<2>` 保证它是二维浮点张量，但行数和列数在运行时确定；
- 默认浮点 dtype 由 Device settings 决定，也可在创建时显式指定；
- 同一个 `Tensor<2>` 类型可由 Flex、CUDA 或 WGPU Device 创建。

因此 Burn 在编译期捕获“秩和操作类别”，把“尺寸、精确元素类型与执行后端”
保留为运行时属性。这是一种折中：避免把所有动态信息都编码进 Rust 类型，
同时让许多秩不匹配在编译期暴露。

## 创建、形状与 dtype

常见创建方式包括 `from_data`、`from_floats`、`zeros`、`ones` 和
`random`。张量提供：

- `dims() -> [usize; D]`：取得固定长度的尺寸数组；
- `shape()`：取得 Shape 对象；
- `dtype()`：运行时元素类型；
- `device()`：执行与存储所在设备。

创建时只传 `&Device` 会使用设备默认 dtype；传入 `(&device, DType)` 可
显式选择。不同后端未必支持全部 dtype，系统应返回或报告不支持的组合。

## 广播

广播（broadcasting）让不同 shape 的张量参与逐元素运算。从尾部维度开始，
两个尺寸只有在相等或其中一个为 1 时才兼容。例如：

```text
[3, 1]
[1, 2]
──────
[3, 2]
```

秩相同不保证 shape 兼容，因此广播检查主要发生在运行时。广播还会影响
反向传播：被扩展的输入需要沿广播维度归约梯度，恢复原输入 shape。

例如上面的 `[3, 1] ⊙ [1, 2] → [3, 2]`：若输出每个元素的伴随值为 1，
右侧输入的两个元素各自被 3 行复用，其梯度是把对应列的 3 个伴随值
**求和**回 `[1, 2]`，而不是把 `[3, 2]` 的梯度原样截断。 Broadcasting
在内存上“免费”扩展了读视图，反向时则必须付出一次归约。

## 用 shape 和 dtype 估算内存

秩在编译期已知，但内存占用取决于运行时的 shape 和 dtype。一个
`[128, 1024]` 的 `f32` 张量占
$128 \times 1024 \times 4\ \text{B} = 512\ \text{KiB}$；改用 2 字节的
`bf16` 直接减半。训练时同一组参数往往同时存在多份拷贝：参数本身、
梯度，以及 Adam 类优化器的两个矩估计——参数量为 $P$ 的模型仅这部分
就约 $P \times 4\ \text{B} \times (1 + 1 + 2) = 16P$ 字节，还不包括
上一节讨论的激活。这类“每份状态几个字（word）”的口算是第 6 章优化器
选择和第 7 章部署 dtype 决策的基础。

## Device 与 Dispatch

0.22 的用户张量不再携带 Backend 类型参数。`Device` 内部包装
`DispatchDevice`，张量操作通过运行时分派到具体后端：

```text
Tensor<D, K>
    │ BridgeTensor
    ▼
Dispatch + DispatchDevice
    ├── Flex
    ├── CUDA / WGPU / ROCm
    ├── LibTorch / NdArray
    ├── Remote
    └── Autodiff(内部设备)
```

Cargo feature 决定哪些分派变体被编译进程序，Device 工厂方法选择其中一个
实例。教材默认使用 `Device::flex()`，因此无需 GPU 驱动。

## Backend trait 位于实现层

`burn-backend` 中的 `Backend`、`BackendTypes` 和操作 trait 仍然重要，
但它们主要面向后端与扩展作者。用户侧 Module 不再需要写成
`Model<B: Backend>`。

统一 Backend 契约保证基本操作语义，却不保证所有能力相同。例如 Flex 的
graph primitive 是不支持状态，CubeCL 后端则可能提供融合和 graph capture。

## 所有权与 clone

Rust 运算符通常按值接收 Tensor。一个值要参与多个后续操作时，需要显式
`clone()`：

```rust,ignore
let squared = input.clone() * input;
```

Tensor clone 通常是共享底层 buffer 的浅克隆，而不是立即复制全部设备
数据。后端可以在引用唯一且操作允许时复用存储；因此所有权信息也是运行时
优化依据。

`to_data` 或 `into_data` 会把结果读回主机并形成同步边界。仅为了继续设备
计算时，不应频繁读回数据。

## 错误与同步

`device.sync()` 等待已提交工作完成并返回 `ExecutionError`。异步后端可能
在同步或读回时才报告先前提交的错误。Flex 多数操作立即执行，但使用统一
同步 API 能让示例迁移到其他后端时保留正确边界。

Shape、广播或 Device 不匹配的张量运算目前可能通过带上下文的 panic 报告；
设备配置、执行和记录读写则有各自错误类型。教材示例会在可恢复边界传播
错误，而不是在库代码中无条件 `unwrap`。

