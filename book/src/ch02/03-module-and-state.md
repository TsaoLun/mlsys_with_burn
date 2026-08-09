# Module、参数与模型状态

神经网络通常由层递归组合而成。系统不仅要调用这些层，还要回答：

- 哪些 Tensor 是可训练参数？
- 参数位于哪个 Device？
- 模型共有多少参数？
- 如何切换训练与验证状态？
- 如何保存并重新加载模型状态？

Burn 用 `Module` 统一这些操作。

## Module derive

一个模型可以像普通 Rust struct 一样声明字段，并派生 `Module`：

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:module}}
```

`Linear` 自身实现 Module，内部权重和可选偏置会被递归访问；`Relu` 没有
可训练参数，但仍能作为模型组件。`forward` 是普通 Rust 方法，输入输出的
秩直接出现在 `Tensor<2>` 类型中。

`Module` 提供参数遍历、`num_params`、设备迁移和状态转换等操作。宏生成的
实现减少样板代码，但结构和字段所有权仍是普通 Rust 语义。

参数统计可以直接口算验证。`Linear(d_in, d_out)` 含权重
$d\_{in} \times d\_{out}$ 和偏置 $d\_{out}$；一个 784→128→10 的两层
MLP 共有 $(784 \times 128 + 128) + (128 \times 10 + 10) = 101770$
个参数，按 `f32` 计约 $398\ \text{KiB}$。本章实验中的
`num_params` 断言的就是这类递归求和；当统计结果与预期不符时，通常
意味着某个字段被错误地声明（或漏声明）为 `Param`。

## Param 与普通 Tensor

可训练张量通常包在 `Param<Tensor<D>>` 中。Param 让 Module visitor 识别
参数，并维护稳定标识与延迟初始化等信息。

不是 Module 中的每个 Tensor 都应成为参数：

- 需要优化并写入模型状态的权重使用 Param；
- 可从配置重新构造的常量可以保留为普通字段；
- 临时激活值只存在于 forward 调用中。

准确区分三者决定参数统计、设备迁移和保存行为。

## 参数 visitor 是状态边界

`Module` 的递归 visitor 不是单纯的反射工具，它定义了后续几个系统操作的
共同遍历边界：

```text
Module
 ├─ Param<Tensor> ── ParamId ── optimizer / ModuleRecord
 ├─ 子 Module ─────── recursive visit
 └─ 普通字段 ─────── config / 常量 / runtime handle
```

因此，新增一个模型字段时至少要问三件事：

1. 它是否参与梯度和 optimizer step？
2. 它是否必须随 ModuleRecord 保存，还是可以从 Config 重建？
3. 它迁移到另一个 Device 时，是否需要转换或拒绝？

例如，训练中的 BatchNorm 统计量、量化 scale、词表或 tokenizer 可能不
属于同一种参数；把它们全部包成 `Param` 会改变 optimizer 和 record 的
语义，把它们全部留作普通字段又可能导致恢复后推理不一致。固定 Burn
源码能验证 visitor/Param 的行为，但业务模型如何分类仍是应用的 schema
决定。

## Config 与初始化

层配置和已初始化参数是不同对象。`LinearConfig::new(3, 2)` 描述输入输出
尺寸，`.init(device)` 才在指定 Device 上创建 Linear 及其参数。

复杂模型可派生 `Config`，集中保存隐藏维度、层数或 Dropout 概率，再由
`init(&Device)` 生成 Module。这样配置可以序列化，而随机初始化和设备资源
留在明确的初始化阶段。

## ModuleRecord

固定版本使用非泛型 `ModuleRecord` 保存 Module 状态。典型流程是：

```text
Module
  └─ into_record() → ModuleRecord → save / into_bytes

新 Module + ModuleRecord
  └─ load_record() → 参数恢复后的 Module
```

这与旧文档中的 `Record<B>` 不同。Record 保存参数值，不替代模型结构和
Config；加载时仍需要兼容的 Module。dtype 策略可以选择遵循记录或转换到
目标 Module。

模型持久化格式、外部权重和 burn-onnx 会在第 7 章详细讨论。

## 训练状态与梯度状态

Module 的有效训练/验证模式、参数值和自动微分 tape 是三类不同状态：

- 本版中 `train()` / `valid()` 主要通过 autodiff Device 与 inner
  Device 转换有效模式，Dropout 等层据输入 Device 能力改变 forward，而非
  读取一个全局 Module 布尔标志；
- 参数值由优化器更新并由 ModuleRecord 保存；
- 梯度存在于一次 backward 返回的 Gradients 中。

保存参数不会自动保存当前动态计算图；训练检查点还可能包含优化器、步数和
随机状态。第 6 章会把这些状态组合成可恢复训练。

