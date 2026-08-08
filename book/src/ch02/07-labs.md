# 实验：张量、Module 与梯度

本章实验位于 `examples/ch02-tensor-basics`，统一使用 GitHub 固定 revision
的 Burn 和 Flex CPU。根 workspace 启用了 `std`、`flex`、`autodiff`
features。

## 1. 逐元素运算

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:example}}
```

`Tensor<1>` 在编译期固定秩为 1，长度 3 来自运行时数据。`left` 和
`right` 被运算符消费，因为之后不再使用，所以无需 clone。

完整文件中的 `values` helper 将 Tensor 读回 `Vec<f32>`，并把转换失败
映射为 `ExampleError`。

## 2. 广播与运行时 shape

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:broadcasting}}
```

两个输入都是 `Tensor<2>`，但 shape 分别为 `[3, 1]` 与 `[1, 2]`。相加
结果是 `[3, 2]`，按行展开为：

```text
[11, 21,
 12, 22,
 13, 23]
```

这证明 Rust 类型只固定了二维，广播兼容性和结果尺寸仍由运行时检查。

## 3. Module 与参数注册

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:module}}
```

Linear 从 3 个输入特征映射到 2 个输出特征，共有 $3 \times 2 = 6$ 个权重
和 2 个偏置，因此 `num_params()` 返回 8。输入 `[4, 3]` 的 batch 维保持
不变，输出 shape 为 `[4, 2]`。

实验在创建模型前调用 `device.seed(42)`，使随机初始化可复现；测试不依赖
具体随机值，只断言参数量与 shape。

## 4. Device 与 autodiff 能力标志

示例还分别读取 `Device::flex()` 和 `.autodiff()` 的
`is_autodiff()`。这只是设备能力标签，不是一次 backward；真正的 tape
行为仍由下一节的 `require_grad`、前向操作和 `backward` 验证。

## 5. 动态自动微分

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:autodiff}}
```

前向计算为：

```text
left  = [1, 7]
right = [4, 7]
product = left * right = [4, 49]
```

两个输入都通过 `require_grad()` 标记。`product.backward()` 以全 1 根
梯度反向传播，因此：

```text
∂product/∂left  = right = [4, 7]
∂product/∂right = left  = [1, 7]
```

`grad` 返回 Option，因为没有参与跟踪的 Tensor 不一定有梯度。示例把缺少
叶子梯度作为可传播错误，而不是在库代码中 panic。

## 6. 控制流只记录实际分支

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:branch_autodiff}}
```

`use_double = true` 时前向为 `input * 2`，梯度为全 2；为 `false` 时前向
为 `input + 1`，梯度为全 1。两次调用各自构建 tape，互不混入未执行分支。
这固定了图外控制流与 eager autodiff 的边界，不是完整训练循环。

## 7. detach 是 tape 边界

```rust,ignore
{{#include ../../../examples/ch02-tensor-basics/src/lib.rs:detach_autodiff}}
```

这个负向实验先让 `original` 成为 autodiff leaf，再用
`original.detach().require_grad()` 建立新的 `detached` leaf。前向只使用
新的 leaf，因此 backward 后：

- `original_gradient == None`：原始路径没有参与本次 tape；
- `detached_gradient == Some([3, 3])`：新 leaf 按 `detached * 3` 获得梯度；
- 输出 shape 和梯度 shape 都保持为 `[2]`。

这证明的是 tape 的连接语义，不是“所有 runtime 错误都会返回 Result”；
shape/device 错误仍应按固定 backend 的 API 契约单独验证。

## 8. 运行

```bash
cargo run -p ch02-tensor-basics
```

输出应包含：

```text
逐元素平均值：[2.0, 3.0, 4.0]
广播：[3, 2] -> [11.0, 21.0, 12.0, 22.0, 13.0, 23.0]
乘法前向值：[4.0, 49.0]
left 梯度：[4.0, 7.0]
right 梯度：[1.0, 7.0]
TinyModel：8 个参数，输出形状 [4, 2]
Device autodiff：普通=false，autodiff 包装=true
控制流分支 double：输出 [4.0, 6.0]，梯度 [2.0, 2.0]
detach：原始梯度=None，新 leaf 梯度=Some([3.0, 3.0])
```

## 9. 测试

```bash
cargo test -p ch02-tensor-basics
```

测试覆盖逐元素数值、广播、Module 参数注册、乘法梯度、控制流分支梯度和
detach 的 `Option`/数值/shape 状态。张量、广播和梯度行为可在固定上游的
`burn-backend-tests` 找到对应回归；Module 参数遍历与统计由 `burn-core`
的 Module 测试支撑。

## 10. 沿源码追踪

建议按顺序阅读：

1. `burn-tensor/src/tensor/api/base.rs`：`Tensor<D, K>`；
2. `burn-tensor/src/device.rs`：Device 与 autodiff 包装；
3. `burn-dispatch/src/device.rs`：运行时后端枚举；
4. `burn-core/src/module/base.rs`：Module visitor；
5. `burn-tensor/src/tensor/api/autodiff.rs`：backward 与 grad；
6. `burn-autodiff/src/runtime/`：动态 tape 的节点与反向步骤。

