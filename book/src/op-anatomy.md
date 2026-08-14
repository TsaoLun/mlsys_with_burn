# 算子解剖：tanh 的完整一生

随便挑一个算子，你能不能说出一次调用会牵动哪些文件、每一层为什么必须
存在、缺了哪一层会发生什么？本页用 `tanh` 走一遍——它足够简单（一元、
无形状变化），又足够完整（有反向规则、有 GPU kernel、参与融合）。
对照总表见[一次调用会经过哪些层](crate-map.md)。

## 一次调用牵动的十层

| # | 层 | 源码位置（相对 `burn/crates/`） | tanh 在这一层的形态 |
|---|---|---|---|
| 1 | 用户 API | `burn-tensor/src/tensor/api/float.rs` | 一行转发 |
| 2 | Backend 契约 | `burn-backend/src/backend/ops/tensor.rs` | trait 方法声明 |
| 3 | 运行时分派 | `burn-dispatch/src/ops/tensor.rs` | 按设备转发宏 |
| 4 | autodiff 装饰器 | `burn-autodiff/src/ops/tensor.rs` | 反向规则 + checkpoint 策略 |
| 5 | Flex CPU 实现 | `burn-flex/src/ops/float.rs`、`ops/unary.rs` | 元素级闭包 |
| 6 | CubeCL 桥 | `burn-cubecl/src/ops/tensor.rs` | JIT 一元 kernel 请求 |
| 7 | Fusion 前端 | `burn-fusion/src/ops/tensor.rs` | 描述注册 + 回退执行体 |
| 8 | IR 词汇表 | `burn-ir/src/operation.rs` | 枚举变体 `Tanh(UnaryOpIr)` |
| 9 | 契约测试 | `burn-backend-tests/tests/` | 同一断言跑所有后端 |
| 10 | 你的验证 | `examples/ch02-ch04-op-anatomy` | 数值断言（本页末） |

下面逐层走读。每层只贴关键行；行号以固定版本为准。

## 1. 用户 API：一行转发

`Tensor::tanh` 在 API 层几乎不存在——它把工作交给张量类别 `K`：

```rust,ignore
// burn-tensor/src/tensor/api/float.rs
pub fn tanh(self) -> Self {
    Tensor::new(K::tanh(self.primitive))
}
```

API 层的职责是类型与文档，不是计算。第 2 章讲的「编译期秩、运行时
形状」就体现在这一层的签名上。

## 2. Backend 契约：每个后端必须回答的问题

```rust,ignore
// burn-backend/src/backend/ops/tensor.rs
fn float_tanh(tensor: FloatTensor<B>) -> FloatTensor<B>;
```

trait 声明是全书反复出现的「契约」一词的字面形态：任何自称 Burn
后端的类型都必须给出这个函数。**缺了这一层的实现，代码无法编译**——
这就是 Rust 类型系统替框架执行的后端完备性检查。

## 3. 运行时分派：设备决定走哪个后端

```rust,ignore
// burn-dispatch/src/ops/tensor.rs
fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
    unary_float!(tensor, float, |tensor| B::float_tanh(tensor) => Float)
}
```

第 1 章探测过的 `DispatchDevice` 在这里起作用：`unary_float!` 宏按
张量所在设备把调用转发给 Flex、CubeCL 等具体后端。这一层解释了为
什么这一版的 `Tensor` 类型不携带后端泛型——后端选择发生在运行时。

## 4. autodiff 装饰器：反向规则住在哪里

这是解剖中最值得细读的一层。`Autodiff<B>` 也是一个后端，它包装
内层后端并在 `float_tanh` 里注册反向规则：

```rust,ignore
// burn-autodiff/src/ops/tensor.rs
fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
    #[derive(Debug)]
    struct Tanh;

    retro_unary!(RetroTanh, B::float_tanh);

    impl<B: Backend> Backward<B, 1> for Tanh {
        type State = NodeId;

        fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients,
                    checkpointer: &mut Checkpointer) {
            let input = checkpointer.retrieve_node_output(ops.state);
            let state = B::float_tanh(input);
            unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                let value = B::float_add_scalar(
                    B::float_neg(B::float_powi_scalar(state, 2.into())),
                    1f32.into(),
                );
                B::float_mul(grad, value)
            });
        }
    }
    // …prepare::<C>([tensor.node]).memory_bound()
    //  .retro_forward(RetroTanh::…).parents([&tensor]).stateful()…
}
```

四个机制在十几行里同时出现：

- **反向规则就是数学**：`grad * (1 - tanh(x)^2)`，用后端算子而不是
  标量循环表达，所以反向传播本身也能在 GPU 上跑；
- **反向需要前向值**：第 2 章说 tape 必须保存中间结果，这里的
  `State` 与 `checkpointer.retrieve_node_output` 就是那句话的实现；
- **checkpoint 策略按算子声明**：`memory_bound()` + `retro_forward`
  表示「不保存输出，反向时用 `RetroTanh` 重算」——第 6 章讲的激活
  重计算不是全局开关，而是每个算子自己声明的属性；
- **装饰器模式**：autodiff 不是编译器 pass，而是一层实现了同一契约
  的后端包装——这就是第 2 章「tape 与 Fusion IR 是两套机制」的
  结构原因。

## 5. Flex CPU：语义的最终落点

```rust,ignore
// burn-flex/src/ops/float.rs
fn float_tanh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
    unary::tanh(tensor)
}

// burn-flex/src/ops/unary.rs
pub fn tanh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.tanh(), |x| x.tanh())
}
```

绕了一大圈，`tanh` 最终就是标准库的 `f32::tanh`/`f64::tanh` 逐元素
执行（两个闭包对应两种浮点宽度）。这也是本页示例用标量参考做断言
的依据。

## 6. CubeCL 桥：同一契约的 GPU 回答

```rust,ignore
// burn-cubecl/src/ops/tensor.rs
fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
    unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Tanh)
}
```

CubeCL 后端不写标量循环，而是请求一个一元 elementwise kernel：
`BasicFloatUnaryKind::Tanh` 最终 lower 成目标平台的 `tanh` 指令或
内建函数（第 4 章的 lowering/JIT/缓存全流程从这里开始）。

## 7. Fusion 前端：先描述，再决定怎么执行

```rust,ignore
// burn-fusion/src/ops/tensor.rs
fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
    unary_float_ops!(TanhOps, B::float_tanh);

    let streams = StreamId::current();
    let client = tensor.client.clone();
    let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

    client.register(
        streams,
        OperationIr::Float(desc.out.dtype, FloatOperationIr::Tanh(desc.clone())),
        TanhOps::<B>::new(desc),
    )
    // …
}
```

Fusion 后端把调用翻译成**两样东西**一起注册：一份描述
（`FloatOperationIr::Tanh`，供搜索融合块）和一个回退执行体
（`TanhOps`，融合不成立时逐算子执行）。第 4 章 FusionInspector 观察
到的执行计划，源头就是这一层的注册流。

## 8. IR 词汇表：一个算子 = 一个词

```rust,ignore
// burn-ir/src/operation.rs
pub enum FloatOperationIr {
    // …
    Tanh(UnaryOpIr),
    // …
}
```

`burn-ir` 的枚举是 Fusion 能「谈论」的算子全集。一个算子要参与
融合，先得在这本词汇表里有名字——枚举之外的操作对优化器不可见，
只能走回退路径。

## 9. 契约测试：同一断言，所有后端

`burn-backend-tests` 里没有「Flex 的 tanh 测试」和「CUDA 的 tanh
测试」，只有**一份**测试（如 `tests/autodiff/trig.rs` 的
`should_diff_tanh`）：构造输入、`tanh`、`backward`、按容差比对。
后端由 Cargo feature 在编译期注入——上游用 `cargo test-cpu`、
`cargo test-wgpu`、`cargo test-cuda` 等别名切换。这套机制是「Backend
抽象真的成立」的可执行证据：任何后端答错同一道题，测试就红。第 5
层与第 6 层给出不同实现、必须通过同一测试，正是契约的意义。

## 10. 亲手验证

`examples/ch02-ch04-op-anatomy` 把上面的关键结论变成断言——前向
等于标量参考（第 5 层）、反向精确等于 `1 - tanh(x)^2`（第 4 层）、
数值梯度独立核对、组合算子满足乘积法则（tape 的组合性）：

```rust,ignore
{{#include ../../examples/ch02-ch04-op-anatomy/src/lib.rs:backward}}
```

```bash
cargo run  -p ch02-ch04-op-anatomy --locked
cargo test -p ch02-ch04-op-anatomy --locked
```

```text
前向（API → dispatch → Flex）与标量 f32::tanh 的最大误差：0.00e0
反向（autodiff Tanh backward）与 1 - tanh(x)^2 的最大误差：0.00e0
组合 y = x·tanh(x) 的梯度与乘积法则的最大误差：0.00e0
```

## 换一个算子，同样的十层

把 `tanh` 换成任何算子，地图不变，变的是每层的形态。最有信息量的
是第 4 层：**`State` 里存什么，由这个算子的数学决定**。三个标本
对照（均摘自 `burn-autodiff/src/ops/tensor.rs`）：

```rust,ignore
// add：State = (Shape, Shape)——不需要任何前向值，
// 只需要两个输入的形状；广播的反向是按形状归约。
impl<B: Backend> Backward<B, 2> for Add {
    type State = (Shape, Shape);
    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients,
                _checkpointer: &mut Checkpointer) {
        let (shape_lhs, shape_rhs) = ops.state;
        binary::<B, _, _>(ops.parents, ops.node, grads,
            |grad| broadcast_shape::<B>(grad, &shape_lhs),
            |grad| broadcast_shape::<B>(grad, &shape_rhs),
        );
    }
}
```

```rust,ignore
// sum：State = Shape——归约的反向是广播，
// 梯度就是 ones(输入形状) × 上游梯度。
unary::<B, _>(ops.parents, ops.node, grads, |grad| {
    let val = B::float_ones(ops.state, &grad.device(), grad.dtype().into());
    let grad = unsqueeze_like::<B>(grad, val.shape());
    B::float_mul(val, grad)
});
```

```rust,ignore
// matmul：State = 两个可选的 NodeId + 广播描述。
// ∂L/∂lhs = grad·rhsᵀ，∂L/∂rhs = lhsᵀ·grad——反向需要两个输入的
// 前向值，所以各存一个 checkpoint 结点引用。
impl<B: Backend> Backward<B, 2> for Matmul {
    type State = (Option<NodeId>, Option<NodeId>, BinaryOpsBroadcast);
    // ...
}
// 注册处的细节：lhs 只在 rhs 被追踪时才 checkpoint——
// 因为 lhs 的前向值只出现在 rhs 的梯度公式里，反之亦然。
let lhs_state = rhs_tracked.then(|| prep.checkpoint(&lhs));
let rhs_state = lhs_tracked.then(|| prep.checkpoint(&rhs));
```

三者排成一行看：`add` 只存形状（`_checkpointer` 根本没用上）、
`sum` 只存输入形状、`tanh` 与 `matmul` 要存结点引用以取回前向值，
而 `matmul` 还会按「对侧是否被追踪」跳过不需要的 checkpoint——
**反向公式用到什么，checkpoint 才负担什么**，第 2 章的激活内存
预算公式在这一层有了逐算子的出处。其余层的变化对应关系：二元
算子的 IR 用 `BinaryOpIr`；归约算子在第 7 层要换 Reduce fuser
（ElementWise fuser 不收）；`matmul` 在第 6 层从一元 elementwise
kernel 换成 CubeK 的 tile/stage/global 组件（第 3 章）。

本页示例的 `add_gradient_reduces_over_broadcast`（`[1,3]` 广播进
`[2,3]`，梯度回到 `[1,3]` 且值为按列归约）与
`sum_gradient_broadcasts_ones`（梯度恰为全 1）把前两个标本的反向
语义变成断言。

反过来读也成立——**缺一层会发生什么**：第 2 层缺实现是编译错误；
第 4 层缺规则，算子能前向但不可训练；第 7/8 层缺注册，算子能算但
会打断融合块。每一层的存在都有一个可观察的失效模式。

默认路径不改上游源码、也不跑 GPU 层；第 6 层的可观察版本见第 3 章的
wgpu 可选实验与第 4 章 FusionInspector。
