# 练习、延伸阅读与来源

## 小结

Burn 0.22 将用户 Tensor 表示为 `Tensor<D, K>`：秩和张量类别进入 Rust
类型，shape、精确 dtype 与后端由运行时 Device 决定。张量操作通过
burn-dispatch 到达具体 Backend。

Module 递归组织层与 Param，Config 负责初始化配置，ModuleRecord 保存模型
状态。Eager 前向执行实际算子，autodiff Device 同时构建一阶反模式动态
tape；backward 产生 Gradients，但不会自动执行优化器更新。

计算图是依赖关系的通用概念。autodiff tape、Burn IR/Fusion 和 backend
graph capture 记录相似操作，却服务于求导、优化和重放等不同目的。

## 练习


练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。

### 概念题

1. 【基础】为什么 `Tensor<2>` 不能保证两个 Tensor 可以相加？列出编译期和运行时
   分别能发现的错误。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【基础】Cargo feature 与 Device 在后端选择中分别承担什么职责？

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>

3. 【基础】为什么 ModuleRecord 不能单独恢复一个任意模型？

<details>
<summary>提示</summary>

运行 `examples/ch07-record-roundtrip`；ONNX/HTTP 另属可选边界。

</details>

4. 【基础】对表达式 $z=(x+y)\times y$，画出前向依赖，并推导
   $\partial z/\partial x$ 和 $\partial z/\partial y$。

<details>
<summary>提示</summary>

回看第 2 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】比较 autodiff tape 与 Fusion IR 的节点生命周期和目标。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

6. 【进阶】为什么向量输出直接调用 backward 等价于使用全 1 根梯度？训练损失通常
   为什么先归约为标量？

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>


### Rust 与 API 题

1. 【基础】修改广播实验，把输入 shape 改成 `[2, 1, 3]` 与 `[1, 4, 1]`。写出
   Tensor 秩和预期结果 shape，再用测试验证。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【基础】在一个需要重复使用 Tensor 的表达式中移除 clone，观察编译器错误。
   解释移动发生在哪里。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

3. 【进阶】为 TinyModel 增加第二个 Linear 层，手工计算并断言参数量。

<details>
<summary>提示</summary>

回看第 2 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【进阶】令 `detached_right = right.detach()` 后再做乘法。验证原 `right`
   没有梯度，而 `detached_right` 因保留 require-grad 意图成为新的可求导
   叶子；再尝试 `set_require_grad(false)` 比较结果。

<details>
<summary>提示</summary>

回看第 2 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】比较 `Device::flex()`、`.autodiff()` 和 `.inner()` 的
   `is_autodiff()`。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

6. 【进阶】调用 `branch_gradient(false)`，断言输出与梯度，并解释为何未执行的
   `* 2` 分支不会出现在 tape 中。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

7. 【进阶】把一个训练函数按 Load/Map、Batch、Model、Loss、Autodiff、
   Optimizer、Evaluate/Save 七个阶段标注输入、输出和可恢复状态；指出
   哪些状态不应只放进 ModuleRecord。

<details>
<summary>提示</summary>

运行 `examples/ch06-training-loop` 并对照第 6 章训练循环节。

</details>


### 源码题

1. 【进阶】在 `DispatchDevice` 中找出哪些变体受 Cargo feature 控制。

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>

2. 【进阶】找到 Flex 的默认 float/int dtype，并与第一章 stack probe 输出比较。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

3. 【进阶】找到 `require_grad` 对非叶子 Tensor 的限制及错误位置。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

4. 【进阶】找到 Module visitor 如何统计 Param，而普通 Tensor 为何不会自动成为
   参数。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

5. 【进阶】在 `burn-backend-tests/tests/autodiff/` 中选择一个广播操作，解释它的
   梯度为何需要归约。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

6. 【进阶】找到 Module visitor 对 `Param` 和普通字段的处理，说明新增一个字段时
   如何判断它应进入 optimizer、ModuleRecord 或 Config。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>


## 延伸阅读

本书固定版本源码中的权威入口：

- `burn/crates/burn-tensor/src/tensor/api/base.rs`
- `burn/crates/burn-tensor/src/tensor/api/float.rs`
- `burn/crates/burn-tensor/src/tensor/api/autodiff.rs`
- `burn/crates/burn-tensor/src/device.rs`
- `burn/crates/burn-dispatch/src/device.rs`
- `burn/crates/burn-core/src/module/`
- `burn/crates/burn-core/src/store/mod.rs`
- `burn/crates/burn-autodiff/src/runtime/`
- `burn/crates/burn-backend-tests/tests/autodiff/`

本版附带的 Burn Book 仍有不少 `Tensor<B, D>` 和泛型 Module 示例。
它可用于理解设计动机，但代码签名必须用以上源码和可编译测试核对。

## 本章系统结论

1. 工作流各阶段携带各自的数据、状态与出错点，不能压成一个模糊的 `train()`。
2. `Tensor` 的编译期秩/类别与运行时 shape、dtype、Device 必须分开看。
3. Module/`Param` 管理可训练状态；autodiff tape 记录实际执行路径上的一阶反向。
4. CPU 上你应观察到广播、参数统计与分支 tape 的梯度行为。
5. GPU 阅读线索：同一 Module API 最终要落到某个 Backend/CubeCL Kernel；中间还隔着 dispatch 与（可选）Fusion。
6. 不能把 autodiff tape、Fusion IR 与 device graph capture 当成同一层实现。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-2-章)。
