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

[「Tensor、Device 与运行时后端」](02-tensor-device-backend.md)开头
写明类型只固定秩与类别；把 shape、dtype、Device 逐项排查，各自的
不匹配在哪一刻暴露？shape 兼容的判定规则在同页「广播」小节。

</details>

2. 【基础】Cargo feature 与 Device 在后端选择中分别承担什么职责？

<details>
<summary>提示</summary>

[「Device 与 Dispatch」](02-tensor-device-backend.md)一节区分了
「编译进哪些后端变体」与「运行时选中哪个实例」两件事；问：只改
Device 工厂方法，能用到没编译进程序的后端吗？

</details>

3. 【基础】为什么 ModuleRecord 不能单独恢复一个任意模型？

<details>
<summary>提示</summary>

[「Module、参数与模型状态」](03-module-and-state.md)的 ModuleRecord
流程图显示 `load_record()` 除了 record 还需要一个新 Module；
`examples/ch07-record-roundtrip` 恢复前就先用同一 Config 重建模型。

</details>

4. 【基础】对表达式 $z=(x+y)\times y$，画出前向依赖，并推导
   $\partial z/\partial x$ 和 $\partial z/\partial y$。

<details>
<summary>提示</summary>

[「计算图的构成与生成」](04-computational-graph.md)开篇画的正是它；
仿照[「自动微分」](05-autodiff.md)的带数字推演从根伴随值 1 反推，
留意 y 被加法和乘法同时消费，两条路径的贡献要相加。

</details>

5. 【进阶】比较 autodiff tape 与 Fusion IR 的节点生命周期和目标。

<details>
<summary>提示</summary>

[「tape 的生命周期」](05-autodiff.md)：Step 被 backward 逐个移除、
用完即弃；[「Burn IR 与运行时融合」](../ch04/03-burn-ir-and-fusion.md)
里的计划则可被缓存命中。问：一次性结构与可复用计划各为谁服务？

</details>

6. 【进阶】为什么向量输出直接调用 backward 等价于使用全 1 根梯度？训练损失通常
   为什么先归约为标量？

<details>
<summary>提示</summary>

[「自动微分」](05-autodiff.md)的带数字推演以根伴随值 1 起步；实验
函数 `multiply_with_gradients` 对向量输出直接 backward，全 1 根梯度
等价于先对输出做哪种归约再求导？成本小节解释了标量输出的优势。

</details>


### Rust 与 API 题

1. 【基础】修改广播实验，把输入 shape 改成 `[2, 1, 3]` 与 `[1, 4, 1]`。写出
   Tensor 秩和预期结果 shape，再用测试验证。

<details>
<summary>提示</summary>

广播判定规则在[「广播」](02-tensor-device-backend.md)小节：从尾部
维度对齐，相等或为 1。改 `broadcast_rows_and_columns` 及其测试时，
秩从 2 变 3，`Tensor<2>` 和 `[usize; 2]` 都要跟着改。

</details>

2. 【基础】在一个需要重复使用 Tensor 的表达式中移除 clone，观察编译器错误。
   解释移动发生在哪里。

<details>
<summary>提示</summary>

可在 `multiply_with_gradients` 里去掉 `left.clone()`：乘法按值拿走
`left`，编译器会在后面的 `left.grad(&gradients)` 处报错。对照
[「所有权与 clone」](02-tensor-device-backend.md)解释是谁拿走了值。

</details>

3. 【进阶】为 TinyModel 增加第二个 Linear 层，手工计算并断言参数量。

<details>
<summary>提示</summary>

[「Module、参数与模型状态」](03-module-and-state.md)「Module derive」
给出了 Linear 参数量的口算方法；新层输入维要衔接第一层输出维 2，
并同步更新 `module_registers_parameters_and_preserves_batch_shape`。

</details>

4. 【进阶】令 `detached_right = right.detach()` 后再做乘法。验证原 `right`
   没有梯度，而 `detached_right` 因保留 require-grad 意图成为新的可求导
   叶子；再尝试 `set_require_grad(false)` 比较结果。

<details>
<summary>提示</summary>

把 `detached_leaf_gradient` 当模板，把 detach 施加到 `right` 上；
[「detach 与 inner」](05-autodiff.md)写明固定版本的 detach 保留
require-grad 意图。问：`set_require_grad(false)` 关掉的又是什么？

</details>

5. 【进阶】比较 `Device::flex()`、`.autodiff()` 和 `.inner()` 的
   `is_autodiff()`。

<details>
<summary>提示</summary>

在 `inspect_device_modes` 里补第三个读数即可动手；
[「detach 与 inner」](05-autodiff.md)写明 `device.inner()` 去除的
是 Device 的 autodiff 包装。先预测 `.autodiff().inner()` 再断言。

</details>

6. 【进阶】调用 `branch_gradient(false)`，断言输出与梯度，并解释为何未执行的
   `* 2` 分支不会出现在 tape 中。

<details>
<summary>提示</summary>

期望输出与梯度可对照测试 `autodiff_tape_follows_executed_branch_only`；
解释部分用[「控制流」](04-computational-graph.md)小节的事实：tape
只登记本次真正执行过的算子，未执行的 Rust 语句不会留痕。

</details>

7. 【进阶】把一个训练函数按 Load/Map、Batch、Model、Loss、Autodiff、
   Optimizer、Evaluate/Save 七个阶段标注输入、输出和可恢复状态；指出
   哪些状态不应只放进 ModuleRecord。

<details>
<summary>提示</summary>

[「从工作流到编程接口」](01-interface-and-workflow.md)的七阶段表已
列好输入/输出/卡点，`examples/ch06-training-loop` 是现成的标注对象；
[「训练状态与梯度状态」](03-module-and-state.md)末段给出判断起点。

</details>


### 源码题

1. 【进阶】在 `DispatchDevice` 中找出哪些变体受 Cargo feature 控制。

<details>
<summary>提示</summary>

在 `burn/crates/burn-dispatch/src/device.rs` 的 `DispatchDevice`
枚举定义上逐个找 `#[cfg(feature = ...)]` 属性，和
[「Device 与 Dispatch」](02-tensor-device-backend.md)的变体图互核。

</details>

2. 【进阶】找到 Flex 的默认 float/int dtype，并与第一章 stack probe 输出比较。

<details>
<summary>提示</summary>

沿 `burn/crates/burn-tensor/src/device.rs` 里 `settings()` 的返回
类型找默认 float/int dtype 的来源；结果可与
[「实验：探测执行栈」](../ch01/06-stack-probe-lab.md)打印的输出对照。

</details>

3. 【进阶】找到 `require_grad` 对非叶子 Tensor 的限制及错误位置。

<details>
<summary>提示</summary>

[「叶子、非叶子与 require_grad」](05-autodiff.md)给出 API 约束；到
`burn/crates/burn-tensor/src/tensor/api/autodiff.rs` 里搜
`require_grad`，看哪个分支报错、它如何判定张量已是图的中间结果。

</details>

4. 【进阶】找到 Module visitor 如何统计 Param，而普通 Tensor 为何不会自动成为
   参数。

<details>
<summary>提示</summary>

从 `burn/crates/burn-core/src/module/base.rs` 找 `num_params` 依赖
的 visitor；对照[「Param 与普通 Tensor」](03-module-and-state.md)
想：visitor 识别的是 `Param` 字段类型，还是任何 Tensor 值？

</details>

5. 【进阶】在 `burn-backend-tests/tests/autodiff/` 中选择一个广播操作，解释它的
   梯度为何需要归约。

<details>
<summary>提示</summary>

先重温[「广播」](02-tensor-device-backend.md)里 [3,1] 与 [1,2] 的
梯度归约推演；再在 `burn/crates/burn-backend-tests/tests/autodiff/`
挑一个输入 shape 不同的二元操作，核对梯度断言归到哪个输入的 shape。

</details>

6. 【进阶】找到 Module visitor 对 `Param` 和普通字段的处理，说明新增一个字段时
   如何判断它应进入 optimizer、ModuleRecord 或 Config。

<details>
<summary>提示</summary>

[「参数 visitor 是状态边界」](03-module-and-state.md)列出了新增
字段的三个判断问题；在 `burn/crates/burn-core/src/module/` 里对比
visitor 对 `Param` 与普通字段的处理，再拿 BatchNorm 统计量自测。

</details>


## 延伸阅读

接口与自动微分的论文见附录[参考文献](../references.md#第-2-章-编程接口与计算图)。
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
