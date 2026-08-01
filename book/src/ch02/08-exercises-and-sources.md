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

### 概念题

1. 为什么 `Tensor<2>` 不能保证两个 Tensor 可以相加？列出编译期和运行时
   分别能发现的错误。
2. Cargo feature 与 Device 在后端选择中分别承担什么职责？
3. 为什么 ModuleRecord 不能单独恢复一个任意模型？
4. 对表达式 $z=(x+y)\times y$，画出前向依赖，并推导
   $\partial z/\partial x$ 和 $\partial z/\partial y$。
5. 比较 autodiff tape 与 Fusion IR 的节点生命周期和目标。
6. 为什么向量输出直接调用 backward 等价于使用全 1 根梯度？训练损失通常
   为什么先归约为标量？

### Rust 与 API 题

1. 修改广播实验，把输入 shape 改成 `[2, 1, 3]` 与 `[1, 4, 1]`。写出
   Tensor 秩和预期结果 shape，再用测试验证。
2. 在一个需要重复使用 Tensor 的表达式中移除 clone，观察编译器错误。
   解释移动发生在哪里。
3. 为 TinyModel 增加第二个 Linear 层，手工计算并断言参数量。
4. 令 `detached_right = right.detach()` 后再做乘法。验证原 `right`
   没有梯度，而 `detached_right` 因保留 require-grad 意图成为新的可求导
   叶子；再尝试 `set_require_grad(false)` 比较结果。
5. 比较 `Device::flex()`、`.autodiff()` 和 `.inner()` 的
   `is_autodiff()`。
6. 调用 `branch_gradient(false)`，断言输出与梯度，并解释为何未执行的
   `* 2` 分支不会出现在 tape 中。
7. 把一个训练函数按 Load/Map、Batch、Model、Loss、Autodiff、
   Optimizer、Evaluate/Save 七个阶段标注输入、输出和可恢复状态；指出
   哪些状态不应只放进 ModuleRecord。

### 源码题

1. 在 `DispatchDevice` 中找出哪些变体受 Cargo feature 控制。
2. 找到 Flex 的默认 float/int dtype，并与第一章 stack probe 输出比较。
3. 找到 `require_grad` 对非叶子 Tensor 的限制及错误位置。
4. 找到 Module visitor 如何统计 Param，而普通 Tensor 为何不会自动成为
   参数。
5. 在 `burn-backend-tests/tests/autodiff/` 中选择一个广播操作，解释它的
   梯度为何需要归约。
6. 找到 Module visitor 对 `Param` 和普通字段的处理，说明新增一个字段时
   如何判断它应进入 optimizer、ModuleRecord 或 Config。

## 延伸阅读

固定上游中的权威入口：

- `burn/crates/burn-tensor/src/tensor/api/base.rs`
- `burn/crates/burn-tensor/src/tensor/api/float.rs`
- `burn/crates/burn-tensor/src/tensor/api/autodiff.rs`
- `burn/crates/burn-tensor/src/device.rs`
- `burn/crates/burn-dispatch/src/device.rs`
- `burn/crates/burn-core/src/module/`
- `burn/crates/burn-core/src/store/mod.rs`
- `burn/crates/burn-autodiff/src/runtime/`
- `burn/crates/burn-backend-tests/tests/autodiff/`

固定快照附带的 Burn Book 仍有不少 `Tensor<B, D>` 和泛型 Module 示例。
它可用于理解设计动机，但代码签名必须用以上源码和可编译测试核对。

## 来源与改编说明

本章改编并重组 OpenMLSys v1：

### 编程接口

- `chapter_programming_interface/index.md`
- `chapter_programming_interface/development_history.md`
- `chapter_programming_interface/ml_workflow.md`
- `chapter_programming_interface/neural_network_layer.md`
- `chapter_programming_interface/ml_programming_paradigm.md`

`c_python_interaction.md` 提供“高层接口与底层实现存在边界”的背景。本章
保留这一系统问题，改用 Rust trait、Device dispatch、CubeCL Kernel 和
显式错误/所有权边界解释扩展路径；没有复用其 Pybind11、MindSpore 或 CUDA
示例。自定义 Kernel 的 launch、lowering 和 Runtime 细节放到第 3、4 章。

### 计算图

- `chapter_computational_graph/background_and_functionality.md`
- `chapter_computational_graph/components_of_computational_graph.md`
- `chapter_computational_graph/generation_of_computational_graph.md`
- `chapter_computational_graph/schedule_of_computational_graph.md`

本章保留图表示、依赖、控制流、静动态图与拓扑调度思想，删除 TensorFlow 1
和 MindSpore 专用 API，并把数据流水线、模型并行后移。补全时增加了拓扑序
小例子、图外控制流与循环展开/循环依赖区分，以及两分支 autodiff 实验；
统一用语见 `docs/TERM_GLOSSARY.md`。

未迁入：原书长控制流教程、`tf.cond`/`while_loop` API 走读和框架专用训练
代码；完整训练工作流已用输入/输出/状态契约建立地图，数据与训练执行仍
分别后移第 5/6 章。

### 自动微分、类型与 IR

- `chapter_frontend_and_ir/ad.md`
- `chapter_frontend_and_ir/intermediate_representation.md`
- `chapter_frontend_and_ir/type_system_and_static_analysis.md`

本章保留求导方法、前向/反向模式和 IR 的通用定义；MindIR、框架前端 pass
和 MLIR 深入内容后移第 4 章。新增 Burn 0.22 Device/autodiff 动态 tape、
Rust 类型分工和全部可运行示例。

本章没有复制 OpenMLSys 图面。完整逐文件映射见
`planning/chapter-sources/ch02.md`。OpenMLSys 原作和本章改编正文采用
CC BY-NC-SA 4.0，原创 Rust 示例采用 MIT OR Apache-2.0。

