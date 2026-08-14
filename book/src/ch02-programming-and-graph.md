# 第 2 章 编程接口与计算图

第 1 章把系统画成从模型接口到设备执行的分层路径。本章打开上半部分：
用户如何表达张量程序，模型如何注册参数，运算依赖如何变成梯度。

OpenMLSys 把「编程接口」和「计算图」分成两章；本书合在一起，因为 eager
执行、Module 状态和反向 tape 本来就是同一套工作流上的三个切面。产业里
对应的是 PyTorch / JAX 这类用户 API，以及「动态 tape / 静态图」之争。

本书所用版本里，用户张量是 `Tensor<D, K>`：`D` 是编译期秩，`K` 是类别，
后端由 `Device` 在运行时选择。若你在别处看到 `Tensor<B, D>`，那是更早
的写法。

## 本章问题

张量程序如何同时表达编译期约束和运行时数据？Module 如何管理参数？
Eager 运算产生的依赖如何形成反向传播所需的动态图？自动微分 tape、融合
IR 和设备 graph capture 为什么不能混为一谈？

## 学习目标

完成本章后，你应该能够：

1. 解释工作流为何需要张量、Module、损失和训练循环等接口；
2. 区分 `Tensor` 的编译期秩/类别与运行时 shape、dtype、Device，并能读回
   字节看内存布局；
3. 描述张量运算如何经 bridge 和 dispatch 到达具体后端；
4. 用 `Module` 和前向方法组织一个最小模型；
5. 用算子、张量边、依赖和控制流解释计算图；
6. 解释 eager 前向与一阶反模式 autodiff tape；
7. 使用 `require_grad`、`backward` 和 `grad` 核对链式法则；
8. 区分 autodiff tape、Burn IR / Fusion 计划与 device graph capture。

改 API、反向规则或分派时先看哪一层，见
[一次调用会经过哪些层](crate-map.md)。

## 先修知识

建议先读第 1 章，并了解向量、矩阵、导数和 Rust 所有权。本章会给出反向
模式的直观推导，不要求先学编译器 IR。

## 本章路线

从完整工作流抽出接口，再依次进入 Tensor/Device、Module、计算图和自动
微分。类型与 IR 一节只建立边界，融合和 JIT 留到第 4 章。实验把广播、
参数注册和梯度连起来，并附一个约百行的迷你反向 tape。

## 小节

1. [从工作流到编程接口](ch02/01-interface-and-workflow.md)
2. [Tensor、Device 与运行时后端](ch02/02-tensor-device-backend.md)
3. [Module、参数与模型状态](ch02/03-module-and-state.md)
4. [计算图的构成与生成](ch02/04-computational-graph.md)
5. [自动微分](ch02/05-autodiff.md)
6. [类型、IR 与调度边界](ch02/06-types-ir-scheduling.md)
7. [实验：张量、Module 与梯度](ch02/07-labs.md)
8. [练习、延伸阅读与来源](ch02/08-exercises-and-sources.md)

第 3 章从 `Device` 向下进入加速器；若你更关心数据和训练，也可以先跳到
第 5、6 章。
