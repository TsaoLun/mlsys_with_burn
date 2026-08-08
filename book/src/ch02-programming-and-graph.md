# 第 2 章 编程接口与计算图

第 1 章把机器学习系统画成从模型接口到设备执行的分层路径。本章进入这条
路径的上半部分：用户如何表达张量程序，模型如何注册参数，系统又如何记录
运算依赖并计算梯度。

固定快照中的 Burn 0.22 正在经历一次重要 API 转变。旧版本常将 Backend
作为 `Tensor<B, D>` 的类型参数；本书使用的源码已经采用
`Tensor<D, K>`，由 `Device` 在运行时选择后端。这个变化会贯穿本章，
也说明为什么代码必须以 `pins.toml` 对应源码为准。

## 本章问题

张量程序如何同时表达编译期约束和运行时数据？Module 如何管理参数与
状态？Eager 运算产生的依赖如何形成反向传播所需的动态图？自动微分 tape、
融合 IR 和设备 graph capture 为什么不能混为一谈？

## 学习目标

完成本章后，你应该能够：

1. 解释机器学习工作流为何需要张量、Module、损失和训练循环等接口；
2. 区分 `Tensor` 的编译期秩/类别与运行时 shape、dtype、Device；
3. 描述张量运算如何经 bridge 和 dispatch 到达具体 Backend；
4. 使用 `Module`、参数化层和前向方法组织一个最小模型；
5. 用算子、张量边、依赖和控制流解释计算图；
6. 解释 Burn 的 eager 前向与一阶反模式 autodiff tape；
7. 使用 `require_grad`、`backward` 和 `grad` 验证链式法则；
8. 区分 autodiff tape、Burn IR / Fusion 计划与 device graph capture，
   并用分支实验说明 tape 只记录实际路径。

## 先修知识

建议先阅读第 1 章，并了解向量、矩阵、导数和 Rust 所有权的基础概念。
本章会给出反向模式的直观推导，不要求预先掌握编译器 IR。

## 本章路线

我们先从完整工作流抽取编程接口，再依次进入 Tensor/Device、Module、
计算图和自动微分。类型与 IR 一节只建立边界，融合、编译和运行时优化留到
第 4 章。最后的 CPU 实验把张量广播、参数注册和梯度计算连接起来。

## 证据状态

以下标签是本书的阅读证据分类，不代表 Burn 官方能力等级；完整定义见
[逐文件对照矩阵导读](crosswalk-guide.md)。

- `CPU 可运行验证`：Tensor、Module、autodiff 和分支 tape 实验；
- `源码核验`：`Tensor`/`Device`/`Module`、参数状态与一阶 autodiff；
- `协议/成本模型`：workflow 输入/输出/状态/错误契约；
- `可选平台实验`：完整静态图 runtime、device graph capture 与跨设备训练；
- `未覆盖`：把 autodiff tape、Fusion IR 和 device graph capture
  当作同一实现的结论。

