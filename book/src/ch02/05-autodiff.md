# 自动微分

自动微分（automatic differentiation，AD）把由基本算子组成的程序与这些
算子的导数规则结合，计算精确到机器浮点误差的梯度。

## 与其他求导方式的区别

- **手工求导**容易出错，也无法适应频繁变化的模型；
- **数值微分**用有限差分近似，成本高且受步长与舍入误差影响；
- **符号微分**生成新的代数表达式，可能产生表达式膨胀；
- **自动微分**沿实际程序应用链式法则，复用中间结果。

AD 不是符号代数系统，也不是有限差分。它对每个基本操作实现局部导数，再
按依赖图组合。

## 前向模式与反向模式

设函数有 $n$ 个输入、$m$ 个输出。前向模式随着原计算传播输入方向的导数，
适合输入较少、输出较多的情况；反向模式从输出向输入传播伴随值，适合大量
参数映射到一个标量损失的训练场景。

神经网络常有数百万参数而损失只有一个，因此反向模式是主流。Burn 固定
快照实现一阶反模式自动微分，不支持通过嵌套 autodiff 直接求高阶导。

## 链式法则示例

令

$$
c = a \odot b
$$

其中 $\odot$ 是逐元素乘法。对每个元素：

$$
\frac{\partial c}{\partial a} = b,\qquad
\frac{\partial c}{\partial b} = a
$$

若 backward 从向量 $c$ 以全 1 作为根梯度开始，则 `a` 和 `b` 的梯度分别
就是另一个输入。训练中通常先对向量损失做 `sum` 或 `mean`，得到语义明确
的标量目标。

## Burn 的动态 tape

启用 `autodiff` feature 后：

1. `Device::flex().autodiff()` 用 Autodiff 包装底层 Flex Device；
2. 在该 Device 上创建的浮点 Tensor 可调用 `require_grad()` 标记叶子；
3. 前向算子立即在 Flex 上计算，同时把需要求导的节点登记到动态 tape；
4. `output.backward()` 从输出反向执行已登记步骤，返回 Gradients；
5. `leaf.grad(&gradients)` 取得对应叶子的梯度 Tensor。

```text
Autodiff Device
    │
    ├─ eager forward ─────────► Flex 数值结果
    │
    └─ register nodes/steps ──► autodiff tape
                                  │ backward
                                  ▼
                               Gradients
```

tape 位于 `burn-autodiff`，不是 `burn-ir`。Flex 前向也不会因为启用
autodiff 自动进入 Fusion。

## 叶子、非叶子与 `require_grad`

用户通常在输入或参数等**叶子 Tensor** 上调用 `require_grad()`。中间结果
如果依赖需要梯度的父节点，会自动参与反向传播；把非叶子强行转换成新的
require-grad 叶子不符合当前 API 约束。

`is_require_grad()` 说明该 Tensor 是否要求保留梯度。没有任何输入要求
梯度时，系统可以避免构建不必要的反向链。

## `detach` 与 `inner`

- `tensor.detach()` 切断旧图并形成新叶子；固定快照会保留原来的
  require-grad 意图；
- autodiff Tensor 的 `inner()` 去除自动微分元数据，返回底层数值 Tensor；
- `device.inner()` 去除 Device 的 autodiff 包装。

三个操作不能仅凭名字类推。尤其是 `detach()` 不等于把 Tensor 移回 CPU，
也不等于清除其数值。

## 梯度与参数更新

`backward()` 返回的 Gradients 是一次反向传播的结果容器。读取梯度不会
自动更新 Module 参数；优化器需要消费梯度、更新参数并管理自己的状态。

因此应把以下概念分开：

```text
autodiff tape ──backward──► Gradients
                                 │
优化器状态 + Module 参数 ◄───────┘
```

本章实验只验证梯度数值。参数更新、梯度清理、混合精度和训练循环留到
第 6 章。

