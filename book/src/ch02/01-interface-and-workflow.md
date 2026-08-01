# 从工作流到编程接口

机器学习框架的接口并不是任意 API 的集合。它们来自一个反复出现的工作流：

```text
数据 → 批次 → 模型前向 → 损失 → 梯度 → 参数更新
                         │
                         └→ 指标、调试、保存与评估
```

数据管道和优化器会在第 5、6 章展开，本章关心这条路径如何被程序表达。

## 一条完整工作流需要哪些契约

原作把机器学习工作流拆成数据、模型、损失、优化器、训练、测试和调试。
用 Rust/Burn 重写时，不能把它们压成一个 `train()` 调用；每个阶段都要
定义输入、输出和状态：

1. **Load/Map**：输入是文件、记录或环境样本，输出 typed item；失败可能
   来自解码、版本或缺失字段。
2. **Batch**：输入 item 集合和 Device，输出 batch tensor；失败可能来自
   shape、dtype 和 padding。
3. **Model**：输入 batch 和 Module state，输出 prediction；失败可能来自
   参数、设备或不支持的算子。
4. **Loss**：输入 prediction 和 target，输出标量或可归约 loss；失败可能
   来自 reduction 和标签语义不一致。
5. **Autodiff**：输入 loss 和 tape，输出 gradients；失败可能来自未跟踪
   参数、分支或中间值内存。
6. **Optimizer**：输入 gradients 和 optimizer state，输出新参数与更新
   状态；失败可能来自 ParamId、学习率或设备归属。
7. **Evaluate/Save**：输入验证数据和 model state，输出 metric 或
   ModuleRecord；失败可能来自 train/eval 模式、格式或恢复协议。

这组契约中的“输出/状态”非常重要。一个函数即使返回了正确形状的 Tensor，
也可能没有把参数注册为 `Param`；一个 loss 即使数值下降，也可能因
reduction 不一致而不能与多设备训练比较；一个 ModuleRecord 即使能够
反序列化，也不等于 optimizer 和 sampler 已经恢复。

因此，本书把工作流分到后续章节，而不是在本章复制一个完整框架教程：
第 5 章处理 Load/Map/Batch，第 6 章处理 Autodiff/Optimizer/训练状态，
第 7 章处理 Evaluate/Save/部署，第 8 章把 Load 替换为环境交互。第 2
章负责让读者能看懂这些阶段之间的类型和所有权接口。

## 易用性与执行效率

早期神经网络库通常把数学算子、设备 Kernel 和模型定义紧密绑定。随着模型
规模与硬件种类增长，现代框架逐渐形成前端与后端分离：

- 用户接口描述张量、层和训练逻辑；
- 中间层检查形状、记录依赖或生成 IR；
- 后端为 CPU、GPU 或远程设备实现相同操作语义；
- 运行时管理内存、提交和同步。

Python 生态常通过 C/C++ 扩展连接这些层。Burn 选择 Rust 作为一等前端和
主要实现语言，减少了部分跨语言边界，但没有消除抽象边界。模型作者仍然
不应直接管理每个 Kernel，Backend 作者也不应重新定义高层模型语义。

当内置算子不够用时，扩展路径也应保持分层：

```text
模型/Module API
      │ shape、dtype、device、错误契约
      ▼
Tensor operation / custom Rust function
      │ Backend operation 与 dispatch
      ▼
CubeCL Kernel 或目标后端实现
      │ launch、buffer、同步、编译缓存
      ▼
Runtime / driver
```

原作通过 Pybind11 和 C/C++ custom op 解释“高层接口如何调用低层实现”。
在本书中，Rust/CubeCL 是主线替代，但问题本身没有消失：扩展仍要处理
ABI 或 trait 边界、shape/dtype 校验、设备地址、workspace、异步错误和
版本兼容。第 3、4 章会把其中的 Kernel、launch 和 lowering 展开；本章
只保留接口责任，避免将一个 backend 内部函数冒充稳定的模型 API。

## 工作流中的接口切面

### 数据与批次

数据接口把外部样本转换成具有固定 dtype 和 shape 约定的批次。它需要处理
解析、打乱、并行加载和错误。进入模型之前，批次通常已被转换为 Tensor 并
放到目标 Device。

### 模型与前向方法

模型由可组合的层构成。前向方法接收 Tensor，依次调用层或张量操作并返回
结果。模型对象还要让系统找到所有可训练参数、把它们移动到设备并序列化。
Burn 通过 `Module` 与 `Param` 表达这些能力。

### 损失与梯度

损失把模型输出和目标压缩成用于优化的量。训练需要损失相对于参数的梯度；
用户通常只描述前向过程，系统在执行时记录必要依赖并运行反向传播。

### 参数更新

优化器读取梯度并产生新参数。它可能维护动量、二阶矩估计等状态。自动微分
只负责求导，不等同于优化器，也不负责决定训练何时保存检查点。

### 评估与推理

评估复用模型前向，但不需要记录梯度，某些层的行为也会变化，例如 Dropout。
在固定快照中，这类训练/验证差异主要通过输入 Device 是否启用 autodiff
表达，而不是 Module 内一个通用布尔开关。部署还要恢复权重、选择设备并
处理请求。本章只介绍其边界，完整训练与部署分别留到第 6、7 章。

## 命令式执行与程序变换

Burn 的当前用户路径是命令式（imperative）eager 执行：Rust 语句运行时
立刻发起对应张量操作。自动微分在前向过程中动态记录依赖，而不是要求用户
先提交完整静态图。

这不表示系统没有图或 IR。自动微分 tape、融合优化和设备 graph capture
会出于不同目的记录程序的一部分。重要的是先问“记录是为了什么”，再讨论
它采用哪种表示。

