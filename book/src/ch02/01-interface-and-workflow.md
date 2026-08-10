# 从工作流到编程接口

机器学习框架的接口并不是任意 API 的集合。它们来自一个反复出现的工作流：

```text
数据 → 批次 → 模型前向 → 损失 → 梯度 → 参数更新
                         │
                         └→ 指标、调试、保存与评估
```

数据管道和优化器会在第 5、6 章展开，本章关心这条路径如何被程序表达。

## 接口简史：为什么会分成高层与低层

早期框架往往把“定义网络”和“调用某块加速器 Kernel”写在同一层：改设备
就要改模型代码，换后端也要换调用习惯。随后出现几条反复出现的边界：

| 阶段（概念） | 用户看到什么 | 系统隐藏什么 |
|---|---|---|
| 算子库时代 | 显式调用卷积/矩阵乘 | 设备指针与 Kernel 细节仍常泄漏 |
| 高层 Module API | 层、参数、`forward` | 分派、内存与同步 |
| 动态图 / tape | 命令式控制流 + 自动求导 | 记录哪些依赖、何时释放中间值 |
| 编译 / Fusion | 同一表达式可被改写 | Pass、IR、缓存与 Runtime |
| 部署 artifact | 权重与拓扑快照 | 服务队列、版本与设备选择 |

Burn 站在这条演进线上：用户侧是 `Tensor` / `Module` / `Device`；执行侧
经 dispatch 到达 Flex 或 CubeCL 系后端；真正的 GPU Kernel 还要再经过
第 3、4 章的 Runtime 与 launch。读 API 时先问“这一层替你挡住了什么”，
再问“错误会在哪一层才暴露”。

## 一条完整工作流有哪些阶段

原作把机器学习工作流拆成数据、模型、损失、优化器、训练、测试和调试。
用 Rust/Burn 重写时，不宜压成一个笼统的 `train()`：每个阶段都携带自己的
数据、状态和常见出错点。

| 阶段 | 接收什么 | 产生什么 | 常见卡点 |
|---|---|---|---|
| Load/Map | 文件、记录或环境样本 | 类型化的样本 item | 解码、版本、缺失字段 |
| Batch | item 集合与 Device | batch tensor | shape、dtype、padding |
| Model | batch 与 Module 状态 | prediction | 参数、设备、不支持的算子 |
| Loss | prediction 与 target | 标量或可归约 loss | reduction、标签语义 |
| Autodiff | loss 与 tape | gradients | 未跟踪参数、分支、中间值内存 |
| Optimizer | gradients 与优化器状态 | 新参数与更新状态 | ParamId、学习率、设备归属 |
| Evaluate/Save | 验证数据与模型状态 | metric 或 ModuleRecord | train/eval 模式、格式、恢复协议 |

“产生什么”里往往还带着**可恢复状态**，这一点比“函数返回了正确形状”
更重要：Tensor 形状对了，参数也可能没注册为 `Param`；loss 数值下降了，
reduction 不一致时仍不能和多设备训练比较；ModuleRecord 能反序列化，
也不等于 optimizer 和 sampler 已经恢复。

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
在本书中，Rust/CubeCL 是本书的实现路径替代，但问题本身没有消失：扩展仍要处理
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
在本版中，这类训练/验证差异主要通过输入 Device 是否启用 autodiff
表达，而不是 Module 内一个通用布尔开关。部署还要恢复权重、选择设备并
处理请求。本章只介绍其边界，完整训练与部署分别留到第 6、7 章。

## 命令式执行与程序变换

Burn 的当前用户路径是命令式（imperative）eager 执行：Rust 语句运行时
立刻发起对应张量操作。自动微分在前向过程中动态记录依赖，而不是要求用户
先提交完整静态图。

这不表示系统没有图或 IR。自动微分 tape、融合优化和设备 graph capture
会出于不同目的记录程序的一部分。重要的是先问“记录是为了什么”，再讨论
它采用哪种表示。

## 从 Module API 到 GPU Kernel 隔着哪些层

一次 `module.forward(x)` 在语义上是张量表达式；在 GPU 上真正跑起来，
中间至少隔着：

```text
Module / Tensor API
      → Device 选择与 burn-dispatch
      → Backend op（Flex 或 CubeCL 桥）
      → （可选）Fusion 计划 / CubeCL IR
      → CubeCL Runtime（CPU / WGPU / CUDA / HIP…）
      → Kernel launch → 设备完成 → host read/sync
```

本章实验默认在 Flex CPU 上运行，是为了把类型、广播、Module 状态和 tape
语义看清楚。设备与 Runtime 地图见第 1 章；拓扑与多 Runtime 见第 3 章；
Pass 与 stream 见第 4 章。不要把“Tensor API 写对了”读成“已经测过某张
GPU 的吞吐”。

## 产业对照（概念对齐，不是性能对等）

| 本书 / Burn·CubeCL | 常见产业说法 | 对齐点 | 不要外推 |
|---|---|---|---|
| `Tensor` + `Device` | PyTorch Tensor + device | 统一用户 API、运行时选设备 | API 相似 ≠ 算子集/性能相同 |
| autodiff tape | autograd tape | 按实际路径记依赖 | 不是静态整图 |
| Fusion / CubeCL IR | TorchInductor / XLA 等编译栈（概念） | 表达式可被改写再执行 | 不能直接比墙钟 |
| CubeCL Runtime | CUDA runtime / 图形 API | launch、buffer、同步 | 完成边界因栈而异 |

