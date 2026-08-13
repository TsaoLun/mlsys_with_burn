# 练习、延伸阅读与来源

## 小结

机器学习系统把模型、数据和硬件连接成完整生命周期。应用类别本身不足以
决定系统设计，还要分析计算、数据、硬件和运行目标。现代框架通常需要模型
编程、自动微分、数据、训练部署、硬件加速和分布式六类能力，但框架并不
等于数据平台、服务系统和 GPU 集群的总和。

在本书固定的 Burn 0.22 快照中，`Tensor<D, K>` 使用 Device 选择运行时
后端，经 burn-dispatch 到达 Flex 或 CubeCL 等具体实现。CubeCL 提供
Kernel 语言、IR 和运行时，CubeK 提供建立在其上的高性能算子；burn-onnx
则是有独立版本关系的模型导入项目。

## 练习


练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。

### 概念题

1. 【基础】选择一个你熟悉的机器学习应用，分别列出其计算、数据、硬件和生命周期
   约束。哪些约束无法从模型结构本身看出？

<details>
<summary>提示</summary>

按[「机器学习应用与系统负载」](01-applications-and-loads.md)的四组
提问逐项作答，或直接填该节的四行负载卡片。判断「能否从模型结构看出」
时，想想尾延迟目标、数据来源和恢复点写在模型定义的哪个位置。

</details>

2. 【基础】为什么“提供统一 Tensor API”不等于“所有后端能力完全一致”？举出
   dtype、融合、同步或部署方面的两个例子。

<details>
<summary>提示</summary>

素材在[「Burn 技术栈」](04-burn-stack.md)的「分派与后端契约」与
「能力边界」两段：融合路径、量化支持都因后端而异；同步的例子是
[「实验：探测执行栈」](06-stack-probe-lab.md)里 `sync()` 对 Flex
很轻量、对异步后端才是真正等待点的注解。

</details>

3. 【基础】自动微分和算子融合都可能记录操作。它们记录操作的目的有何不同？

<details>
<summary>提示</summary>

[「从编程接口到硬件」](03-system-architecture.md)的「张量执行与计算
表示」把两种记录分开：tape 记录反向依赖，Fusion 把操作注册为 Burn IR
再搜索执行计划。问自己：去掉各自的记录，哪个丢正确性，哪个只丢性能？

</details>

4. 【基础】为什么把数据处理系统直接当成机器学习框架，或把机器学习框架直接当成
   集群调度器，都会遗漏关键抽象？

<details>
<summary>提示</summary>

[「机器学习系统的设计目标」](02-design-goals.md)末尾的系统层次表是
答案骨架：第三列写着每类系统需要外部补充的能力。为两种混用各找出
被遗漏的条目，比如张量语义与梯度、调度与集群治理。

</details>

5. 【进阶】解释 CubeCL 与 CubeK 的职责区别，并说明 Flex 实验为何不经过它们。

<details>
<summary>提示</summary>

[「Burn 技术栈」](04-burn-stack.md)分别用两节介绍 CubeCL（语言、IR、
运行时）与 CubeK（其上的高性能算子），`cubecl/README.md` 与
`cubek/README.md` 的自述可对照；Flex 的去向写在「分派与后端契约」末段。

</details>

6. 【进阶】为同一个分类模型填写训练、离线推理和在线服务三张负载卡片，比较
   数据供给率、设备内存、吞吐、尾延迟和恢复性约束。

<details>
<summary>提示</summary>

模板在[「机器学习应用与系统负载」](01-applications-and-loads.md)的
负载卡片一节：训练要计入梯度、优化器状态与恢复点，在线服务不再需要
梯度却新增尾延迟与失败重试。逐行问：换一张卡，这一行为什么会变？

</details>

7. 【进阶】用有效吞吐和可用内存的预算式找出一个假想系统的首要瓶颈，并说明
   为什么更快的 Kernel 或更多模型副本不一定是正确修复。

<details>
<summary>提示</summary>

仿照[「机器学习应用与系统负载」](01-applications-and-loads.md)中
负载卡片一节的两条预算式填一组数字，看 min 卡在哪一项。再问：更快
的 Kernel 或更多副本各自抬高哪一项？副本又会挤占可用内存的哪一项？

</details>


### 源码题

1. 【进阶】在 `burn/crates/burn/Cargo.toml` 找到 `flex`、`autodiff`、`fusion`、
   `train` 和 `store` feature，画出它们启用的直接 crate。

<details>
<summary>提示</summary>

按 `pins.toml` 检出仓库后读 `[features]` 表：等号右边混着依赖名与
feature 引用（如 `train` 还引用 `optim`、`dataset`），间接项要再追
一层；[「Burn 技术栈」](04-burn-stack.md)「可组合能力」一节可核对。

</details>

2. 【进阶】在 `DispatchDevice` 中列出当前 feature 允许的变体。为什么源码中的
   枚举定义不等于你的实验二进制会包含全部变体？

<details>
<summary>提示</summary>

打开本章列出的 `burn/crates/burn-dispatch/src/device.rs`，每个变体
上方都有 `#[cfg(feature = ...)]` 属性；再对照根 `Cargo.toml` 里 burn
依赖实际启用的 feature，想想条件编译在编译期会留下哪些变体。

</details>

3. 【进阶】找到 Flex 的 `BackendTypes::GraphPrimitive`。它表达了什么能力边界？

<details>
<summary>提示</summary>

[「实验：探测执行栈」](06-stack-probe-lab.md)「沿源码追踪」的清单里
有 `burn/crates/burn-flex/src/backend.rs`：找到该关联类型被赋成哪个
类型，类型名本身就在回答问题。再联系「图捕获因后端而异」这句边界。

</details>

4. 【进阶】比较仓库根目录版本钉扎与 `burn-onnx/Cargo.toml` 使用的
   revision。只有版本号相同，为什么仍不足以断言 API 兼容？

<details>
<summary>提示</summary>

对照根目录 `pins.toml` 与 `burn-onnx/Cargo.toml` 各自 pin 的 Burn
commit；[「Burn 技术栈」](04-burn-stack.md)「训练、状态与模型交换」
一段指出两者不同。问自己：预发布期一个版本号对应多少个 commit？

</details>


### 实验题

1. 【基础】在不修改根依赖快照的前提下，为 `StackReport` 增加默认 bool dtype，
   更新测试和实验说明。

<details>
<summary>提示</summary>

照 `examples/ch01-stack-probe/src/lib.rs` 里 `float_dtype` 的写法在三处
扩展：`StackReport` 字段、`probe_execution_stack` 读取 `settings()`、
测试断言；布尔项的命名可沿本章列出的
`burn/crates/burn-tensor/src/device.rs` 找到。

</details>

2. 【基础】启用 `autodiff` feature，新增一个测试比较普通设备与
   `device.autodiff()` 的 `is_autodiff()`，但不要在本章实现反向传播。

<details>
<summary>提示</summary>

根 `Cargo.toml` 的 burn 依赖已带 `autodiff` feature；照
`reports_pinned_flex_execution` 的写法对 `Device::flex()` 与其
`.autodiff()` 包装各断言一次 `is_autodiff()`；两种设备为何不能用
相等性区分，见[「Burn 技术栈」](04-burn-stack.md)「可组合能力」。

</details>

3. 【进阶】在有合适设备的机器上新建独立实验，对比 Flex 与另一后端的 Device
   输出和同步行为。记录环境，不要把硬件特定结果写成通用结论。

<details>
<summary>提示</summary>

以 `examples/ch01-stack-probe` 为模板换用
[「Burn 技术栈」](04-burn-stack.md)里的其他 Device 工厂方法，前置
环境见[如何运行本书示例](../running-examples.md)。观察两点：device
字符串的分派变体，以及 `sync()` 从轻量调用变成真正的等待点。

</details>


## 延伸阅读

原理与产业背景的论文见附录[参考文献](../references.md#第-1-章-导论)。
源码阅读入口如下，路径均相对于本书固定版本的源码仓库（如何获取见
[如何运行本书示例](../running-examples.md) 的「阅读固定源码」）：

- `burn/crates/burn/src/lib.rs`：Burn 能力总览与限制；
- `burn/crates/burn-backend/src/backend/base.rs`：Backend 设计契约；
- `burn/crates/burn-tensor/src/device.rs`：0.22 Device API；
- `burn/crates/burn-dispatch/src/device.rs`：运行时分派变体；
- `cubecl/README.md`：CubeCL 的并行模型、IR 与多后端目标；
- `cubek/README.md`：CubeK 算法范围；
- `burn-onnx/SUPPORTED-ONNX-OPS.md`：固定导入器的算子边界。

在线 Burn Book 可以辅助理解设计动机，但遇到 API 差异时，以本书固定版本
的源码与示例为准。


### 综合小练习

1. 【进阶】为一个在线推荐模型填写训练、离线推理、在线服务和故障恢复四张
   workload card。每张卡写出输入、输出、状态、吞吐/延迟目标、设备约束和
   恢复点，并指出它会进入本书的哪一章。最后列出一个需要真实 GPU、网络或
   外部系统才能验证的字段。

<details>
<summary>提示</summary>

卡片四行模板在[「机器学习应用与系统负载」](01-applications-and-loads.md)，
「进入哪一章」可对照该节的生命周期路径图；哪些字段需要真实 GPU、
网络或外部系统才能验证，判定口径见
[范围、证据与对照附录](../appendix-scope-and-evidence.md)。

</details>

## 本章系统结论

1. 机器学习系统要把应用负载拆成计算、数据、硬件与生命周期约束，而不是只谈模型结构。
2. 现代框架需要分层：用户接口 → 执行/微分 → IR/融合 → Kernel → Runtime。
3. Burn 0.22 用 `Tensor<D,K>` + `Device` 选择后端；Flex 是默认 CPU 路径，不经过 CubeCL。
4. CPU 上你应已跑通 `ch01-stack-probe`：选中 Flex、同步并读回结果。
5. GPU/多后端阅读时对照：`Device::wgpu` / `Device::cuda`、dispatch 变体，以及后文 CubeCL Runtime。
6. 不能从一次 CPU probe 推出 GPU 吞吐、分布式或全部后端能力一致。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-1-章)。
