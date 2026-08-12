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

回看第 1 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【基础】为什么“提供统一 Tensor API”不等于“所有后端能力完全一致”？举出
   dtype、融合、同步或部署方面的两个例子。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

3. 【基础】自动微分和算子融合都可能记录操作。它们记录操作的目的有何不同？

<details>
<summary>提示</summary>

运行/阅读 `examples/ch04-fusion-inspector` 与第 4 章 Fusion 节。

</details>

4. 【基础】为什么把数据处理系统直接当成机器学习框架，或把机器学习框架直接当成
   集群调度器，都会遗漏关键抽象？

<details>
<summary>提示</summary>

回看第 1 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】解释 CubeCL 与 CubeK 的职责区别，并说明 Flex 实验为何不经过它们。

<details>
<summary>提示</summary>

见第 3 章 GPU 并行层次节与配图。

</details>

6. 【进阶】为同一个分类模型填写训练、离线推理和在线服务三张负载卡片，比较
   数据供给率、设备内存、吞吐、尾延迟和恢复性约束。

<details>
<summary>提示</summary>

回看第 1 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

7. 【进阶】用有效吞吐和可用内存的预算式找出一个假想系统的首要瓶颈，并说明
   为什么更快的 Kernel 或更多模型副本不一定是正确修复。

<details>
<summary>提示</summary>

回看第 1 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


### 源码题

1. 【进阶】在 `burn/crates/burn/Cargo.toml` 找到 `flex`、`autodiff`、`fusion`、
   `train` 和 `store` feature，画出它们启用的直接 crate。

<details>
<summary>提示</summary>

见第 2 章自动微分节与 `burn-autodiff` 导读清单。

</details>

2. 【进阶】在 `DispatchDevice` 中列出当前 feature 允许的变体。为什么源码中的
   枚举定义不等于你的实验二进制会包含全部变体？

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>

3. 【进阶】找到 Flex 的 `BackendTypes::GraphPrimitive`。它表达了什么能力边界？

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

4. 【进阶】比较仓库根目录版本钉扎与 `burn-onnx/Cargo.toml` 使用的
   revision。只有版本号相同，为什么仍不足以断言 API 兼容？

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>


### 实验题

1. 【基础】在不修改根依赖快照的前提下，为 `StackReport` 增加默认 bool dtype，
   更新测试和实验说明。

<details>
<summary>提示</summary>

回看第 1 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

2. 【基础】启用 `autodiff` feature，新增一个测试比较普通设备与
   `device.autodiff()` 的 `is_autodiff()`，但不要在本章实现反向传播。

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>

3. 【进阶】在有合适设备的机器上新建独立实验，对比 Flex 与另一后端的 Device
   输出和同步行为。记录环境，不要把硬件特定结果写成通用结论。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>


## 延伸阅读

以下路径均相对于本书固定版本的源码仓库（如何获取见
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

回看第 1 章应用与负载小节；范围边界见[范围、证据与对照附录](../appendix-scope-and-evidence.md)。

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
