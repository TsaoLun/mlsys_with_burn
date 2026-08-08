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
（正文小节、示例 crate 或固定源码路径），不提供完整答案；挑战题常涉及
`可选平台实验` 或开放设计，不在默认 CPU CI 中验证。

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

用 `pins.toml` 固定 revision 打开对应 Cargo/源码路径。

</details>

3. 【进阶】找到 Flex 的 `BackendTypes::GraphPrimitive`。它表达了什么能力边界？

<details>
<summary>提示</summary>

在固定 revision 源码中按章节末“源码入口”定位，勿跟 online main。

</details>

4. 【进阶】比较根 `pins.toml` 的 Burn revision 与 `burn-onnx/Cargo.toml` 使用的
   revision。只有版本号相同，为什么仍不足以断言 API 兼容？

<details>
<summary>提示</summary>

用 `pins.toml` 固定 revision 打开对应 Cargo/源码路径。

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

用 `pins.toml` 固定 revision 打开对应 Cargo/源码路径。

</details>

3. 【进阶】在有合适设备的机器上新建独立实验，对比 Flex 与另一后端的 Device
   输出和同步行为。记录环境，不要把硬件特定结果写成通用结论。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>


## 延伸阅读

以下路径均相对于本地固定上游：

- `burn/crates/burn/src/lib.rs`：Burn 能力总览与限制；
- `burn/crates/burn-backend/src/backend/base.rs`：Backend 设计契约；
- `burn/crates/burn-tensor/src/device.rs`：0.22 Device API；
- `burn/crates/burn-dispatch/src/device.rs`：运行时分派变体；
- `cubecl/README.md`：CubeCL 的并行模型、IR 与多后端目标；
- `cubek/README.md`：CubeK 算法范围；
- `burn-onnx/SUPPORTED-ONNX-OPS.md`：固定导入器的算子边界。

在线 Burn Book 可以辅助理解设计动机，但遇到 API 差异时，以 `pins.toml`
固定的源码为准。

## 来源与改编说明

本章改编并重组了 OpenMLSys v1 以下文件：

- `chapter_introduction/index.md`
- `chapter_introduction/applications.md`
- `chapter_introduction/design.md`
- `chapter_introduction/architecture.md`
- `chapter_introduction/ecosystem.md`
- `chapter_introduction/readers.md`

保留的核心思想包括机器学习应用分类、框架六类设计目标、从接口到硬件的
系统分层和框架中心生态。主要修改包括：

- 将应用枚举改写为计算、数据、硬件与生命周期负载分析；
- 删除以 Python、MindSpore、Ascend 为默认栈的表述；
- 用 Burn 0.22 的 Device/Dispatch 架构映射框架层次；
- 增加 CubeCL、CubeK、burn-onnx 的职责与版本边界；
- 按本书九章重写范围和阅读路径；
- 新增固定源码快照方法、实验与练习。

本章没有复用 OpenMLSys 的 `framework-architecture.png` 和
`system-ecosystem.png` 图面；文本架构图是基于通用分层思想重新设计，并与
第 2、4 章与[术语表](../glossary.md)使用同一套层名。

未迁入：原书以 Python/MindSpore/Ascend 为默认栈的图示与生态叙述。

OpenMLSys 原作及本章改编正文采用 CC BY-NC-SA 4.0。完整署名与许可证见
本书的“许可、来源与独立性声明”和仓库根目录 `NOTICE.md`。

