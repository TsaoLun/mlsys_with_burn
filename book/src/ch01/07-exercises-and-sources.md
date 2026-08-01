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

### 概念题

1. 选择一个你熟悉的机器学习应用，分别列出其计算、数据、硬件和生命周期
   约束。哪些约束无法从模型结构本身看出？
2. 为什么“提供统一 Tensor API”不等于“所有后端能力完全一致”？举出
   dtype、融合、同步或部署方面的两个例子。
3. 自动微分和算子融合都可能记录操作。它们记录操作的目的有何不同？
4. 为什么把数据处理系统直接当成机器学习框架，或把机器学习框架直接当成
   集群调度器，都会遗漏关键抽象？
5. 解释 CubeCL 与 CubeK 的职责区别，并说明 Flex 实验为何不经过它们。
6. 为同一个分类模型填写训练、离线推理和在线服务三张负载卡片，比较
   数据供给率、设备内存、吞吐、尾延迟和恢复性约束。
7. 用有效吞吐和可用内存的预算式找出一个假想系统的首要瓶颈，并说明
   为什么更快的 Kernel 或更多模型副本不一定是正确修复。

### 源码题

1. 在 `burn/crates/burn/Cargo.toml` 找到 `flex`、`autodiff`、`fusion`、
   `train` 和 `store` feature，画出它们启用的直接 crate。
2. 在 `DispatchDevice` 中列出当前 feature 允许的变体。为什么源码中的
   枚举定义不等于你的实验二进制会包含全部变体？
3. 找到 Flex 的 `BackendTypes::GraphPrimitive`。它表达了什么能力边界？
4. 比较根 `pins.toml` 的 Burn revision 与 `burn-onnx/Cargo.toml` 使用的
   revision。只有版本号相同，为什么仍不足以断言 API 兼容？

### 实验题

1. 在不修改根依赖快照的前提下，为 `StackReport` 增加默认 bool dtype，
   更新测试和实验说明。
2. 启用 `autodiff` feature，新增一个测试比较普通设备与
   `device.autodiff()` 的 `is_autodiff()`，但不要在本章实现反向传播。
3. 在有合适设备的机器上新建独立实验，对比 Flex 与另一后端的 Device
   输出和同步行为。记录环境，不要把硬件特定结果写成通用结论。

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
第 2、4 章及 `docs/TERM_GLOSSARY.md` 使用同一套层名。

未迁入：原书以 Python/MindSpore/Ascend 为默认栈的图示与生态叙述。

OpenMLSys 原作及本章改编正文采用 CC BY-NC-SA 4.0。完整署名与许可证见
本书的“许可、来源与独立性声明”和仓库根目录 `NOTICE.md`。

