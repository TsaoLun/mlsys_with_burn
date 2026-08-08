# 对照、证据与范围边界

这一节把本章的地图和 OpenMLSys v1 的导论文件逐项对齐。对照的对象是
固定 revision，而不是某个会变化的在线页面：

- 原作的应用、设计目标、架构和生态材料保留为框架无关的系统问题；
- 本书把实现路径重写为 `Tensor → autodiff → IR/Fusion → Kernel → Runtime`；
- 第 5–7 章继续把数据、训练、artifact 和推理连接成一个可运行 workflow；
- 第 9 章补充控制面，但不把 Burn 的训练数据面说成集群 scheduler。

## 证据分类（本书自定义）

本节使用本书定义的五个证据维度，不是 OpenMLSys 或 Burn 官方评级：
C（Correctness，正确性）、S（Source，源码证据）、R（Runnable，可运行性）、
L（Learning，学习路径）和 E（Engineering，工程可复核性）。

| 维度 | 本章可核验内容 | 不能从本章推出的结论 |
|---|---|---|
| C（正确性） | workload card、系统分层和 Burn/CubeCL/CubeK 职责 | 所有后端能力相同 |
| S（源码） | `pins.toml`、固定源码入口和 OpenMLSys 逐文件对照矩阵 | 最新版本 API |
| R（可运行性） | CPU `ch01-stack-probe` 的 Device/Backend/Tensor 路径 | GPU 性能或网络吞吐 |
| L（学习路径） | 由应用负载连接第 2–9 章 | 本书覆盖 OpenMLSys 全部专题 |
| E（工程复核） | 章节导航、来源文件和许可证入口 | 上游项目官方背书 |

OpenMLSys 的推荐系统、联邦学习、可解释 AI、机器人和机器学习附录不在
本书首版九章正文范围。它们是可追踪的范围差异，不应被“九章”这一数字误读
为对原作全部内容的能力对等性（parity）声明。

## 小练习

为一个在线推荐模型填写训练、离线推理、在线服务和故障恢复四张 workload
card。每张卡都写出输入、输出、状态、吞吐/延迟目标、设备约束和恢复点，
并指出它会进入本书的哪一章。最后列出一个需要真实 GPU、网络或外部系统
才能验证的字段。

完整逐文件对照矩阵、固定 revision 和未纳入文件清单见
[逐文件对照矩阵导读](../crosswalk-guide.md)给出的仓库入口；本节不复制上游图表或
历史硬件数字。
