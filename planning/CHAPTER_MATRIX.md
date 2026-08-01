# 章节矩阵

本表是内容范围的入口，不代表允许直接复制原文。逐文件对照、处理状态、
证据级别和范围差异见
[`comparison/openmlsys-v1-crosswalk.md`](comparison/openmlsys-v1-crosswalk.md)；
实际写作时仍需在每章“来源与改编说明”中列出精确文件。

| 新章 | OpenMLSys v1 主要来源 | Burn 技术主线 | 改写强度 |
|---|---|---|---|
| 1 导论 | `chapter_introduction` | Burn 分层、Backend 生态 | 中 |
| 2 编程接口与计算图 | `chapter_programming_interface`、`chapter_computational_graph`、自动微分相关内容 | Tensor、Device、Module、Autodiff、IR | 高 |
| 3 AI 加速器与编程 | `chapter_accelerator` | GPU 基础、CubeCL、CubeK | 高 |
| 4 AI 编译器与运行时 | `chapter_frontend_and_ir`、`chapter_backend_and_runtime` | burn-ir、Fusion、CubeCL IR/opt/runtime | 高 |
| 5 数据处理系统 | `chapter_data_processing` | burn-dataset、迭代器与流水线 | 中 |
| 6 训练系统 | `chapter_distributed_training` | burn-train、优化器、多设备与通信 | 高 |
| 7 模型服务 | `chapter_model_deployment` | burn-onnx、Record、Remote、WASM/no_std | 高 |
| 8 强化学习系统 | `chapter_reinforcement_learning` | burn-rl、环境交互与采样管线 | 高 |
| 9 大规模 GPU 集群 | 分布式训练中的集群内容及新增资料 | 调度、遥测、容错；Burn 仅作工作负载案例 | 高 |

## 暂不纳入主线

OpenMLSys v1 的推荐系统、联邦学习、可解释 AI、机器人和机器学习附录不
进入首版九章主线。完整文件清单和排除原因见 crosswalk。稳定版之后再评估
为附录或专题，避免在基础路径尚未完成前扩张。

## 每章必须回答

1. 该主题解决了什么系统问题？
2. 关键抽象与成本模型是什么？
3. Burn 技术栈在哪一层实现它？
4. 固定源码快照中有哪些能力与限制？
5. Rust 的类型、所有权或并发模型带来了什么不同？
6. 读者能通过什么实验观察该机制？

