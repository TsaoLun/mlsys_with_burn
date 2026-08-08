# 机器学习系统：基于 Burn 与 Rust 的设计和实现

本书面向已经接触过 Rust、希望深入机器学习系统的读者。我们将使用一套
连续技术栈观察系统的不同层次：

```text
模型与训练
    Burn
张量、自动微分、图与融合
    Burn Core / Burn IR
设备无关 Kernel 与编译
    CubeCL
高性能算子
    CubeK
CPU、GPU、Web 与嵌入式运行时
```

Burn 是贯穿全书的可执行案例，不是机器学习系统的全部。分布式通信、
集群调度、存储和服务等主题仍会从框架无关的系统原理出发。

本书是固定 `burn-0.22.0-pre.1` 源码快照的九章候选版，默认采用
CPU 可运行路径（CPU-first）。每章开头的“证据状态”使用统一标签区分
CPU 可运行验证、源码核验、协议/成本模型、可选平台实验和未覆盖能力；
这些是本书的阅读证据分类，不是 Burn 官方能力等级。真实 GPU、NCCL、
ONNX、DDP 和集群控制面不会因为本书使用 Burn 就自动变成已验证事实。

本书与 OpenMLSys v1 的逐文件对应关系和证据标签定义见
[逐文件对照矩阵导读](crosswalk-guide.md)，横向主题比较见
[OpenMLSys 核心主题比较卡](comparison-cards.md)。运行示例的环境与命令见
[如何运行本书示例](running-examples.md)，关键术语见[术语表](glossary.md)。
公式由 mdBook 配置的 MathJax 渲染，浏览器阅读公式时需要访问相应的
CDN 资源；Cargo 依赖可离线复现不代表浏览器资源离线可用。

