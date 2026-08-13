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

九章正文默认在 CPU 上可运行；真实 GPU、分布式通信、ONNX 端到端服务和
集群控制面需要额外环境，各章实验会标明刻意不做的范围。全书示例基于
Burn `0.22.0-pre.1`。

运行示例见[如何运行本书示例](running-examples.md)，术语见
[术语表](glossary.md)，各章论文与教材出处见[参考文献](references.md)。
与 OpenMLSys 的对照、版本边界和来源总录见附录
[范围、证据与对照](appendix-scope-and-evidence.md) 与
[来源与改编总录](appendix-sources.md)。公式由 MathJax 渲染，浏览器阅读
时需要访问相应的 CDN 资源。
