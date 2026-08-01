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

本书是固定 `burn-0.22.0-pre.1` 源码快照的九章候选版，默认路径采用
CPU-first 实验。每章都区分固定源码核验、CPU 可运行验证、框架无关协议
模型、可选平台实验和明确未覆盖能力；真实 GPU、NCCL、ONNX、DDP 和集群
控制面不会因为本书使用 Burn 就自动变成已验证事实。

发布审计入口位于项目的 `planning/comparison/`、`release.toml` 和
`tools/check_release.py`。本书公式由 mdBook MathJax 渲染，Cargo 的
offline gate 只保证源码和依赖可离线构建，不保证浏览器访问 CDN 资源。

