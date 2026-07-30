# 第 4 章 AI 编译器与运行时系统

## 本章问题

系统如何在不改变模型语义的前提下变换计算、生成 Kernel，并管理设备上的
资源与执行？

## 计划内容

- 中间表示与编译流水线
- Burn IR 与运行时融合
- CubeCL IR、优化和代码生成
- Kernel 选择、融合与自动调优
- 内存规划、流与异步执行
- AOT、JIT 与动态图方案的边界

## 实验

观察一段张量程序生成的 IR，并对比融合前后的执行与性能。

## 来源与改编说明

计划参考 OpenMLSys v1 `chapter_frontend_and_ir/` 和
`chapter_backend_and_runtime/`。MindSpore/Ascend 专用实现将替换为
Burn Fusion 与 CubeCL 编译运行时案例。

