# 第 2 章 编程接口与计算图

## 本章问题

张量程序如何表达数据、设备和梯度？系统如何记录、转换并执行用户计算？

## 计划内容

- Tensor、形状、数据类型与 Device
- Backend 抽象和运行时设备选择
- Module、参数与模型状态
- 自动微分和反向传播
- Eager 执行、IR 与图级优化
- Rust 类型系统带来的接口设计差异

## 起始实验：张量基础

以下代码直接来自可测试的示例：

```rust
{{#include ../../examples/ch02-tensor-basics/src/lib.rs:example}}
```

运行：

```bash
cargo run -p ch02-tensor-basics
```

## 来源与改编说明

计划参考 OpenMLSys v1 `chapter_programming_interface/`、
`chapter_computational_graph/` 及自动微分相关内容。Python/C++ 扩展和
MindSpore 执行模式将按 Burn 的 Tensor、Device、Autodiff、IR 重写。

