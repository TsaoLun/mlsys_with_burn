# 实验：探测执行栈

本实验不追求复杂计算，而是走通全书最短的一条可运行路径：

```text
启用 flex feature
  → Device::flex()
  → DispatchDevice::Flex
  → 后端执行与同步
  → 主机读回结果
```

示例代码在编译期嵌入本书使用的版本说明，因此运行输出会带上写作时的
快照名称。环境与依赖准备见[如何运行本书示例](../running-examples.md)。

## 1. 阅读实验源码

下面代码直接来自 `examples/ch01-stack-probe`：

```rust,ignore
{{#include ../../../examples/ch01-stack-probe/src/lib.rs:example}}
```

完整源码在同一文件前半部分定义了 `StackReport`、`ProbeError`，并用
`snapshot_name()` 读出嵌入的版本名称；这里只展示连接设备和张量执行的
主路径。

需要注意四点：

1. 报告里的 snapshot 名称来自本书写作时固定的版本，方便你对照正文；
2. `Device::flex()` 只在启用 Burn 的 `flex` feature 后存在；
3. `settings()` 和 `is_autodiff()` 观察的是设备能力，而不是 Tensor
   类型参数；
4. `sync()` 提供统一同步接口。对 Flex 来说它很轻量，但异步或缓冲后端
   需要同一语义来等待已提交工作。

## 2. 运行探测

```bash
cargo run -p ch01-stack-probe --locked
```

输出应类似：

```text
snapshot: burn-0.22.0-pre.1
device: Device<Flex(...)>
default float dtype: F32
default int dtype: I32
autodiff enabled: false
observed value after sync: 7
```

设备的 Debug 细节可能随内部实现变化，不应把整个字符串当作稳定 API。
请确认其中包含 `Flex`。默认整数 dtype 也以你本机输出为准。

## 3. 运行测试

```bash
cargo test -p ch01-stack-probe --locked
```

测试会核对：

- snapshot 名称与本书版本一致；
- 设备确实分派到 Flex；
- 默认浮点类型为 F32；
- 同步后读回的值为 7。

## 4. 沿源码追踪

建议按以下顺序在本书固定版本的源码仓库中查找：

1. `burn/crates/burn-tensor/src/device.rs`：公开 `Device`；
2. `burn/crates/burn-dispatch/src/device.rs`：`DispatchDevice::Flex`；
3. `burn/crates/burn-flex/src/backend.rs`：Flex 的 `Backend` 实现；
4. `burn/crates/burn-backend/src/backend/base.rs`：后端共同契约。

不要在第一遍阅读时追踪每个宏展开。先确认公开调用怎样跨过一层边界，再在
第 2、4 章分别进入张量操作和 IR/运行时。

## 5. 思考

如果把根依赖的 `flex` feature 换成 `cuda`，哪些代码保持不变？哪些前提
会改变？至少考虑编译依赖、设备工厂参数、驱动、异步执行和支持的 dtype。
现在不必真的改依赖配置——这个问题用来区分“统一 API”与“统一环境”。

本实验只走通“选中 Flex 并完成一次同步读回”。张量形状、Module 状态、
自动微分、数据管道和训练循环会在后续章节展开。
