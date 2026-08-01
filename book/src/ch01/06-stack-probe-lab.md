# 实验：探测执行栈

本实验不追求复杂计算，而是验证全书最重要的可复现链条：

```text
pins.toml
  → 固定 Burn revision 与 Cargo.lock
  → flex feature
  → Device::flex()
  → DispatchDevice::Flex
  → 后端执行与同步
  → 主机读回结果
```

## 1. 校验上游快照

在项目根目录运行：

```bash
make check-upstreams
```

脚本会读取 `pins.toml` 和 `Cargo.lock`，确认项目 Cargo manifest 使用
GitHub 上的固定 Burn revision、没有本地 path 依赖，并核对解析出的
CubeCL/CubeK revision。

这项检查不要求本地存在上游源码镜像。如果 Agent 工作区提供了可选的
`burn/`、`cubecl/` 等只读镜像，还可以运行：

```bash
make check-local-sources
```

该命令额外比较镜像的 origin 和 HEAD。镜像仅用于快速阅读，不能通过
Cargo `path` 或 `[patch]` 参与构建。

## 2. 阅读实验源码

下面代码直接来自 `examples/ch01-stack-probe`：

```rust,ignore
{{#include ../../../examples/ch01-stack-probe/src/lib.rs:example}}
```

完整源码在同一文件前半部分定义了 `StackReport`、`ProbeError`，并用
`snapshot_name()` 从编译时嵌入的 `pins.toml` 提取名称；这里仅展示连接
设备和张量执行的主路径。

需要注意四点：

1. `include_str!` 在编译期嵌入根目录 `pins.toml`，实验报告因而携带写作
   快照名称；
2. `Device::flex()` 只在启用 Burn 的 `flex` feature 后存在；
3. `settings()` 和 `is_autodiff()` 观察的是设备能力，而不是 Tensor
   类型参数；
4. `sync()` 提供统一同步接口。对 Flex 来说它很轻量，但异步或缓冲后端
   需要同一语义来等待已提交工作。

## 3. 运行探测

```bash
cargo run -p ch01-stack-probe
```

固定快照下，输出应类似：

```text
snapshot: burn-0.22.0-pre.1
device: Device<Flex(...)>
default float dtype: F32
default int dtype: I32
autodiff enabled: false
observed value after sync: 7
```

设备的 Debug 细节可能随内部实现变化，不应把整个字符串作为稳定 API。
实验只断言它包含 `Flex`。默认整数 dtype 也应由实际输出观察，而不是在
正文中永久写死。

## 4. 运行测试

```bash
cargo test -p ch01-stack-probe
```

测试同时验证：

- snapshot 名称与写作周期一致；
- 设备确实分派到 Flex；
- 默认浮点类型为 F32；
- 普通 Flex 设备没有启用梯度跟踪；
- Tensor 经后端执行、同步和读回后仍为 `7.0`；
- 编译时嵌入的 pin 包含教材 Burn revision。

## 5. 沿源码追踪

建议按以下顺序在固定上游中查找：

1. `burn/crates/burn-tensor/src/device.rs`：公开 `Device`；
2. `burn/crates/burn-dispatch/src/device.rs`：`DispatchDevice::Flex`；
3. `burn/crates/burn-flex/src/backend.rs`：Flex 的 `Backend` 实现；
4. `burn/crates/burn-backend/src/backend/base.rs`：后端共同契约。

不要在第一遍阅读时追踪每个宏展开。先确认公开调用怎样跨过一层边界，再在
第 2、4 章分别进入张量操作和 IR/运行时。

## 6. 思考

如果把根依赖的 `flex` feature 换成 `cuda`，哪些代码保持不变？哪些前提
会改变？至少考虑编译依赖、设备工厂参数、驱动、异步执行和支持的 dtype。
现在不必真的修改固定配置，这个问题旨在区分“统一 API”与“统一环境”。

