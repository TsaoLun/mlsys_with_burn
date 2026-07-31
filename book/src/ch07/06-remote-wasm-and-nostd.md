# 7.6 Remote、WASM/no_std 与部署边界

## Remote 执行的真正含义

Remote backend 不是“把一个模型上传到服务器并自动暴露 API”。它把
Tensor operation 的执行位置移到 compute peer。客户端仍然创建
`Device` 和 tensor，Remote client/router 把操作提交给 peer，peer 用自己
拥有的具体 backend 执行，再在需要时返回 tensor data。

```text
client process
  Device::remote_iroh(...)
       │ operations / tensor transfers
       ▼
compute peer
  RemoteServerBuilder<ConcreteBackend>
       │
       └── Flex / WGPU / CUDA / other backend
```

固定 `burn-remote` 的新路径以 Iroh 为主：endpoint 有 identity，peer 通过
地址和发现机制连接；在不能直连时可以使用配置的 relay。固定源码也保留
WebSocket transport，`Device::remote_websocket` 主要是兼容入口。

Remote server 的 `RemoteServerBuilder` 接受一个具体 backend 的设备列表，
并可注册 typed custom operation。server 还必须有非空设备列表和适合目标
平台的 runtime。客户端与 server 的 endpoint、peer discovery、relay policy
和授权 credential 不是 Burn 自动设计的租户系统。

## Remote 的成本和状态

Remote 把计算位置和应用进程分离，可能带来：

- 客户端轻量、peer 可以拥有 GPU 或大内存；
- 多个客户端可以按应用协议使用同一类 compute peer；
- operation、输入和输出需要序列化/传输；
- 网络抖动、peer 断线和远端队列会进入 tail latency；
- 模型 artifact 仍要在客户端或 peer 的某一侧加载，不能把“远端计算”
  与“远端模型注册中心”混为一谈。

固定源码对跨 Iroh peer 的 tensor movement 使用有认证能力的短期 token；
这保护传输操作的授权边界，但不替代应用层的用户认证、模型访问控制和
审计。若模型服务要求请求级隔离，仍需在 Remote 外层建立 policy。

Remote 也可以和 Fusion feature 组合，把重复的 operation group 在 server
侧缓存并按 id 重放。缓存减少重复图传输的机会，但会引入 model graph
版本、shape、生命周期和失效协议；不能仅凭 feature 名称推断所有模型都
会得到性能收益。

## WASM 客户端为什么需要异步连接

浏览器主线程不能执行阻塞式网络连接。固定 `burn-tensor/src/device.rs`
为 Iroh remote device 提供 native 的 `remote_iroh`，并为 WASM 提供
`remote_iroh_async`；`burn-remote` 的 client service 在 wasm 条件下用
浏览器 event loop/future，而 native 路径可以使用线程和 Tokio runtime。

因此浏览器模型服务至少要区分：

1. JavaScript/DOM 事件和模型输入收集；
2. WASM 中的模型结构与权重加载；
3. 异步连接 compute peer；
4. Remote tensor operation；
5. 输出 future 完成后的 UI 更新。

固定 Burn 仓库的 `remote-inference-web` 示例展示了这种架构：浏览器侧
持有 model definition/weights，tensor operations 发送给 Iroh compute
peer。它需要 native peer、浏览器构建工具、网络/relay 配置和匹配的 topic，
不能作为本项目默认 CPU 示例。

## `no_std` 的范围

`no_std` 只表示某个 crate 不依赖 Rust 标准库；它不自动提供文件系统、
线程、浏览器 API 或一个可用 backend。固定源码中的：

- `burn-core`、`burn-tensor` 和核心 `ModuleRecord` 使用 `alloc` 方向；
- `burn-store` 的核心 snapshot 设计标注了 no_std/embedded 方向，但
  `std`、mmap、PyTorch importer 和部分文件操作是 feature/target 条件；
- `burn-onnx` converter 本身是 build-time/CLI 工具，依赖 ONNX parser、
  code generation 和带 `std` 的 store；
- 生成 model 的 `LoadStrategy::Bytes` 或 `Embedded` 可以避免运行时
  `File` loader，前提是目标 backend 和依赖 feature 也支持该目标；
- `LoadStrategy::File` 生成的 `from_file` 使用 `std::path::Path`，固定
  源码为这个生成代码显式加入 `extern crate std`。

所以“转换器在 host 上运行”和“生成的 model 在 no_std 固件里运行”是
两个构建阶段。嵌入式方案还必须测量 binary 大小、静态内存、堆分配、
tensor shape 上限和 backend 的具体算子覆盖。

## 安全边界

Remote transport、WASM binary、Burnpack/SafeTensors 和模型混淆解决的
问题不同：

- TLS/Iroh/授权：主要保护传输或 peer 访问；
- SafeTensors/Burnpack：主要是参数数据格式和加载路径；
- Embedded：减少外部文件依赖，不等于加密；
- TEE/密态计算/混淆：属于额外的运行时或密码学方案；
- service policy：决定谁能调用哪个模型。

不能因为模型以 bytes 传递，就声称它在运行时保持机密；也不能因为 Remote
server 有 `PeerAuthorizer` 扩展点，就声称项目已经提供完整的模型安全
产品。先写清威胁模型，再选择机制和验证方法。

本书将 Remote、WASM 和 no_std 作为固定源码可核验的扩展路径；CPU
round-trip 实验只需要本地内存和 Flex，故把网络、浏览器和固件前提留在
本节及练习中。
