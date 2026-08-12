# 练习、延伸阅读与来源

## 小结

模型服务把训练状态变成可验证、可加载、可执行和可治理的 artifact。
`ModuleRecord` 是 Burn 核心的参数记录：它保存 parameter path、`ParamId`、
shape、dtype 和 bytes，但不保存可以独立执行的 topology。`burn-store` 在
feature 开启后增加 snapshot、Burnpack/SafeTensors/PyTorch store、过滤、
重命名和 adapter。

固定 `burn-onnx` 的 `ModelGen` 将 ONNX graph 解析成 Burn graph，生成 Rust
source，并收集 Burnpack 权重。`File`、`Embedded`、`Bytes` 和 `None`
决定权重从哪里进入生成的 model。由于该固定仓库的 manifest 仍依赖旧
Burn revision，本章对 ONNX 的源码对照与本书 Record 实验保持分离。

Remote 负责把 tensor operation 送到 compute peer；WASM client 的连接
需要异步事件循环；no_std 只缩小标准库依赖，不自动提供文件、网络、线程
或所有 backend。HTTP/gRPC、版本发布、鉴权、限流、指标和故障恢复属于
应用服务层。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么 `ModuleRecord` 不能单独恢复一个任意模型？列出 topology、
   参数路径、shape 和 dtype 各自的职责。

<details>
<summary>提示</summary>

运行 `examples/ch07-record-roundtrip`；ONNX/HTTP 另属可选边界。

</details>

2. 【基础】比较 `LoadStrategy::File`、`Embedded`、`Bytes` 和 `None` 的 artifact
   生命周期；哪一种会在生成代码中使用 `std::path::Path`？

<details>
<summary>提示</summary>

运行 `examples/ch07-record-roundtrip`；ONNX/HTTP 另属可选边界。

</details>

3. 【基础】为什么把 F32 权重存成 F16 不等于完成了低精度推理？列出加载、算子、
   activation 和 reference 校准还需要的条件。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【基础】用 $T\_{\mathrm{queue}}$、$T\_{\mathrm{pre}}$、$T\_{\mathrm{copy}}$、
   $T\_{\mathrm{forward}}$、$T\_{\mathrm{readback}}$ 和
   $T\_{\mathrm{post}}$ 分解一次请求，说明动态 batching 可能改善和恶化
   哪些项。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】Remote peer、模型 registry、HTTP service 和授权系统分别负责什么？
   为什么它们不能仅由 `Device::remote_iroh` 代替？

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【进阶】“生成的 model 可以 no_std”与“ONNX converter 可以在 no_std 目标运行”
   为什么是两个构建问题？

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


## Rust 与 API 题

1. 【基础】把实验的 Linear 改成两个 Linear 组成的 `Module`，断言 record tensor
   数量和输出误差。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

2. 【基础】使用 `try_load_record` 构造 shape mismatch，记录 `RecordError::Validation`
   的行为；再比较 `validate(false)` 和 `allow_partial(true)` 的差异。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【进阶】让目标 module 使用不同 dtype，比较默认 `FromRecord` 与
   `cast_to_module_dtype` 的结果。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

4. 【进阶】写一个服务 runner，让 model 只在启动阶段加载一次；用 Rust ownership
   明确 handler 的借用、锁或 actor 边界。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

5. 【进阶】给输入增加版本化 schema 和显式前处理函数，测试错误输入不会进入
   `forward`。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【进阶】用一个 `Bytes` provider 模拟远端/固件来源，比较 `from_bytes` 和
   `from_file` 的错误处理。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>


## 源码题

1. 【进阶】阅读 `burn/crates/burn-core/src/store/mod.rs`，追踪
   `into_record → into_bytes → from_bytes → try_load_record` 的数据路径，
   并找出 `ParamId` 恢复位置。

<details>
<summary>提示</summary>

按本章末「延伸阅读与固定源码入口」打开本书固定版本；配合
`examples/ch07-record-roundtrip` 观察错误 topology 如何被
`RecordError::Validation` 拒绝。

</details>

2. 【进阶】阅读 `burn/crates/burn-core/src/module/base.rs`，比较
   `load_record` 与 `try_load_record` 的错误边界。

<details>
<summary>提示</summary>

见第 2 章对应小节与 `examples/ch02-tensor-basics`。

</details>

3. 【进阶】阅读 `burn-onnx/crates/burn-onnx/src/model_gen.rs` 和
   `burn-onnx/crates/burn-onnx/src/burn/graph.rs`，画出 ONNX parser、
   graph simplification、codegen 和 Burnpack loader 的调用链。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

4. 【进阶】阅读 `burn-onnx/crates/burn-onnx/src/burn/graph.rs` 的
   `LoadStrategy` 测试，确认四种策略生成了哪些 constructor。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

5. 【进阶】阅读 `burn/crates/burn-store/src/traits.rs`、`adapter.rs` 和
   `tensor_snapshot.rs`，说明 lazy snapshot、filter、remap 和 adapter
   的边界。

<details>
<summary>提示</summary>

见第 9 章拓扑与调度节及网络配图。

</details>

6. 【进阶】阅读 `burn/crates/burn-remote/src/lib.rs`、`server/builder.rs` 和
   `burn/crates/burn-tensor/src/device.rs`，比较 native Iroh、兼容
   WebSocket 和 WASM async device 的入口。

<details>
<summary>提示</summary>

按章节末「源码入口」阅读本书固定版本的源码，不要跟着在线最新文档改 API。

</details>

7. 【进阶】对照仓库版本钉扎、`burn-onnx/Cargo.toml` 和根 `Cargo.toml`，解释为什么
   两个 Burn revision 不能直接共用 `Tensor`/`Module` 类型。

<details>
<summary>提示</summary>

按章节末「源码入口」打开本书固定版本的对应路径。

</details>


## 性能与系统题

1. 【进阶】测量 cold start、artifact load、warmup、forward、readback 和
   post-processing，报告设备、backend、dtype、shape 和同步边界。

<details>
<summary>提示</summary>

运行 `examples/ch07-record-roundtrip`；ONNX/HTTP 另属可选边界。

</details>

2. 【挑战】实现动态 batching，分别固定最大 batch size 和最大 queue delay，
   报告 throughput、p50、p95、p99 以及 queue wait。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

3. 【挑战】用固定 reference 输入比较 F32 与一种目标 dtype，分别记录模型大小、
   peak memory、输出误差和延迟；不要只报告压缩率。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

4. 【挑战】设计模型版本发布协议，列出 topology checksum、weight checksum、
   schema version、code revision、backend 和回滚点。

<details>
<summary>提示</summary>

运行 `examples/ch09-cluster-simulator`；真实集群属可选平台。

</details>

5. 【挑战】设计 Remote 服务的网络故障测试：peer 断开、请求超时、重复提交、
   tensor transfer 失败和模型 reload。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

6. 【挑战】为 `Embedded`/`Bytes` 的嵌入式部署列出 binary、静态内存、堆、最大
   tensor shape 和算子覆盖预算。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

7. 【挑战】比较“模型文件加密”“transport authorization”“TEE”和“模型混淆”
   所保护的威胁，避免把它们当成同一个开关。

<details>
<summary>提示</summary>

回看第 7 章与本题对应的小节；需要实现时优先改本章 `examples/` 测试。

</details>

8. 【挑战】为一个激活张量选择 PTQ 校准集，计算非对称量化的 scale/zero-point，
   比较逐层、逐通道和离群值裁剪的误差与 metadata 成本。

<details>
<summary>提示</summary>

见第 7 章压缩节的 $s,z$ 带数字演算。

</details>

9. 【挑战】为一个线上模型写四层威胁模型：静态 artifact、传输、运行时内存和
   恶意行为；给每层列出验证证据，并说明 `ModuleRecord` 哪些问题不能
   单独解决。

<details>
<summary>提示</summary>

运行 `examples/ch07-record-roundtrip`；ONNX/HTTP 另属可选边界。

</details>


## 延伸阅读与固定源码入口

本书示例使用的 Burn：

- `burn/crates/burn-core/src/store/mod.rs`
- `burn/crates/burn-core/src/module/base.rs`
- `burn/crates/burn-pack/src/base.rs`（Burnpack 头部、对齐与容量上限）
- `burn/crates/burn-pack/src/writer.rs`（CBOR metadata 与 tensor 布局规划）
- `burn/crates/burn-store/src/traits.rs`
- `burn/crates/burn-store/src/burnpack/`
- `burn/crates/burn-store/src/safetensors/`
- `burn/crates/burn-store/src/adapter.rs`
- `burn/crates/burn-store/src/tensor_snapshot.rs`
- `burn/crates/burn/Cargo.toml`
- `burn/crates/burn-tensor/Cargo.toml`
- `burn/crates/burn-tensor/src/device.rs`
- `burn/crates/burn-remote/README.md`
- `burn/crates/burn-remote/src/lib.rs`
- `burn/crates/burn-remote/src/server/builder.rs`
- `burn/examples/remote-inference-web/README.md`

固定 `burn-onnx`：

- `burn-onnx/Cargo.toml`
- `burn-onnx/README.md`
- `burn-onnx/crates/burn-onnx/src/model_gen.rs`
- `burn-onnx/crates/burn-onnx/src/burn/graph.rs`
- `burn-onnx/crates/burn-onnx/src/bin/onnx2burn.rs`
- `burn-onnx/examples/onnx-inference/README.md`

OpenMLSys v1：

- `openmlsys/v1/zh_chapters/chapter_model_deployment/index.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_deployment_introduction.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_converter_and_optimizer.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_compression.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_inference.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_security.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/summary.md`

## 本章系统结论

1. 部署闭环是：训练态 → artifact → 校验 → 推理入口 → 服务侧 batch/队列/版本。
2. `ModuleRecord`/Burnpack 管参数状态；ONNX/codegen 管图转换，二者版本不能混用。
3. 推理选用哪个 Device（CPU/WGPU/CUDA）与 artifact 格式正交，但影响延迟与批处理。
4. CPU 上你验证了 Linear 参数往返保存/恢复与输出误差边界。
5. GPU 阅读线索：同一 record 加载到加速 Device 后的同步与 batch 合并；LLM 服务专题见正文边界段。
6. 不能把本地 load 成功当成完整服务上线或 ONNX 端到端已在同一依赖图验证。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-7-章)。
