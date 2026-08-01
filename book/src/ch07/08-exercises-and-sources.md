# 7.8 练习、延伸阅读与来源

## 小结

模型服务把训练状态变成可验证、可加载、可执行和可治理的 artifact。
`ModuleRecord` 是 Burn 核心的参数记录：它保存 parameter path、`ParamId`、
shape、dtype 和 bytes，但不保存可以独立执行的 topology。`burn-store` 在
feature 开启后增加 snapshot、Burnpack/SafeTensors/PyTorch store、过滤、
重命名和 adapter。

固定 `burn-onnx` 的 `ModelGen` 将 ONNX graph 解析成 Burn graph，生成 Rust
source，并收集 Burnpack 权重。`File`、`Embedded`、`Bytes` 和 `None`
决定权重从哪里进入生成的 model。由于该固定仓库的 manifest 仍依赖旧
Burn revision，本章源码核验与当前主线 Record 实验保持分离。

Remote 负责把 tensor operation 送到 compute peer；WASM client 的连接
需要异步事件循环；no_std 只缩小标准库依赖，不自动提供文件、网络、线程
或所有 backend。HTTP/gRPC、版本发布、鉴权、限流、指标和故障恢复属于
应用服务层。

## 概念题

1. 为什么 `ModuleRecord` 不能单独恢复一个任意模型？列出 topology、
   参数路径、shape 和 dtype 各自的职责。
2. 比较 `LoadStrategy::File`、`Embedded`、`Bytes` 和 `None` 的 artifact
   生命周期；哪一种会在生成代码中使用 `std::path::Path`？
3. 为什么把 F32 权重存成 F16 不等于完成了低精度推理？列出加载、算子、
   activation 和 reference 校准还需要的条件。
4. 用 $T_{\mathrm{queue}}$、$T_{\mathrm{pre}}$、$T_{\mathrm{copy}}$、
   $T_{\mathrm{forward}}$、$T_{\mathrm{readback}}$ 和
   $T_{\mathrm{post}}$ 分解一次请求，说明动态 batching 可能改善和恶化
   哪些项。
5. Remote peer、模型 registry、HTTP service 和授权系统分别负责什么？
   为什么它们不能仅由 `Device::remote_iroh` 代替？
6. “生成的 model 可以 no_std”与“ONNX converter 可以在 no_std 目标运行”
   为什么是两个构建问题？

## Rust 与 API 题

1. 把实验的 Linear 改成两个 Linear 组成的 `Module`，断言 record tensor
   数量和输出误差。
2. 使用 `try_load_record` 构造 shape mismatch，记录 `RecordError::Validation`
   的行为；再比较 `validate(false)` 和 `allow_partial(true)` 的差异。
3. 让目标 module 使用不同 dtype，比较默认 `FromRecord` 与
   `cast_to_module_dtype` 的结果。
4. 写一个服务 runner，让 model 只在启动阶段加载一次；用 Rust ownership
   明确 handler 的借用、锁或 actor 边界。
5. 给输入增加版本化 schema 和显式前处理函数，测试错误输入不会进入
   `forward`。
6. 用一个 `Bytes` provider 模拟远端/固件来源，比较 `from_bytes` 和
   `from_file` 的错误处理。

## 源码题

1. 阅读 `burn/crates/burn-core/src/store/mod.rs`，追踪
   `into_record → into_bytes → from_bytes → try_load_record` 的数据路径，
   并找出 `ParamId` 恢复位置。
2. 阅读 `burn/crates/burn-core/src/module/base.rs`，比较
   `load_record` 与 `try_load_record` 的错误边界。
3. 阅读 `burn-onnx/crates/burn-onnx/src/model_gen.rs` 和
   `burn-onnx/crates/burn-onnx/src/burn/graph.rs`，画出 ONNX parser、
   graph simplification、codegen 和 Burnpack loader 的调用链。
4. 阅读 `burn-onnx/crates/burn-onnx/src/burn/graph.rs` 的
   `LoadStrategy` 测试，确认四种策略生成了哪些 constructor。
5. 阅读 `burn/crates/burn-store/src/traits.rs`、`adapter.rs` 和
   `tensor_snapshot.rs`，说明 lazy snapshot、filter、remap 和 adapter
   的边界。
6. 阅读 `burn/crates/burn-remote/src/lib.rs`、`server/builder.rs` 和
   `burn/crates/burn-tensor/src/device.rs`，比较 native Iroh、兼容
   WebSocket 和 WASM async device 的入口。
7. 对照 `pins.toml`、`burn-onnx/Cargo.toml` 和根 `Cargo.toml`，解释为什么
   两个 Burn revision 不能直接共用 `Tensor`/`Module` 类型。

## 性能与系统题

1. 测量 cold start、artifact load、warmup、forward、readback 和
   post-processing，报告设备、backend、dtype、shape 和同步边界。
2. 实现动态 batching，分别固定最大 batch size 和最大 queue delay，
   报告 throughput、p50、p95、p99 以及 queue wait。
3. 用固定 reference 输入比较 F32 与一种目标 dtype，分别记录模型大小、
   peak memory、输出误差和延迟；不要只报告压缩率。
4. 设计模型版本发布协议，列出 topology checksum、weight checksum、
   schema version、code revision、backend 和回滚点。
5. 设计 Remote 服务的网络故障测试：peer 断开、请求超时、重复提交、
   tensor transfer 失败和模型 reload。
6. 为 `Embedded`/`Bytes` 的嵌入式部署列出 binary、静态内存、堆、最大
   tensor shape 和算子覆盖预算。
7. 比较“模型文件加密”“transport authorization”“TEE”和“模型混淆”
   所保护的威胁，避免把它们当成同一个开关。
8. 为一个激活张量选择 PTQ 校准集，计算非对称量化的 scale/zero-point，
   比较逐层、逐通道和离群值裁剪的误差与 metadata 成本。
9. 为一个线上模型写四层威胁模型：静态 artifact、传输、运行时内存和
   恶意行为；给每层列出验证证据，并说明 `ModuleRecord` 哪些问题不能
   单独解决。

## 延伸阅读与固定源码入口

Burn 主线：

- `burn/crates/burn-core/src/store/mod.rs`
- `burn/crates/burn-core/src/module/base.rs`
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

## 来源与改编说明

本章改编并重组 OpenMLSys v1 的 `chapter_model_deployment/`：

- `index.md`：保留训练到部署的主问题和学习目标，改为 artifact/runtime/
  service/policy 四层路线；
- `model_deployment_introduction.md`：保留转换、常量折叠、融合、数据
  排布和安全的系统动机，删除未经固定 Burn 验证的厂商实现结论；
- `model_converter_and_optimizer.md`：保留 ONNX 图/算子映射和离线优化，
  以 `ModelGen`、`BurnGraph`、Rust codegen 和 Burnpack 重写；
- `model_compression.md`：保留 PTQ/QAT、稀疏、剪枝和蒸馏的原理，明确
  Burn 固定快照中的 QAT 与通用量化流水线边界；
- `model_inference.md`：保留前/后处理、并行、访存和延迟问题，改为
  Burn Device/runtime 与应用 batcher 的边界；
- `model_security.md`：保留静态/动态保护和威胁模型，区分 artifact、
  transport、Remote authorization 与 TEE/混淆；
- `summary.md`：重写为固定源码证据和本章实验边界。

没有复制 OpenMLSys 的 MindSpore/PyTorch/ARM 汇编代码、图片或 Mate30
性能数字。完整 revision 关系、逐文件核验和不作出的能力承诺见
`planning/chapter-sources/ch07.md`；D010 记录 `burn-onnx` 与主线 Burn
revision 隔离的决定。OpenMLSys 改编正文采用 CC BY-NC-SA 4.0；新增 Rust
示例采用 MIT OR Apache-2.0。
