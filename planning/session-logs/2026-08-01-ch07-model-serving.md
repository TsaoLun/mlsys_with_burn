# 2026-08-01：第 7 章模型服务

## 会话目标

承接第 6 章交接，审查固定 OpenMLSys v1
`chapter_model_deployment/` 和 Burn 0.22.0-pre.1 相关源码，完成第 7 章
的来源映射、部署边界、CPU 可验证实验、正文和交接状态。

## 源码核验

### OpenMLSys

逐文件审查 v1 中文章节：

- `index.md`：模型转换、硬件约束、时延/功耗/内存和安全地图；
- `model_deployment_introduction.md`：转换、常量折叠、融合、数据排布和
  模型保护；
- `model_converter_and_optimizer.md`：ONNX、图/算子映射、融合、替换和
  重排；
- `model_compression.md`：量化、PTQ/QAT、稀疏、剪枝和蒸馏；
- `model_inference.md`：前处理/后处理、并行、ARM 指令和卷积优化；
- `model_security.md`：静态/动态保护、TEE、密态计算和模型混淆；
- `summary.md`：部署指标与优化方向。

正文没有复制原章节图片、MindSpore/PyTorch 代码或设备性能数字。

### Burn 主线

固定 Burn revision 为 `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`。核验：

- `burn-core/src/store/mod.rs` 的 `ModuleRecord`、Burnpack bytes、dtype
  policy、validation、partial loading 和 `ParamId` 恢复；
- `burn-core/src/module/base.rs` 的 `into_record`、`try_load_record`、
  `load_record` 和 file convenience；
- `burn-store` 的 `ModuleSnapshot`、`ModuleStore`、Burnpack/SafeTensors、
  snapshot lazy materialization、filters、remap 和 PyTorch/dtype adapters；
- `burn/Cargo.toml` 的 store/remote/quantization 文档边界；
- `burn-tensor` 的 remote、WASM async device 和 no_std 条件；
- `burn-remote` 的 Iroh/WebSocket client、server builder、custom op 和
  native/WASM 条件；
- `remote-inference-web` 示例的 browser model + remote compute peer 路线。

### `burn-onnx`

固定 `burn-onnx` revision 为 `af2dfb43af43bf363dc2d7d858d933d86e2a65a8`，
其 `Cargo.toml` 明确把 `burn`、`burn-flex` 和 `burn-store` 指向
`78f10aec1ca6c6ffb1edd17a0fa131ae59ad5403`。`ModelGen`/`BurnGraph` 核验：

- ONNX parse 与可选 simplify；
- ONNX nodes 到 BurnGraph；
- Rust source codegen 和大图 partition；
- `.bpk` snapshot；
- `File`、`Embedded`、`Bytes`、`None` 四种 loader 生成；
- `onnx2burn` CLI 与旧 workspace example。

该 revision 关系导致 `burn-onnx` 与根项目主线 Burn 的 Tensor/Module/store
类型不能直接混用。D010 记录了隔离决定。

## 实现

新增：

- `examples/ch07-record-roundtrip/Cargo.toml`
- `examples/ch07-record-roundtrip/src/lib.rs`
- `examples/ch07-record-roundtrip/src/main.rs`
- `book/src/ch07/01-deployment-boundary.md`
- `book/src/ch07/02-onnx-and-codegen.md`
- `book/src/ch07/03-record-and-artifacts.md`
- `book/src/ch07/04-compression-and-optimization.md`
- `book/src/ch07/05-inference-runtime-and-service.md`
- `book/src/ch07/06-remote-wasm-and-nostd.md`
- `book/src/ch07/07-record-roundtrip-lab.md`
- `book/src/ch07/08-exercises-and-sources.md`
- `planning/chapter-sources/ch07.md`

修改：

- 根 `Cargo.toml`/`Cargo.lock`：加入第 7 章 workspace example；
- `book/src/ch07-model-serving.md`：入口、学习目标、路线和版本边界；
- `book/src/SUMMARY.md`：加入八个小节导航；
- `planning/DECISIONS.md`：增加 D010；
- `planning/STATUS.md`：完成第 7 章交接并指向第 8 章；
- 本日志和 `planning/session-logs/README.md`。

实验使用主线 Burn Flex CPU：

```text
Linear model
  → reference forward
  → Module::into_record
  → ModuleRecord::into_bytes/from_bytes
  → fresh model::try_load_record
  → output shape + max absolute error
```

## 验证

已通过：

- `cargo fmt --all`
- `cargo test -p ch07-record-roundtrip`（1 test passed）
- `cargo clippy -p ch07-record-roundtrip --all-targets -- -D warnings`
- `cargo run -p ch07-record-roundtrip`，输出
  `record_tensors=2 output_shape=[3, 1] max_abs_error=0.000000e0`
- `mdbook build book`
- `make check`（workspace lint 与 7 个 workspace package 测试通过）
- `make check-local-sources`
- `git diff --check`

Cargo 仍提示用户目录同时存在 `~/.cargo/config` 和
`~/.cargo/config.toml`，未修改用户级配置。

## 决策与边界

- D010：固定 `burn-onnx` 旧 Burn revision 不进入当前根 workspace；当前
  实验只验证主线 `ModuleRecord`。
- 没有添加本地 path dependency、`[patch]`、生成 mdBook 输出或上游修改。
- 没有把 Record round-trip 写成 ONNX importer、Remote、HTTP/gRPC、WASM、
  量化或 GPU 端到端验证。
- 没有把 Embedded bytes 写成加密，也没有把 Remote transport 写成完整
  模型服务安全方案。

## 交接

第 7 章内容、来源映射、CPU artifact 实验和全量检查已完成。下一步为从
OpenMLSys v1 `chapter_reinforcement_learning/` 开始第 8 章映射；首先核验
固定 `burn-rl`、环境 trait 和 CPU 可测试采样路径。
