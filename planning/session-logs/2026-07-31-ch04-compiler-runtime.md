# 2026-07-31：第 4 章 AI 编译器与运行时系统

## 会话目标

继续 M3 系统篇，完成第 4 章的来源映射、固定源码核验、正文和 CPU 可验证
实验。

## 开始状态

- Git：`main` 位于 `5eda378`，工作区干净。
- 第 3 章已提交，M2 基础篇完成。
- 当前目标：OpenMLSys frontend/IR、backend/runtime 与 Burn/CubeCL 映射。

## 研究与边界

### OpenMLSys

逐文件检查 v1：

- `chapter_frontend_and_ir/`
- `chapter_backend_and_runtime/`

保留 IR 分类、多层编译、经典 Pass、融合、Kernel 选择、layout/dtype、
生命周期、内存池、异步调度和 compute/schedule 思想。MindIR、MindSpore
Graph Kernel、SOMAS、Ascend task 下沉及长 Python/TVM 示例不作为 Burn
主线。v2 第 4 章只有 TODO，没有可迁移正文。

### Burn

固定源码核验：

- Flex 是 eager 路径，不产生 Fusion OperationIr；
- `Device::cpu()` 在 `cpu`+`fusion` feature 下进入 CubeCL CPU Fusion；
- Tensor op 通过 Fusion client/server 注册到按 StreamId 管理的队列；
- OperationIr、TensorIr、TensorStatus 与 HandleContainer 描述操作和资源；
- burn-cubecl 注册 ElementWise、Matmul、Reduce、ReduceBroadcasted、
  NHWCRelayout fuser；
- read/sync drain stream；同步会切断跨边界融合；
- FusionInspector 只在 `burn-fusion/test-util` 下提供。

### CubeCL

核验 `#[cube]` expand → Scope → KernelBuilder → KernelDefinition →
Compiler/optimizer → JIT/cache → ComputeClient launch。不同 Compiler 的优化
管线不同。CPU 使用进程内编译缓存，部分设备后端有条件支持持久化缓存。
flush 主要是提交/推进接口，不跨 Runtime 承诺设备完成；read/sync 才是可靠
完成边界。

## 章节实现

新增八节：

1. 编译栈与中间表示；
2. 静态信息、Pass 与自动微分边界；
3. Burn IR 与运行时融合；
4. 图优化、Kernel 选择与回退；
5. CubeCL Lowering、JIT 与缓存；
6. 内存、Stream 与异步执行；
7. FusionInspector 实验；
8. 练习、延伸阅读与来源。

建立 `planning/chapter-sources/ch04.md`，不复制 OpenMLSys ch04/ch05 图片或
框架专用代码。

## 实验实现

新增 `examples/ch04-fusion-inspector`：

- 仅在该 crate 局部启用 Burn `cpu`、`fusion`；
- 局部启用 `burn-fusion` 的 `test-util`；
- 先同步输入，避免初始化操作污染目标报告；
- 每次调用使用 `StreamId::allocate()` 隔离 Inspector；
- 比较连续 add→exp 与中间 `sync()` 的语义和执行计划；
- 使用官方 operation matchers 确认实际观察到 add 和 exp；
- 将内部报告转换为稳定的教材 `FusionSummary`；
- 同时断言计划结构、非空报告、数值一致与 $e^2$ 容差。

实验模式参考固定 Burn
`crates/burn-backend-tests/tests/fusion/fusion_shape.rs`，但重新设计了错误
传播、独立 stream、summary 和教学输出；来源已明确记录。

## 审校发现与修复

1. 初版 workspace 为所有 Burn 示例统一启用 CPU/Fusion，扩大了第 1、2 章
   单包构建依赖。改为只在 ch04 crate 局部追加 feature。
2. 初版只用 fuser 名称和操作数判断融合，且同步路径零报告也可能通过。
   增加 add/exp matcher、非空报告和双路径观测断言。
3. 初版使用 `StreamId::current()`，可能与同 stream 重入冲突；改为
   `StreamId::allocate()`。
4. 补充实验对 Burn 上游回归模式的来源，删除“完全原创”表述。
5. 将持久化编译缓存改为后端条件能力，明确 CPU 是进程内缓存。
6. 区分 flush 提交边界与 read/sync 完成边界。
7. 对设备 graph capture 增加底层 Runtime 能力条件，明确 CPU 不支持。
8. 修正 BeamSearchConfig 描述，不把候选 block 上限外推为通用评分算法。
9. 补充 CubeOptimization 从 Fusion 计划进入 ElementWise/CubeK/fallback
   再到 CubeCL 的桥梁。
10. 最终 Clippy 报告 matcher 的显式闭包冗余，按建议直接把 matcher 传给
    `Iterator::any`。

## 验证

成功运行：

```text
cargo test -p ch04-fusion-inspector
cargo run -p ch04-fusion-inspector
make check
make check-local-sources
git diff --check
```

观测：

```text
连续表达式：1 个报告，ElementWise 两操作块
同步切分后：2 个报告，各一个操作
输出前四项：[7.389056, 7.389056, 7.389056, 7.389056]
```

首次加入 CPU Fusion 依赖的冷构建约 90 秒，增量测试约数秒。用户级 Cargo
仍提示 `.cargo/config` 与 `.cargo/config.toml` 同时存在，不影响项目验证。

## 状态与下一步

- 第 4 章完成，M3 继续；
- 下一步映射第 5 章数据处理系统，核验 burn-dataset、DataLoader、Batcher
  和多线程/随机性边界。

