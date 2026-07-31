# 2026-08-01：第 6 章训练系统

## 会话目标

承接第 5 章交接，审查固定 OpenMLSys v1
`chapter_distributed_training/` 和 Burn 0.22.0-pre.1 的训练源码，完成第
6 章的来源映射、训练循环实验、正文和能力边界。

## 源码核验

### OpenMLSys

逐文件审查 v1 中文章节：

- `index.md`：分布式训练动机和章节地图；
- `overview.md`：算力/内存瓶颈、分而治之和集群动机；
- `methods.md`：数据、模型、混合和流水线并行；
- `collective.md`：集合通信算子、AllReduce 和延迟/带宽模型；
- `parameter_servers.md`：同步/异步、straggler、热点与副本；
- `cluster.md`：节点/机柜网络拓扑和带宽瓶颈；
- `summary.md`：章节总结。

固定 v2 第 6 章仍是 TODO；正文依据 v1 并重写为 Rust/Burn 路线，没有复制
MindSpore/PyTorch 代码或原章节图片。

### Burn

固定 Burn revision 为 `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`。核验：

- `burn-train` 的 `TrainStep`、`Learner`、`SupervisedTraining`、指标、
  检查点和单设备 epoch；
- `burn-optim` 的 `ModuleOptimizer::step/step_multi`、参数组、梯度裁剪、
  learning-rate scheduler 和 optimizer record；
- `ExecutionStrategy` 的 `SingleDevice`、本机 `MultiDevice` 和 DDP；
- `split_dataloader`、本机 worker、`OptimMainDevice`/
  `OptimSharded`；
- `DistributedContext`、autodiff gradient registration、backend
  `all_reduce`/`sync_collective` 和 DDP worker 生命周期；
- `burn-flex` 默认不支持 collective，CubeCL 提供 backend collective 入口，
  CUDA 路径使用 NCCL。

完整逐文件列表和证据级别见 `planning/chapter-sources/ch06.md`。

## 实现

新增：

- `examples/ch06-training-loop/Cargo.toml`
- `examples/ch06-training-loop/src/lib.rs`
- `examples/ch06-training-loop/src/main.rs`
- `book/src/ch06/01-training-state-and-cost.md`
- `book/src/ch06/02-forward-backward-loop.md`
- `book/src/ch06/03-burn-train-orchestration.md`
- `book/src/ch06/04-optimizer-and-checkpoint.md`
- `book/src/ch06/05-local-data-parallel.md`
- `book/src/ch06/06-collective-and-ddp.md`
- `book/src/ch06/07-training-loop-lab.md`
- `book/src/ch06/08-exercises-and-sources.md`
- `planning/chapter-sources/ch06.md`

修改：

- 根 `Cargo.toml`/`Cargo.lock`：加入第 6 章 workspace example；
- `book/src/ch06-training-systems.md`：入口、学习目标、路线和边界；
- `book/src/SUMMARY.md`：加入八个小节导航；
- `planning/DECISIONS.md`：增加 D009；
- `planning/STATUS.md`：待检查完成后更新；
- 本日志和 `planning/session-logs/README.md`。

实验使用五个固定小样本的线性回归：

```text
Tensor batch → Linear → MSE → backward
           → GradientsParams → SGD → loss report
```

测试断言 loss 下降、参数发生变化、步数正确和非法配置返回错误。实验只走
`Device::flex().autodiff()`，不把 Flex DDP 或跨节点通信包装成已验证能力。

## 验证

已通过：

- `cargo fmt --all --check`
- `cargo test -p ch06-training-loop`（2 tests passed）
- `cargo clippy -p ch06-training-loop --all-targets -- -D warnings`
- `cargo run -p ch06-training-loop`（40 步，loss 从约 13.15 降至接近 0）
- `mdbook build book`
- `git diff --check`
- `make check`
- `make check-local-sources`

Cargo 仍提示用户目录同时存在 `~/.cargo/config` 和
`~/.cargo/config.toml`，未修改用户级配置。

## 决策与边界

- D009：CPU Flex 只验证单设备训练循环；固定源码虽有 DDP API，但 Flex
  没有 collective 实现，因此 DDP 需匹配后端和设备环境单独验证。
- 没有添加本地 path dependency、`[patch]`、生成 mdBook 输出或上游修改。
- 没有把本机多设备策略写成跨节点分布式，也没有把参数服务器、流水线并行、
  集群调度和容错写成 Burn 固定快照能力。

## 交接

第 6 章内容、来源映射和 CPU 实验已完成。待全工作区检查完成后，下一步为
从 OpenMLSys v1 `chapter_model_deployment/` 开始第 7 章映射，核验
`burn-onnx`、Record、Remote 和 WASM/no_std 边界。
