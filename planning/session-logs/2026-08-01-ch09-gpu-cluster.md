# 第 9 章大规模 GPU 集群管理

## 目标

完成第 9 章正文、固定 OpenMLSys/Burn/CubeCL 来源映射和 CPU-first 集群
控制面模拟器。保持真实 GPU/NCCL/跨节点能力与 CPU 协议模拟的边界。

## 来源与决策

- OpenMLSys v1 固定 revision：
  `9c289782ccbb165ac8ad7c960ecffc12942a5560`。
- Burn 固定 revision：
  `976aa9c5ec1d2dd3412710f99759e3c44bdff03d`。
- CubeCL 固定 revision：
  `be278a1e76aed881e2cc6b165414ee6103ca4634`。
- OpenMLSys 的 `cluster.md`、`collective.md`、`methods.md` 和
  `parameter_servers.md` 提供拓扑、通信、并行、straggler 和副本动机。
- 固定 Burn/CubeCL 源码提供 DDP、`DistributedContext`、collective、
  `ComputeClient`、stream、memory 和 CUDA/NCCL 数据面入口；不提供集群
  作业队列、拓扑放置、多租户、elastic membership、自动重试或统一遥测。
- 新增 D013：第 9 章以纯 Rust 虚拟时间模拟器隔离集群控制面和真实 GPU
  集群，不把模拟结果写成硬件 benchmark。

## 主要修改

- 将 `book/src/ch09-gpu-cluster.md` 从骨架补成完整章节入口。
- 新增第 9 章 8 个小节，覆盖 workload card、控制面/数据面/运行时、
  rack/ToR/Spine、成组调度、拓扑放置、通信成本、多租户、故障、
  checkpoint、遥测、实验、练习和逐文件来源说明。
- 新增 `examples/ch09-cluster-simulator`：
  - 逻辑 GPU/node/rack/memory 模型；
  - FIFO 与 topology-aware placement；
  - gang admission；
  - `alpha + beta * bytes` 通信成本和跨机柜 penalty；
  - deterministic failure、checkpoint replay、retry 和资源归还；
  - queue wait、makespan、cross-rack bytes、collective time、p95 等报告。
- 更新 `SUMMARY.md`、`TERM_GLOSSARY.md`、`planning/chapter-sources/ch09.md`、
  `planning/DECISIONS.md`、`planning/STATUS.md` 和本日志索引。

## 当前验证

已完成：

```text
cargo fmt --all
cargo fmt --all --check
cargo test -p ch09-cluster-simulator
cargo clippy -p ch09-cluster-simulator --all-targets -- -D warnings
cargo run -p ch09-cluster-simulator
mdbook build book
make check
make check-local-sources
git diff --check
```

模拟器测试覆盖 5 项：资源/成组准入、拓扑通信差异、通信单调性、确定性
trace、失败后 checkpoint replay 和资源释放。全量 workspace lint/test、
mdBook 构建、pin 检查和本地上游镜像检查均通过。公式静态复查结果为：
源码无未转义数学下标和 display 列表标记，生成 HTML 有 96 个 display
公式和 244 个行内公式候选，未发现 `<em>`/`<ul>`/`<ol>` 结构污染，含
公式页面均加载 MathJax。

## 交接

第 9 章正文、来源映射、实验、导航和交接记录已完成；下一步进入首个稳定
版的全书链接、许可证和来源审计。
