# 一次调用会经过哪些层

改框架时最常见的失误，是在错误的一层动手：API 改了、后端没实现；反向
公式对了、融合规则却把 NaN 引进来；Kernel 变快了、autotune 键没更新。
下面这张表回答一个更具体的问题——**你看到的行为，代码写在哪**。

完整走读见[算子解剖：tanh 的完整一生](op-anatomy.md)。路径相对于
`burn/crates/`（CubeCL / CubeK 在各自仓库）。本书示例对齐
[如何运行本书示例](running-examples.md) 里的依赖版本；向当前上游仓库
提交改动前，请用符号名搜索，不要复制过期行号。

## 张量运算

| 你想动的行为 | 先打开 | 旁边还要看 |
|---|---|---|
| 用户 API 形状与文档 | `burn-tensor` 的 `tensor/api/` | 编译期秩 `D`、类别 `K` |
| 后端必须实现哪些算子 | `burn-backend` 的 `ops/tensor.rs` | 缺实现 = 该后端编不过 |
| 运行时按设备分派 | `burn-dispatch` | 第 1 章的 `DispatchDevice` |
| 反向公式、checkpoint | `burn-autodiff` 的 `ops/` | `burn-backend-tests` 里同名 `should_diff_*`；换算子从 `mean`/`sum`/`add` 对照开始 |
| CPU eager 实现 | `burn-flex` | 默认示例走这里，不经过 CubeCL |
| GPU / JIT Kernel | `burn-cubecl`，再进 `cubecl/` | 第 3 章 `#[cube]` |
| 高性能 matmul / 注意力 | `cubek/` 的对应 crate | 第 3 章 Strategy 与 tune key |
| 融合是否发生、如何回退 | `burn-fusion`、`burn-cubecl-fusion` | 第 4 章 Fusion 计划 |
| IR 词汇表（有没有这个算子） | `burn-ir` 的 `operation.rs` | 新算子通常要在这里挂号 |

## 数据、训练与产物

| 你想动的行为 | 先打开 | 旁边还要看 |
|---|---|---|
| Dataset / map / 多 worker | `burn-dataset` | 第 5 章：守恒 ≠ 保序 |
| DataLoader 与 Batcher | `burn-core` 的 `data/` | Device 投放发生在 Batcher |
| 训练循环、Learner、checkpoint | `burn-train`、`burn-optim` | 第 6 章状态机 |
| 本机多设备 / DDP 策略 | `burn-train` 的 `learner/supervised/strategies/` | collective 在 backend，不在 train |
| `all_reduce` 契约 | `burn-tensor` 的 `distributed.rs`、`burn-backend` 的 `DistributedOps` | Flex **没有** collective 实现 |
| 参数导出与加载 | `burn-core` 的 `module/`、`burn-store` | 第 7 章 Burnpack |
| ONNX → Rust 代码生成 | 独立仓库 `burn-onnx` | 与本书示例不是同一份 Burn 提交 |
| 环境、replay、Policy 组合 | `burn-rl` | 第 8 章；具体算法由应用实现 |

## 怎么判断改对了层

1. **契约在上、实现在下。** 只改 `burn-tensor` 的一行转发，所有后端的
   数值都不会变。
2. **测试跟层走。** 反向规则进 `burn-backend-tests`；Kernel 用 host
   reference；融合看计划结构，不要只看墙钟。
3. **默认示例走 Flex。** 若行为只在 GPU 路径出现，应到 `burn-cubecl` /
   CubeCL，而不是在 Flex 里找。
4. **训练编排不是通信。** 改 DDP worker 生命周期解决不了 NCCL 算法；
   改作业队列解决不了 AllReduce 语义。后者分别在第 6 章与第 9 章。
