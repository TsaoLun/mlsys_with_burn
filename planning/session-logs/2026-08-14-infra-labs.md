# 2026-08-14 并行/服务成本实验

## 目标

用户已将 D025 重编提交到 `main`（`82c0475`），要求继续 STATUS 中未做的
三项内容增量：并行策略虚拟时间实验、服务队列加厚、归约算子解剖练习。

## 选择

- 新 crate `ch06-parallel-strategies` 零依赖，公式用整数与交叉相乘，
  避免把空泡写成小数协议。
- 不新增第 6/7 章小节（保持 8 节）。动手版写在 `ch06/06` 与 `ch07/05`。
- 服务队列在原 crate 上扩展：TTFT/TPOT 字段 + `simulate_chunked_prefill`；
  `simulate_continuous` 定义为 `chunk = u32::MAX`，用测试锁住逐步重合。
- 原练习「实现 chunked prefill」改为扫描 chunk 权衡；KV 抢占仍为挑战题。
- 新增 `capstone-infra.md` 合读两张表，不另起一条数据→训练 capstone。
- `mean` 走 Burn `Tensor::mean()` 的 autodiff 路径，与 `sum` 对照。

## 操作

- 示例：`examples/ch06-parallel-strategies/**`，
  `examples/ch07-serving-queue-sim/src/{lib,main}.rs`，
  `examples/ch02-ch04-op-anatomy/src/{lib,main}.rs`。
- 正文：`ch06/06–08`、`ch07/05`/`08`、解剖页、SUMMARY、running-examples、
  infra/crate-map、glossary、附录证据账本。
- 决策：D026。来源映射：ch02/ch06/ch07 各追加一节。

## 验证

见 `planning/STATUS.md` 本次交接（本会话在提交后跑测试）。

## 下一步

发布者审阅 D026 与新页；有环境时重跑完整 `make check`。
