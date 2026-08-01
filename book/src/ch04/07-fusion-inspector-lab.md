# 实验：观察 Fusion 执行计划

实验位于 `examples/ch04-fusion-inspector`。根 workspace 为 Burn 启用
`cpu` 与 `fusion`，并直接依赖带 `test-util` feature 的 `burn-fusion`。
后者提供 `FusionInspector`，用来捕获固定 Runtime 的执行报告。

`FusionInspector` 明确是测试观测 API。生产程序通常使用
`BURN_FUSION_LOG` 和 tracing/性能工具，而不依赖 test-util。

## 1. 问题设置

比较语义相同的两个程序：

```text
连续：out = exp(left + right)

切分：intermediate = left + right
      sync()
      out = exp(intermediate)
```

输入都是 $4\times4$ 的全 1 Tensor，预期每项为：

$$
\exp(1+1)=e^2\approx 7.389056。
$$

实验不比较两个路径的墙钟性能。首次 JIT、Fusion 搜索、线程调度和缓存会
污染这种微小输入的计时；本节目标是验证计划结构。

## 2. 观测代码

```rust,ignore
{{#include ../../../examples/ch04-fusion-inspector/src/lib.rs:inspect}}
```

关键步骤：

1. 使用 `Device::cpu()`，确保进入 CubeCL CPU Fusion，而不是 Flex；
2. 先创建并同步输入，避免 `ones` 初始化进入目标报告；
3. 分配独立 StreamId，并为它安装 Inspector；
4. 注册 add 与 exp；
5. 可选地在两者之间同步；
6. `to_data()` 物化结果，再 drain Inspector；
7. 把内部报告转换为教材自己的 `FusionSummary`。

自定义 summary 避免让实验的公开返回类型直接泄漏全部内部结构；示例仍
依赖 test-util，所以它不是长期稳定库接口。

## 3. 运行

```bash
cargo run -p ch04-fusion-inspector
```

固定快照上的一种输出为：

```text
连续表达式：1 个报告，[BlockSummary { fuser: Some("ElementWise"), operations: 2 }]
同步切分后：2 个报告，[BlockSummary { fuser: None, operations: 1 }, BlockSummary { fuser: None, operations: 1 }]
输出前四项：[7.389056, 7.389056, 7.389056, 7.389056]
```

连续表达式中的 add 和 exp 被同一个 `ElementWise` block 捕获。同步切分后，
两个单操作报告分别执行，不能跨同步边界形成两操作 block。

这里的 `operations: 2` 是 Fusion OperationIr 数量；它不证明底层只提交了
某个精确数量的设备 Kernel，也不包含 JIT、autotune 或内存命令计数。

## 4. 测试

```bash
cargo test -p ch04-fusion-inspector
```

测试断言：

- 连续路径存在两操作 ElementWise block；
- 同步路径不存在这样的 block；
- 两条路径都实际观察到 add 和 exp，报告数非零；
- 两条路径输出完全相同；
- 数值在容差内等于 $e^2$。

“计划不同但语义相同”正是编译优化需要满足的条件。若只断言输出，Fusion
完全回退也会通过；若只断言计划，不验证输出，又可能忽略错误变换。

## 5. 为什么不用 Flex 对照

`Device::flex()` 是 eager dispatch，不注册 Fusion OperationIr。它可以做
数值 reference，却无法让 FusionInspector 捕获同类报告。比较“CPU Fusion
开/关”需要在 Cargo feature 和 Backend 类型层选择不同实现，不是给 Flex
安装 Inspector。

## 6. 日志扩展

无需 test-util 时，可运行：

```bash
BURN_FUSION_LOG=full cargo run -p ch04-fusion-inspector
```

日志级别和环境变量属于固定 Burn 配置接口。Full 日志可显示 stream、fuser、
plan 与 explorer 信息，输出量较大；不要在自动测试中依赖完整日志文本，
其格式比结构化断言更容易演进。

## 7. 三操作 ElementWise 块

同 crate 还提供 `inspect_add_mul_exp`：

```rust,ignore
{{#include ../../../examples/ch04-fusion-inspector/src/lib.rs:inspect_triple}}
```

连续 `((left + right) * scale).exp()` 在固定快照上可落入同一个三操作
`ElementWise` block；数值仍为 $e^2$（因为 `scale` 为全 1）。这把练习中的
扩写题收敛为已交付断言，而不改变主实验对同步切分的关注。

## 8. 重复计划与缓存日志

`inspect_add_mul_exp_twice` 在相同 shape、dtype、CPU Fusion device 和
stream 上重复 `add → mul → exp`，比较两次的 `reports`、block 结构和
输出值。测试只断言计划/输出一致，不断言第二次更快，也不读取私有 cache
key：

```bash
cargo run -p ch04-fusion-inspector --locked --offline
BURN_FUSION_LOG=full cargo run -p ch04-fusion-inspector --locked --offline
```

主程序的 `cache_log_enabled=true` 只表示请求了可选日志，不表示发生了
cache hit。Fusion block 数、cache hit、kernel launch count 和 wall-clock
time 是四种不同指标；固定快照没有提供稳定的公开 cache-key 读取接口，
因此正文保留日志/源码核验边界。

## 9. 可继续观察的边界

1. 在不同位置插入 `sync()`，记录报告切分；
2. 加入 broadcast，观察 ElementWise fuser 是否接受；
3. 加入 reduce 或 matmul，记录 fuser 名称和 fallback；
4. 比较 `StreamId::allocate()` 与 `StreamId::current()` 的测试隔离效果；
5. 分别记录首次和稳态时间，但把它们作为环境相关测量。

