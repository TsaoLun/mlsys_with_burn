# 2026-08-13 深度批次三：算子解剖贯穿页

## 背景与定位

用户目标澄清：不做任何明面上的「贡献者内容」（不加贡献附录、社区
流程、PR 指引），书的职责是把框架/系统机制**彻底**讲透——搞懂到
「知道一个算子牵动哪些文件、每层为什么存在、缺一层会怎样」的程度，
使贡献成为自然副产品。本批交付该哲学下的核心资产：算子解剖贯穿页。
纯内容深化，无范围决策需要。

## 交付

1. `book/src/op-anatomy.md`（SUMMARY「贯穿实验」区）：以 tanh 为
   样本的十层解剖，全部摘录自固定 revision——
   - API `burn-tensor/api/float.rs`（一行转发）；
   - 契约 `burn-backend/ops/tensor.rs`（`fn float_tanh` 声明，缺
     实现=编译错误）；
   - 分派 `burn-dispatch/ops/tensor.rs`（`unary_float!` 宏，运行时
     后端选择）；
   - autodiff `burn-autodiff/ops/tensor.rs:2737`（`grad·(1-tanh²)`
     用后端算子表达、`State`/checkpointer、`memory_bound()` +
     `retro_forward` 的按算子 checkpoint 策略、装饰器模式）；
   - Flex `burn-flex/ops/unary.rs:258`（f32/f64 双闭包落到标准库）；
   - CubeCL `burn-cubecl/ops/tensor.rs:590`
     （`BasicFloatUnaryKind::Tanh` JIT 一元 kernel）；
   - Fusion `burn-fusion/ops/tensor.rs:1783`（描述 + 回退执行体
     双注册）；
   - IR `burn-ir/operation.rs:170`（`Tanh(UnaryOpIr)` 词汇表）；
   - 契约测试 `burn-backend-tests`（同一 `should_diff_tanh` 断言，
     `cargo test-cpu/-wgpu/-cuda` feature 切后端）。
   页末给出「换一个算子怎么走」（二元/归约/带状态的差异点）与
   「缺一层的失效模式」。
2. `examples/ch02-ch04-op-anatomy`（4 测试）：前向 vs 标量参考、
   反向 vs 解析式、中心差分独立核对、组合乘积法则；运行输出三行
   0.00e0 已引入页面。
3. 交叉指针：ch02/05 源码导读、ch04/03 开头各一句；附录范围新增
   「算子解剖」条目；running-examples、Makefile、workspace 同步。

## 验证

- `cargo test -p ch02-ch04-op-anatomy --locked --offline`：4 通过；
  clippy 零警告；`cargo fmt --all --check` 通过；
- `mdbook build/test book`（90 章）、`check_release.py
  --require-built-book --json`（`ok=true`、`errors=[]`、
  `warnings=[]`）、`git diff --check` 通过；
- 十层摘录均以 `git show/grep <pin>` 核实（含行号）。

## 边界与偏差

- 摘录为短片段引用（数行级），与既有 `rust,ignore` 示意片段体例
  一致；上游许可 MIT/Apache-2.0，NOTICE 已有归属。
- GPU/Fusion 层默认不运行，页内指向第 3 章 wgpu 路径与第 4 章
  FusionInspector 的可观察版本。

## 后续候选（同一哲学下的机制纵深）

- `#[cube]` 与 `#[derive(Module)]` 的宏展开机制小节（从
  cubecl-macros / burn-derive 源码给等价展开形式）；
- ch07「一个 ONNX 算子的旅程」（onnx-ir 节点 → 注册 → codegen →
  权重装载全链路）；
- 二元/归约算子的解剖对照（广播梯度归约、fuser 接受边界）。
