# 2026-08-13 深度批次二：迷你 Pass 流水线、PTQ 校准、服务队列模型

## 背景与决策

按深度菜单顺序继续（批次一为 GEMM 阶梯 + 迷你 tape）。本批交付
C/D/E 三项，全部纯 Rust、零外部依赖、默认 CPU 可跑：

- C `ch04-mini-pass-pipeline`：把第 4 章 Pass 契约变成可运行断言；
- D `ch07-ptq-calibration`：把第 7 章 PTQ 演算变成可复现数字；
- E `ch07-serving-queue-sim`：连续批处理与 KV 预算的队列协议模型，
  需要且已记录扩围决策 **D024**（LLM 机制以协议模型进入第 7 章，
  Burn 能力边界不变）。

## 交付与关键数字

1. **C（8 测试）**：五算子 IR + 常量折叠/DCE/CSE（16 个随机图按位
   一致）+ 融合分组分析（中间值被观察即切组，与 FusionInspector
   同步切分同构）+ 故意非法的 fast-math 消去——正文
   $(10^{16}+1)-10^{16}$ 反例的可运行版（小常量无害 vs 大常量
   0 变 1 两测试并排）。主程序输出 15→9→7 节点流水线。接入
   ch04/02、ch04/07 §10、练习两题。
2. **D（6 测试）**：int8 仿射量化、min-max vs 百分位、per-channel、
   i32 累加 GEMM。开发中发现并修正了两个天真断言——整体 MSE 下
   分位校准反而更差（0.034 vs 58.5，被裁剪离群值主导），收益在
   主体 MSE（0.0335 vs 0.000065）；per-channel 收益集中在窄通道
   （3.2e-5 vs 4.3e-9）。测试与正文都按「校准是明码交易、指标须
   与任务对齐」重写。接入 ch07/04「动手验证」小节、练习题 8。
3. **E（5 测试，D024）**：虚拟时间队列模型。混合负载（64 条，
   prompt 32–512/decode 16–256）：连续批处理平均延迟 90 vs
   268 ms、吞吐 45k vs 17k tok/s，静态批空转 5646 token 槽步；
   等长负载差距收窄（收益来自长度方差）；KV 预算 2k→32k 吞吐
   15k→52k 单调、峰值驻留守约。简化（prefill 一步完成、预算全额
   预留无抢占）写入正文。接入 ch07/05 边界段改写、ch01/05 声明
   同步、附录、练习 chunked prefill 挑战题。

## 验证

- 三 crate `cargo test --locked --offline`：8 + 6 + 5 全部通过；
  clippy（--all-targets）零警告；`cargo fmt --all --check` 通过；
- 三个主程序 `cargo run` 输出与正文引用一致；
- `mdbook build/test book`、`check_release.py --require-built-book
  --json`（`ok=true`、`errors=[]`、`warnings=[]`）、
  `git diff --check` 通过。

## 边界与偏差

- 队列模型与 PTQ 均为「协议/成本模型」证据类：解释机制，不代表
  低精度 backend 或真实服务 runtime 的行为与速度（正文已标注）。
- D 的两个断言在首版实现中失败过（整体 MSE 与 per-channel 整体
  差距），修正为按误差结构分解的断言——失败本身成为教学内容。
- 深度菜单剩余项：F 真实数据集训练（需可选下载路径决策）、GEMM
  阶梯第 3–5 级、迷你 tape 向量化等，见练习与 STATUS。

## 交接

推送后抽查 ch04/07 §10、ch07/04 动手小节、ch07/05 动手版三处
渲染。下一步候选：F（MNIST 级可选下载训练案例）或把三个协议模型
的输出接入对应练习的自查参考。
