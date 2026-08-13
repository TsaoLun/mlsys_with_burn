# 2026-08-13 全书审计修复批次（P0–P3）

## 目标

执行前一轮全书审计（教科书化 + 机制深化两条主线）给出的修复清单。
修复原则：不只是删改，每一处尽量顺手加深机制呈现——否定墙改成
「归属分工表」、ASCII 换成信息更完整的 SVG、解剖页补真实源码摘录。

## P0 错误级（8 处）

- ch07/04 收尾段与新 PTQ 实验自相矛盾 → 改写为「两个实验各管一段
  （Record 语义 / 校准协议层），缺席的是低精度 backend benchmark」。
- ch03/07 「真实共享内存 Kernel 仍属练习」与第 8 节 GEMM 阶梯矛盾 →
  指向第 8 节可选实验。
- references.md Orca 条目「本书标注为未覆盖」与队列实验矛盾 →
  改为「`ch07-serving-queue-sim` 演示的正是这一机制」。
- ch08/08 四处 + ch07/08 一处失效标题引用（标题去「固定」后提示未
  跟上）→ 与实际标题对齐。
- ch04/02 「亲手写一个 Pass」两段近重复拼接 → 合并为一段。
- ch07/01 SVG 与同图 ASCII 并存 → 删 ASCII。
- op-anatomy「三个标本」只贴两段代码 → 补 `matmul` 真实摘录：
  `State = (Option<NodeId>, Option<NodeId>, BinaryOpsBroadcast)`，
  并写明「lhs 只在 rhs 被追踪时才 checkpoint」（burn-autodiff
  `tensor.rs` float_matmul 逐行核对）。

## P1 体例级

- 定语式「固定 Burn/CubeCL/API/实现/仓库」约 35 处收敛为直接名称；
  页首「声明一次」的用法（如 ch05/08、ch09/08 开头）保留。
- 三面否定枚举墙改为分工表：ch06/06（完整分布式训练还差哪几层、
  归属层、与 DDP 入口的关系）、ch09/04（集群级能力 × 控制面/数据面
  × 通信入口的假设）、ch09/06（容错闭环环节 × Burn 提供 × 缺口由谁
  补）；ch09/08 源码入口段改为「模拟器建模的正是这段协议层」。
- ch09 第 2–6 节小结去同款「泼冷水」收尾，改为正向机制归属陈述；
  边界声明集中在第 1 节与章末。
- 附录整顿：appendix-scope-and-evidence 删 D024 编号泄漏与备忘句、
  C/S/R/L/E 与证据标签只定义一次（其余处引用）、比较卡 7 个 `##`
  降为 `###`、两个「如何使用」合并为「如何使用本附录」、
  「发布级比较结论」措辞消除；appendix-sources 许可证声明 9 处
  合并为开头 1 处、「文本架构图」过时表述更新为 SVG 说法。

## P2 结构级

- 双总结去重：ch06/08 小结删与结论逐字重复的第三段（改为指向
  ch06/06 分工表的链接）；ch09/08 小结第二段改为指向第 4/6 节
  分工表。
- 新增 5 张承重 SVG（沿用现有风格，均通过 XML 校验）：
  `ch01-dispatch-tree`（ch01/04 与 ch02/02 共用，消除两处重复
  ASCII）、`ch01-ml-ecosystem`、`ch03-memory-hierarchy`（同时删除
  与图几乎同义的表格）、`ch06-tape-to-optimizer`、
  `ch08-offpolicy-loop`。ch09-gpu-cluster 章首路线 ASCII 改一句话
  正文（下方已有控制/数据面 SVG）。ch02/04 三形态并存（text/SVG/
  表格）删 text 块。
- 注意：Write 工具写 `.svg` 会损坏中文编码，SVG 一律用 python 写入。

## P3 补全

- ch07/08 小结补 `ch07-ptq-calibration` 与 `ch07-serving-queue-sim`
  段落；系统结论 4/5 更新。
- 术语表部署组新增：训练后量化（PTQ）、校准、量化参数
  （scale/zero-point）、连续批处理、KV 缓存。
- 语病与措辞：ch02/01「是本书的实现路径替代」、ch02/07 语序、
  ch07/05「伪装/掩盖」审计腔、op-anatomy「验收标准」→「检验标准」。

## 验证

- `mdbook build book`、`mdbook test book` 通过。
- `cargo fmt --all --check` 通过。
- `python3 tools/check_release.py --require-built-book --json`：
  ok=true，无错误无警告。
- 图片引用扫描：无缺失引用、无未使用 SVG。
- 完整 `make check` 仍被本机 `tracel-llvm` `macos-x64.checksums.json`
  404 阻断在 CubeCL CPU 构建（既有环境问题，见 STATUS 下一步 3）；
  本批未改任何示例代码，Rust 侧无需重测。

## 交接

- 下一步：提交推送本批；在能访问 tracel-llvm 资产的环境重跑完整
  `make check`；发布者决定候选 tag。
- 遗留（低优先）：其余小型 ASCII 图（ch01/04 CubeK 栈、ch06/01 等）
  可在后续批次按需转 SVG；ch05 边界重复 4 处未动（影响小）。
