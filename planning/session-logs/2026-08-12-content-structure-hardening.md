# 2026-08-12 内容合理性与结构加固

## 目标

用户要求在不扩张发布工程的前提下，尽本轮所能优化学习项目，随后明确优先
处理“内容（合理性与深度）和结构”。本日志只记录可审计事实、操作、验证
与交接，不包含隐藏推理。

## 审计结论摘要

- 若干正文断言与实际示例不一致：第 2 章 detach 输出、第 5 章 values/
  progress 输出、第 6 章 final loss 口径、第 8 章 replay/TD 关系、
  第 9 章 per-step collective 成本。
- 三张核心 SVG（第 2/4/7 章）含控制字符和乱码，可能影响浏览器/XML 渲染。
- 章首页导航体例不一致，部分阅读路径与先修条件冲突；若未说明，读者会把
  源码入口误认为本地必须存在的 checkout。
- 多处练习提示指向无关章节；默认 Cargo 命令未统一 `--locked`。
- `make check` 在本机被 `tracel-llvm-bundler v22.1.4-5` 下载
  `macos-x64.checksums.json` 的上游 404 阻断；这与本次改动无关。

## 修改

### 实验与概念口径

- `examples/ch02-tensor-basics/src/main.rs`：打印 detach 原始/新 leaf
  梯度，与正文输出块一致。
- `examples/ch05-data-pipeline/src/main.rs`：打印单 worker 的预处理值和
  进度，与正文一致。
- `examples/ch06-training-loop/src/lib.rs`：`final_loss` 改为训练结束后
  用最终参数重新 forward；正文说明它是最终模型评估，不是最后一次更新前
  观察值。
- `examples/ch08-rl-rollout/`：动作序列先触发自然 done，再走截断；
  报告拆分 `done_transitions`/`truncated_transitions`；固定 Q reference
  为 `1.2125 ± 1e-6`；正文明确 replay sample 只验证 batch shape 与存储，
  不是 TD 更新输入。
- `examples/ch09-cluster-simulator/`：`gradient_bytes` 改为
  `gradient_bytes_per_step`；每个执行 step（含失败前和 replay）累计
  collective 时间与跨 rack bytes；新增 step 倍增和 failure+replay 精确
  成本测试。

### 结构与读者路径

- 第 1–4 章首页补齐可点击「小节」列表；九章末尾增加下一章/综合实验桥接。
- 第 1 章系统与框架开发者路径改为第 1–7 章后进入第 8、9 章。
- `running-examples.md` 增加“阅读固定源码”说明：示例不依赖本地镜像；
  源码入口按 `pins.toml` 检出，禁止改 path 依赖。
- 综合实验页增加四段阅读提示和一个最小学习者改动任务；代码函数内加入
  四个 stage 注释。
- 修复第 3/4/7/8/9 章明显错配或空泛的练习提示。
- 第 3 章统一「算术强度」术语，清理几处「本书所用」前的异常空格。
- 正文默认示例命令统一补 `--locked`。

### 资产与检查

- 重写 `ch02-tape-vs-fusion.svg`、`ch04-pass-fusion-runtime.svg`、
  `ch07-deploy-loop.svg` 的损坏文本；保持原有教学意图。
- `tools/check_release.py` 新增：
  - `cargo-command-locking`：书内 package 级 `cargo test/run/check` 必须
    带 `--locked`；
  - `svg-assets`：所有书内 SVG 必须是 UTF-8、无 XML 禁止控制字符、可被
    标准 XML parser 解析。
- 更新 `planning/chapter-sources/ch01.md`–`ch09.md` 与
  `planning/STATUS.md`。

## 验证

- `cargo fmt --all --check`：通过。
- `cargo test -p ch02-tensor-basics -p ch05-data-pipeline
  -p ch06-training-loop -p ch08-rl-rollout -p ch09-cluster-simulator
  -p ch05-ch07-capstone --locked`：通过。
- 受影响 crate 的 `cargo clippy ... --all-targets --locked -- -D warnings`：
  通过。
- `cargo run` 观察第 2、5、6、8、9 章示例：输出与正文口径一致；第 8 章
  打印 `done_transitions=1 truncated_transitions=1 initial_right_q=1.2125`。
- `mdbook build book`、`mdbook test book`：通过。
- 18 张 SVG 的 UTF-8、控制字符与 XML 解析检查：通过。
- `python3 tools/check_release.py --require-built-book --json`：
  `ok=true`、`errors=[]`、`warnings=[]`，包含新增两项检查。
- `git diff --check`：通过。
- 完整 `make check`：未完成；在 workspace Clippy 阶段被
  `tracel-llvm-bundler v22.1.4-5` 请求
  `https://github.com/tracel-ai/tracel-llvm/releases/download/v22.1.4-5/macos-x64.checksums.json`
  返回 404 阻断。`curl -I -L` 同步确认该 URL 为 404。需要在上游资产恢复
  或 CI/其他可达环境中重跑完整 gate。

## 偏差

- 本次未新增章节或扩题；重点是修复会误导读者的语义和结构闭环。
- 未处理需要管理员在线完成的 Pages Settings、tag 与 GitHub Release。
- 本机完整 `make check` 受外部 LLVM 资产 404 影响；不能据此宣称当前
  环境通过了完整默认 gate。

## 下一步

1. 提交并推送本批修订。
2. 推送后抽查 Pages 的章首页小节、三张 SVG、第 8/9 章实验页与综合实验页。
3. 在固定资产可访问环境重跑 `make check`，再决定是否创建候选 tag/发布
   归档。

## 2026-08-12 下午追加：第 8 章回放驱动学习加深（读者向）

### 目标

回应“内容深度是否有提升”：在不扩张广度的前提下，把第 8 章实验从
“replay 只验证 shape”推进到“learner 真正从 replay batch 学习”，并
保持正文为学习者口吻（D020/D021）。

### 修改

- `examples/ch08-rl-rollout/src/lib.rs`：
  - 提取 `scheduled_action` 与 `collect_rollout`；
  - 新增 `ReplayUpdateReport` 与 `run_replay_driven`（阶段 B：先收集、
    不学习，再按 round 对 `buffer.sample` 的 batch 应用同一 TD 规则）；
  - `RolloutError` 增加 `ZeroUpdateRounds` 与 `Readback`；
  - 测试新增 3 项：capacity=1 的 retained transition 字段对齐、
    capacity=1 时 `initial_right_q == 0.0` 的确定性对照、配置错误边界；
    总计 9 项测试。
- `examples/ch08-rl-rollout/src/main.rs`：打印 online / replay cap=1 /
  replay cap=6 三行对照（第三行标注随机可变）。
- `book/src/ch08/07-rollout-lab.md`：开头图拆成阶段 A/B；第 1 节记录
  `done || truncated` 合并写入 replay 的语义边界；新增第 4 节
  「用 replay batch 驱动更新」（概念先行：数据顺序 vs 采样池/覆盖/重复
  学习）；第 5 节输出块更新为三行并说明哪些值是固定 reference。
- `book/src/ch08-rl-systems.md` 章首页与 `08` 练习页同步：新增一道
  在线 vs 回放对照概念题；系统结论补“容量与采样分布是 learner 的数据
  边界”。
- `planning/chapter-sources/ch08.md`：修订 2026-08-12 条目，去掉
  “replay 不接回 learner”的过时表述，记录两阶段设计与精确 reference。

### 验证

- `cargo test -p ch08-rl-rollout --locked`：9 项通过。
- `cargo clippy -p ch08-rl-rollout --all-targets --locked -- -D warnings`：
  通过（`scheduled_action` 改用 `is_multiple_of`）。
- `cargo run -p ch08-rl-rollout --locked`：
  `phase=online ... initial_right_q=1.2125`；
  `phase=replay capacity=1 ... initial_right_q=0.0000`；
  `phase=replay capacity=6 ...`（随机，非负有限）。
- `mdbook build book`、`mdbook test book`：通过。
- `python3 tools/check_release.py --require-built-book --json`：
  `ok=true`、`errors=[]`、`warnings=[]`。
- `cargo fmt --all --check`、`git diff --check`：通过。

### 教学要点（供后续维护参考）

- 深度来源是“同一算法、不同数据路径”的因果对照：在线路径按产生顺序
  逐条学习；回放路径从 capacity 窗口随机抽样，顺序打乱、旧数据被覆盖、
  同一条可重复学习。capacity=1 把“容量截断数据分布”变成精确观察
  （`initial_right_q` 恒为 0），而不是比喻。
- capacity=1 同时让随机 sample 变确定，从而能写字段级精确断言；
  capacity=6 的对照只声明随机可变，避免把随机输出钉成 reference。

## 2026-08-12 傍晚追加：读者语言可读性扫尾

### 方法

按 CI/门禁/audit/D 编号/P0/P1/crosswalk/smoke/pins/workspace/上游/回归/
reference/快照等模式扫描 `book/src` 全部 Markdown，逐项判定“读者必需或
领域通用”还是“维护者泄漏”。

### 修复

- `capstone.md`：裸「reference」改为「参考结果」。
- `ch08/07-rollout-lab.md`：下午新写内容中的「bootstrap 的 reference」
  「回归观察值」「更新前两条的 reference」改为「对照实现/对照值」；
  「第 5 节讨论 off-policy 编排」有歧义（与本页第 5 节撞名），改为
  指向 `05-learning-and-off-policy.md` 的明确链接；两处「随机 sample」
  改为「随机采样」。
- `running-examples.md`：`make check` 段落改为先声明「按章学习不必全
  仓库检查」，再限定「修改示例代码后」使用。
- `ch01/06`「上游」→「源码仓库」；`ch01/07`「本地固定上游」补充指向
  running-examples 的「阅读固定源码」说明；四章练习页「固定上游中的
  权威入口」统一为「本书固定版本源码中的权威入口」；`ch02/07`「固定
  上游」同步改写并补「回归测试」全称。

### 有意保留（读者必需或领域通用）

- `pins.toml` / `checkout` / `revision`：仅出现在 running-examples 与
  阅读路径的动手指导中，读者对照源码必须操作它们。
- `workspace`：ch01/ch06/ch07 均指 GPU 工作区内存（cuDNN 语义），非
  Cargo workspace。
- `host reference`：术语表已有定义行；ch03 首次出现处有中英定义。
- 「镜像」（ch09）：容器镜像，领域术语。
- 「固定 Burn 快照 / 本书固定版本」：D020 规范措辞。
- 附录与 `planning/` 中的 crosswalk、C/S/R/L/E、门禁等：D021 有意后置。

### 验证

- `mdbook build book`、`python3 tools/check_release.py
  --require-built-book --json`（`ok=true`）、`git diff --check` 通过。

## 2026-08-12 傍晚追加：SVG 中文问号事故与修复

### 事故

上午用专用文件写入工具重写三张损坏 SVG 后，用户发现图里全是问号。
确认：写入工具把多字节中文字符逐个替换成了 ASCII `?`。由于 `?` 是
合法 XML/UTF-8 字符，上午新增的 `svg-assets` 检查（UTF-8 + 控制字符 +
XML 解析）无法拦截——这是 2026-08-08 会话已记录过的同类坑的复发：
当时损成乱码+控制字符，本次损成问号。

### 修复

- 改用 `python3` 直接以 UTF-8 写盘三张 SVG（项目历史上验证过的做法），
  写后读回确认中文完好。
- 对全部 18 张 SVG 做严格扫描（U+FFFD、`??`、控制字符、CJK 计数）：
  其余 15 张原本健康，三张重写图修复后健康。
- `check_release.py` 的 `svg-assets` 加固：拒绝 U+FFFD 替换字符与连续
  `??`（两者都是多字节文本被损坏的可靠信号）。

### 教训（后续新增/修改含非 ASCII 文本的资产时必须遵守）

1. 含中文等非 ASCII 字符的 SVG 不要用通用文件写入工具直接写；用 Python
   以 UTF-8 写盘，或写后立即读回核对。
2. 校验资产时不只查“解析是否成功”，要查“预期的非 ASCII 内容是否还在”；
   `?` 与 U+FFFD 都是合法字符，解析器不会报错。
3. 校验脚本中不要写 U+FFFD 字面量——某些传输通道会剥离它，导致
   `''` 空串假阳性；用 `chr(0xFFFD)` 构造。

## 2026-08-12 傍晚追加：第 9 章三级通信域深化（参考原作）

### 目标

继续内容深化。选择缺口：`Gpu.node` 字段原对放置和成本无影响，而
OpenMLSys v1 `chapter_distributed_training/cluster.md` 的节点内互连
（NVLink 600 GB/s vs PCIe 4.0 64 GB/s vs HBM 1935 GB/s，A100 代际）与
机柜间链路（以太网 10–25 Gb/s、InfiniBand 100–200 Gb/s、超额认购
1:4–1:16）正好解释了为什么节点内必须单独成档。

### 修改

- `examples/ch09-cluster-simulator/src/lib.rs`：
  - `NetworkModel` 增加 `cross_node_multiplier`；`new` 保持三参兼容
    （node 档默认 2），`with_multipliers` 支持全自定义；
  - `communication_cost` 按 (rack,node) 区分同节点/同机柜跨节点/跨机柜
    三类 pair，penalty 分档；`CommunicationCost` 新增 `cross_node_bytes`；
  - `choose_placement` 的 TopologyAware 改为同节点 → 同机柜 → 跨机柜；
  - 新测试 2 项（三档精确成本 21/22/24us、node 优先放置），共 10 项。
- 正文：`ch09/02` 新增「为什么节点内值得单独一档：数量级直觉」（原作
  数字标注为写作代际的量级直觉，统一 Gb/s 与 GB/s）；成本函数段与放置
  顺序改三级；`ch09/07` 通信成本改三档表并说明 (rack,node) 节点身份；
  章学习目标、系统结论、练习与扩展任务同步。

### 验证

- `cargo test -p ch09-cluster-simulator --locked`：10 项通过；
  Clippy `-D warnings` 通过；`cargo run` 输出与既有 reference 一致
  （默认参数向后兼容）。
- `mdbook build/test book`、`check_release.py
  --require-built-book --json`（`ok=true`）、`cargo fmt --all --check`、
  `git diff --check` 通过。

## 2026-08-12 傍晚追加：第 7 章 Burnpack 字节级深化

### 评估背景

用户提出“本书是否缺少深入 Burn 的 MLSys 分析”。抽查结论：各章已有
源码机制级纵深（ch02 tape 生命周期、ch04 OperationIr/stream/fuser、
ch07 dtype policy 与威胁模型等），典型未做的纵深形式是“字节级格式
契约”与“字段级调用链”。选择前者：具体、CPU 可验证、部署章核心。

### 修改

- `examples/ch07-record-roundtrip/src/lib.rs`：新增
  `inspect_burnpack_layout`（手写小端 header 读取，不用 serde）、
  `BurnpackLayout`、`LayoutError`、`sample_record_bytes`；测试覆盖
  `NRUB` magic、版本、对齐不重叠、截断与坏 magic；共 5 项测试。
- `examples/ch07-record-roundtrip/src/main.rs`：第二行打印
  `burnpack magic=NRUB version=1 metadata_bytes=133 data_section_start=256
  total_bytes=516`。
- `book/src/ch07/07-record-roundtrip-lab.md`：新增第 3 节「打开
  Burnpack 字节」（格式规格、`BURN`→`NRUB` 字节序细节、256 对齐的
  mmap 动机、DoS 上限），并用本机输出算对齐开销；原 3–5 节顺延。
- `book/src/ch07/08-exercises-and-sources.md`：源码入口补
  `burn-pack/src/base.rs` 与 `writer.rs`；`ch07-model-serving.md`
  学习目标同步。

### 验证

- `cargo test -p ch07-record-roundtrip --locked`：5 项通过；Clippy
  `-D warnings` 通过；`cargo run` 输出与正文一致。
- `mdbook build/test book`、`check_release.py
  --require-built-book --json`（`ok=true`）、`cargo fmt --all --check`、
  `git diff --check` 通过。

### 注意

- `metadata_bytes=133`/`total_bytes=516` 由固定版本的 CBOR 字段与对齐
  规则决定；正文已注明升级版本线时需重新核对，不作为跨版本协议。

## 2026-08-12 夜追加：第 2 章张量字节视图深化

### 目标

回应“前几章的 Burn 相关内容可以进一步深化吗”。评估后选择第 2 章的
存储层缺口：正文讲 shape/dtype 是运行时属性，但读者无法观察张量在内存
中的实际字节；同时与第 7 章 Burnpack 字节分析形成同一 `Bytes` 的呼应。

### 修改

- `examples/ch02-tensor-basics/src/lib.rs`：新增 `TensorBytesReport` 与
  `inspect_tensor_bytes`；测试断言 `1.0f32` 小端字节 `[00,00,80,3F]`、
  `-0.0` 符号位 `[00,00,00,80]`、6×f32=24 字节、`convert_dtype(F64)`
  后 48 字节；共 8 项测试。
- `examples/ch02-tensor-basics/src/main.rs`：打印张量字节行。
- `book/src/ch02/02-tensor-device-backend.md`：新增「字节视图：bytes、
  shape 与 dtype 的分工」概念小节（含固定源码中同宽度原地/跨宽度克隆
  的事实、与 Burnpack 同一 `Bytes` 的呼应）。
- `book/src/ch02/07-labs.md`：新增第 3 节「字节视图」，后续小节编号
  顺延（3→4 … 10→11）；输出块与测试描述同步。
- `book/src/ch02-programming-and-graph.md`：学习目标同步。

### 验证

- `cargo test -p ch02-tensor-basics --locked`：8 项通过；Clippy
  `-D warnings` 通过；`cargo run` 输出与正文一致。
- `mdbook build/test book`、`check_release.py
  --require-built-book --json`（`ok=true`）、`cargo fmt --all --check`、
  `git diff --check` 通过。

### 源码依据（固定 revision）

- `burn-std/src/data/tensor.rs`：`TensorData { bytes, shape, dtype }`、
  `as_bytes()`、`convert_dtype()` 的同宽度原地/跨宽度克隆分支。
- `burn-std` 的 `Bytes` re-export 自 `cubecl-environment`；其实现
  `Deref<Target=[u8]>`。

## 2026-08-12 夜追加 2：第 3 章 matmul 逐层走查

### 目标

继续深化，把上次评估中列出的“字段级调用链走查”做掉：一次
`Tensor::matmul` 从用户 API 到 CubeK Strategy 的六层路径。

### 修改

- `book/src/ch03/04-cubek-and-burn.md`：第 2 节在高层箭头链后新增
  「逐层走查：一次 matmul 经过的六个决策点」，每层给文件路径、固定的
  真实行为和该层回答的问题；收束于“最早两处性能优化都是 kernel 存在
  之前的纯元数据变换”（vec-mat 重解释、broadcast-rhs 折叠共享 handle）。
- `book/src/ch03/08-exercises-and-sources.md`：源码题 3 的错配提示
  （原指向 GPU 并行层次节）改为指向走查小节，并要求说出六层各自的
  问题；延伸阅读补 `numeric.rs`、`ops/tensor.rs`、`launch.rs`、
  `strategy.rs`。
- `book/src/ch03-accelerator.md`：学习目标同步。

### 固定源码事实（已逐层核实）

1. `burn-tensor .../numeric.rs` `Tensor::matmul`：`TensorCheck::matmul`
   校验；`[...,B,1,K] @ [...,1,K,N]` 经 `swap_dims` 重解释。
2. `burn-cubecl/src/ops/tensor.rs` `float_matmul`：
   `MatmulStrategy::default()`（autotune feature 决定 Autotune/Cube）；
   trait 签名无 Result，配置错误在此处 `unwrap`。
3. `burn-cubecl/src/kernel/matmul/base.rs`：`init_matmul_output`；
   broadcast-rhs 折叠为单个 matmul（源码注释 “Pure metadata”）；
   量化输入拆 data/scale 的 `InputBinding` 与 `dtype_to_storage_type`。
4. `cubek-matmul/src/launch.rs` `launch_ref` 仅转发
   `strategy.launch_ref`。
5. `cubek-matmul/src/strategy/strategy.rs` `Strategy` 枚举：Naive、
   CpuGemm、Simple/CMMA/MMA、double buffering、ordered、specialized、
   TMA、VecMat 与 unit 变体。

### 验证

- 本节为阅读走查，无新增示例代码；`mdbook build/test book`、
  `check_release.py --require-built-book --json`（`ok=true`）、
  `git diff --check` 通过。

### 验证

- 严格扫描 18 张 SVG：ufffd=0、qq=0、ctrl=0，CJK 计数符合各图内容。
- `mdbook build book`、`check_release.py --require-built-book --json`
  （`ok=true`）、`git diff --check` 通过。
