# 2026-09-15：pre.3 正文路径、成本模型与门禁

## 目标

能力断言以 `pins.toml` 的 Burn / CubeCL / CubeK / burn-onnx pre.3 为准；
读者可见段落按系统课口吻改写；不削弱默认 CPU gate。

## 选择

- 1F1B：不改 `ch06-parallel-strategies` 数字。正文拆开「前向+反向合成
  \(t\)」与「F/B 分槽」两套数法；文献口径下空泡比相同，收益在激活显存。
- 第 9 章：不改模拟器公式。点名 \(2S\cdot\beta\) 表、模拟器 \(S(p-1)\)、
  环精确 \(2(p-1)S/p\) 三套模型。
- 不把 `ch04-fusion-inspector` / `ch03-cubecl-kernel` 移出
  `CPU_EXAMPLES`。Intel Mac 无 macos-x64 资产写进运行指南即可。

## 核验权威

四个只读镜像已 `fetch` 并对齐 pin（不提交这些仓库）：

- burn `13f0a12b`、cubecl `b566e954`、cubek `73743e34`、
  burn-onnx `fe36b3b6`

`AGENTS.md` 与 `pins.toml` 注释写明：pin SHA / crates.io 版本 → Cargo
checkout → 仅当 HEAD 等于 pin 的镜像。

## 正文要点

- CubeK：顶层 `Tiled` / `MultiLevel` / `Auto`；组件与 routine 在
  `multi_level/`；`tune_key.rs` 在 crate 根；`StorageType` → `ElemType`。
- autodiff：默认 `NoCheckpointing` 忽略 `memory_bound`；重算入口
  `Device::autodiff().gradient_checkpointing()`。
- 第 9 章 InfiniBand 换算、AllReduce 4 倍而非数量级、replay `[2,6)`。
- ONNX 路径补 `import/`；隔离理由改为版本已对齐、仍不进 workspace。
- 练习页：ch05–09 分类改为 `###`；去掉 ch08/09「本节小结」标题；
  章末 OpenMLSys 长路径与裸 SHA 收到附录指针。
- 过程词：章内「口径 / 固定源码 / 固定版本 / 权威入口 / 根 workspace /
  pre.3 / CI / 已核验 / 已交付」改为「本书所用版本」等讲义说法。
  `pins.toml` 只留在运行指南与附录账本。

## 工程

- `check_release.py` 对行内 `$...$` 调用 `formula_issues`。
- Makefile / `running-examples.md` 标明 `mdbook test` 不编译
  `rust,ignore`；tracel-llvm `v22.1.4-6` 无 macos-x64。
- `ch06-training-loop` 增补 NaN / Inf / 负学习率测试。
- 服务队列主程序打印区间改为 32–511 / 16–255。

## 验证

- `mdbook build book`
- `python3 tools/check_release.py --require-built-book --json`
  （`ok=true`、`errors=[]`）
- `python3 tools/check_upstreams.py --check-local`（镜像已对齐）
- `cargo test --locked -p ch06-training-loop --all-targets`（4 项）
- `cargo fmt --all --check`、`git diff --check`
- 未跑完整 `make check`（Intel Mac `cubecl-cpu` / tracel-llvm 404 仍在）

## 下一步

发布者审阅本批后推送；完整 `make check` 在 Linux 或 darwin arm64 上跑。
