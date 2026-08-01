# P0/P1 OpenMLSys 对照发布

## 目标

以固定 OpenMLSys v1 revision 为逐文件对照基线，让九章候选版能够说明：
原作主题、本书重组方式、固定 Burn 栈证据、CPU/协议观察、不可直接比较的
硬件条件和剩余范围差异。默认路径保持 CPU-first，不把 GPU、NCCL、ONNX、
DDP、DQN/MARL 或集群控制面写成已验证 parity。

## 固定事实

- OpenMLSys：`9c289782ccbb165ac8ad7c960ecffc12942a5560`
- Burn：`976aa9c5ec1d2dd3412710f99759e3c44bdff03d`
- CubeCL：`be278a1e76aed881e2cc6b165414ee6103ca4634`
- CubeK：`f82a6d07ebf35a1d446893b32712458744d80f13`
- burn-onnx：`af2dfb43af43bf363dc2d7d858d933d86e2a65a8`
- burn-onnx 的 Burn 关系：`78f10aec1ca6c6ffb1edd17a0fa131ae59ad5403`

## 已完成的修改

### P0 对照与发布门禁

- 新增 `planning/comparison/openmlsys-v1-crosswalk.md`，覆盖固定
  OpenMLSys v1 核心 Markdown、扩展篇排除清单、Burn/CubeCL/CubeK/
  burn-onnx 源码入口和 C/S/R/L/E 验收卡。
- 更新 `planning/CHAPTER_MATRIX.md`、九份 `planning/chapter-sources/`
  和九章入口的证据状态。
- 新增 `tools/check_release.py`：SUMMARY/八小节、未收录 Markdown、
  include/anchor、source crosswalk、pins/Cargo.lock、许可证、书内链接、
  数学公式、生成 HTML MathJax、Git hygiene 和 Cargo offline metadata
  检查；`--check-local-sources` 额外检查本地镜像路径与 HEAD revision。
- `Makefile` 和 CI 增加 `--locked`、`--offline` gate、doctest、mdBook
  snippet test、十个既有 CPU 示例 smoke、capstone smoke 和 release audit。
- 新增 `release.toml`；CI checkout/rust-toolchain/mdBook actions 固定到
  完整 commit SHA。
- 更新中英文 README、书内 README/attribution、NOTICE、AUTHORING 和
  glossary，写明九章候选版、工具版本、快照、burn-onnx revision、
  MathJax CDN 边界和非官方关系。

### P1 可运行证据

- 新增 `examples/ch05-ch07-capstone` 和 `book/src/capstone-p1.md`：
  20 个二维回归样本、16/4 split、`PartialDataset`/`MapperDataset`、
  DataLoader/Batcher、autodiff/SGD、`model.valid()`、ModuleRecord bytes、
  恢复后 inference 和错误 topology `RecordError::Validation`。
- 第 2 章增加 `detach().require_grad()` 负向 tape 实验，断言原始 leaf 的
  `None` 梯度、detached leaf 的数值/shape/Option 状态。
- 第 4 章增加重复 `add → mul → exp` 计划/输出观察和
  `BURN_FUSION_LOG=full` 可选日志路径；不把 cache hit、Fusion block、
  launch count 和 wall-clock 混为一谈。
- 第 3–9 章新增 `book/src/comparison-cards.md`。第 5–9 章的现有示例
  增加了 shard/epoch/backpressure、weighted collective/staleness/quorum/
  pipeline/checkpoint、artifact manifest/checksum/rollback/dynamic batch、
  policy freshness/joint transition 和 versioned machine-readable trace
  的纯 Rust 协议测试。
- `SUMMARY.md` 增加贯穿实验、比较卡和第 1 章第八小节；章节实验链接回
  capstone/比较卡。
- `mdbook test` 中依赖上下文的 include 片段显式使用 `rust,ignore`；
  示例 crate 测试仍是唯一可执行真相。

## 失败尝试与修正

1. 初次 `tools/check_release.py` 发现第 1 章只有七个小节，新增
   `ch01/08-comparison-and-sources.md` 并补入 SUMMARY。
2. 初次 crosswalk 审计发现三份 OpenMLSys preface 文件未列入，加入明确
   排除清单。
3. 初次 license/link 审计发现 NOTICE 未写代码许可证、书内 crosswalk
   链接会落到 mdBook 源树外；补充 NOTICE，并把书内入口改为项目路径文本。
4. capstone 初版学习率对未缩放特征发散；将固定特征缩放到小范围，保留
   确定性协议后 loss 稳定下降。
5. 初次 `mdbook test book` 暴露教学 include 片段不是独立 crate，逐个给
   依赖上下文的 Rust fence 加 `ignore`；同时修正 capstone 从 `book/src`
   到根 `examples/` 的 include 相对路径。

## 已核验命令

已通过：

- `python3 tools/check_release.py --json`
- `python3 tools/check_release.py --check-local-sources --json`
- `mdbook build book`
- `mdbook test book`
- `cargo metadata --locked --offline --format-version 1 --no-deps`
- `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`
- 受影响章节和 capstone 的 `cargo test`、Clippy、CPU run；
- 第 2/4 章实验的 `BURN_FUSION_LOG=full` run；
- `git diff --check`。

终验收随后已通过 `make check`、`make check-local-sources` 和全书最终
release audit；输出摘要已同步到 `planning/STATUS.md`。

## 交接

下一步由发布者审阅 crosswalk 和机器可读 release 输出并决定候选 tag/
归档。真实 GPU/网络/旧 burn-onnx revision 轨道继续保持独立的可选扩展；
默认 CPU gate 不因这些轨道放宽。
