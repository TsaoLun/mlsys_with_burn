# P1 贯穿实验与 OpenMLSys 比较卡复审

## 复审结论

- P1 应保留：第 5–7 章的单章实验分别验证数据、训练和 artifact；
  P1 增加一条可运行的跨边界状态路径，适合说明 Dataset、autodiff、
  `ModuleRecord` 和 inference 的连接。
- 比较卡应保留：它是读者面对的横向证据摘要，而
  `planning/comparison/openmlsys-v1-crosswalk.md` 仍是逐文件映射的真相；
  两者不能互相替代。

## 发现的问题

1. P1 原先用一个独立初始化的 `initial_model` 计算 `initial_loss`，再用
   另一个模型训练。即使 loss 下降，也不能严格证明同一初始状态经过 SGD
   得到改善。
2. loader 原先只检查样本数量、不重叠和 shape 第二维；重复 ID、错误
   split、batch 首维或 batch 数异常可能未被运行时协议捕获。
3. 比较卡对纯 Rust 协议 helper、教学 checksum 和 CPU 集群模拟器的
   边界虽有总体说明，但部分可运行观察容易被快速阅读者理解成 Burn
   runtime 或生产安全能力。

## 修改

- `examples/ch05-ch07-capstone/src/lib.rs`
  - 用训练模型第一次更新前的同一组参数计算 `initial_loss`；
  - 严格验证 train=`0..15`、validation=`16..19`，batch 数和完整
    `[4, 2]`/`[4, 1]` shape。
- `book/src/capstone-p1.md`、`planning/capstone-p1.md`
  - 同步验收协议、同一初始状态语义和 crosswalk/比较卡入口。
- `book/src/comparison-cards.md`
  - 声明 crosswalk 的权威关系；
  - 区分纵向 P1 与横向比较卡；
  - 标注协议 helper、非密码学教学 checksum 和 CPU 模拟器边界。
- `planning/DECISIONS.md`
  - 记录 D017。

## 验证

- `cargo fmt --all --check`
- `cargo test -p ch05-ch07-capstone --locked --offline`
- `cargo clippy -p ch05-ch07-capstone --all-targets --locked --offline -- -D warnings`
- 连续两次 `cargo run -p ch05-ch07-capstone --locked --offline`，输出一致
- `mdbook build book`
- `mdbook test book`
- `python3 tools/check_release.py --require-built-book --json`，`errors=[]`
- `git diff --check`
- IDE lint：受影响文件无诊断

## 下一步

启用 GitHub Pages source 并确认首次部署；候选 tag/归档仍由发布者决定。
