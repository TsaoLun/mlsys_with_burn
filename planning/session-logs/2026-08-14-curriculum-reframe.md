# 2026-08-14：读者主路径重编

## 会话目标

把读者正文从「可核对的 Rust MLSys」叙事改成系统课：OpenMLSys 的问题
清单、当代训练/服务/集群成本模型、以及打开 Burn crate 时的落点。
不把该定位写成口号。工程边界（pins、默认 CPU gate、D010–D013、D022）
不变。

## 决策

新增 D025。AUTHORING / MASTER_PLAN / CHAPTER_MATRIX / AGENTS.md 同步。

## 操作摘要

- 新增 `infra-map.md`（产业名字 ↔ 章节）、`crate-map.md`（改哪一层
  打开哪个 crate）。
- SUMMARY：地图前置；第 8 章标可选；篇名改为「服务、集群与扩展」。
- 重写前言、首页、归属、九章章首、九章系统结论。
- 加厚：ch06/06 的 DP/TP/PP/ZeRO；ch07/05 的 TTFT/TPOT/KV；ch09/01
  的 Slurm/K8s；ch01 阅读路径与大模型杠杆地图。
- 主路径去掉「刻意不做」；「本书固定版本」改为「本书所用版本」。
- 附录保留「固定版本」「结论靠什么支撑」（发布检查依赖）。
- 未新增 Rust crate。

## 验证

见 STATUS「本次交接」；本会话运行 `mdbook build book` 与
`python3 tools/check_release.py --require-built-book`。

## 偏差

80+ 小节的技术推导大部分保留，只改章首、结论、实验口吻和少数关键节。
并行策略模拟器与第二条 capstone 未做，列入 STATUS 下一步。
