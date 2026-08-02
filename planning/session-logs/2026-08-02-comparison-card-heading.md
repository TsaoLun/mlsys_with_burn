# 比较卡章节标题与锚点修正

## 目标

将 `book/src/comparison-cards.md` 中的 `3. GEMM 与加速器` 等标题改成
明确的章节映射格式 `第 N 章：主题`，避免读者把数字理解为比较卡自身
从第 3 张开始。

## 修改

- 统一修改第 3–9 节标题：
  - `第 3 章：GEMM 与加速器`
  - `第 4 章：IR、Fusion、cache 与 launch`
  - `第 5 章：数据处理`
  - `第 6 章：分布式训练`
  - `第 7 章：模型部署`
  - `第 8 章：强化学习`
  - `第 9 章：GPU 集群与控制面`
- 根据 mdBook 生成的实际 slug，同步更新
  `ch03`–`ch09` 入口中的 7 个 `comparison-cards.md#...` 链接。
- 未新增第 1–2 章比较卡；其对照内容仍由对应章节和
  `planning/comparison/openmlsys-v1-crosswalk.md` 覆盖。

## 验证

- `mdbook build book`
- 生成 HTML 确认锚点为 `第-3-章...` 至 `第-9-章...`
- `make check`，退出码 0
- release audit：`errors=[]`、`warnings=[]`
- `git diff --check`

## 结果

书站内容和章节锚点保持一致，SUMMARY 导航和固定快照没有变化。
