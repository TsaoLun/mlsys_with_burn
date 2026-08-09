# 2026-08-09 学习者文风改写

## 目标

按「正文只面向学习者」盘点结论，完成三批改写：章首/练习模板、热点页与
lab 体例、快照元叙述降频与来源节清理。

## 操作

1. 九章着陆页 + 综合实验：`## 证据状态` → `## 本章你能验证什么`，引导句
   改为学习者口吻；比较卡链接改为「横向主题比较」。
2. 九章练习前言：去掉 `` `可选平台实验` / 默认 CPU CI `` 句。
3. 同步 `tools/check_release.py`、`docs/AUTHORING.md`、
   `docs/TERM_GLOSSARY.md`、`book/src/glossary.md`、`crosswalk-guide.md`；
   新增 D020。
4. 重写首页 README、ch02 工作流契约七连（改为阶段表）、ch01–ch09 实验
   节的验收腔（「你会学到什么 / 你会观察到 / 本实验刻意不做」）。
5. 降频「固定快照 / pins.toml / 固定 revision」；练习 tip 统一指向章末
   源码入口；来源节 `planning/chapter-sources` 与 D010/D011 改为对照导读
   与白话说明。
6. 软化 `running-examples.md` 贡献者门禁段与 `attribution.md` 发布腔。

## 验证

见 STATUS 本次交接。

## 下一步

提交本批修订；推送后确认 Pages 章首标题与练习前言已更新。
