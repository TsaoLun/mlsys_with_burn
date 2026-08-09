# 2026-08-09 项目自洽材料后移附录（D021）

## 目标

把五标签墙、对照/比较页、章末来源长文与 ch01 证据专页移出学习者主路径，
整包放入书末附录。

## 操作

1. 新建 `book/src/appendix-scope-and-evidence.md`、`appendix-sources.md`。
2. 删除章首「本章你能验证什么」、`ch01/08-comparison-and-sources.md`、
   `crosswalk-guide.md`、`comparison-cards.md`。
3. 章末来源改为指向附录的一句；workload card 练习并入 ch01/07。
4. SUMMARY：贯穿区只留综合实验；附录增加两页；第 1 章 7 小节。
5. 瘦身 README / attribution / glossary / capstone。
6. `check_release.py`：校验附录；ch01 允许 7 小节；不再要求章首五标签。
7. 新增 D021；更新 AUTHORING、TERM_GLOSSARY、book-authoring 规则、STATUS。

## 验证

见 STATUS 本次交接。

## 下一步

提交推送；确认 Pages 附录导航。
