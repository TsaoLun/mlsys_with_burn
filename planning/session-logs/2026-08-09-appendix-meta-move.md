# 2026-08-09 项目自洽材料后移附录

## 目标

将证据五标签、对照/比较页、章末来源长文与 ch01 证据专页整包移出学习者
主路径，收入书末附录（D021）。

## 操作

1. 新建 `book/src/appendix-scope-and-evidence.md` 与
   `book/src/appendix-sources.md`。
2. 删除九章/综合实验「本章你能验证什么」、`ch01/08`、
   `crosswalk-guide.md`、`comparison-cards.md`；章末来源改为一句指针。
3. 重排 SUMMARY：贯穿区只留综合实验；附录含术语表 + 两份新附录。
4. 瘦身 README / attribution；更新 `check_release.py`（附录校验、
   ch01=7 小节）、AUTHORING、TERM_GLOSSARY、D021、STATUS。

## 验证

见 STATUS 本次交接。

## 下一步

提交推送；确认 Pages 附录导航。
