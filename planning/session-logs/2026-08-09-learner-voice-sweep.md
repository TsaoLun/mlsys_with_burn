# 2026-08-09 读者口吻扫尾

## 目标

复核后清理主路径残留的 CI / 验收 / D0xx / 源码核验 / 根 workspace 等
审计腔，使九章正文与综合实验面向读者。

## 操作

1. 综合实验「验收协议」→「你需要核对什么」。
2. ch1/3/6 等：CI → 默认示例/默认测试；非默认 CI → 非默认示例。
3. ch7/8：去掉 D010；「本项目/根 workspace/源码核验」→ 本书示例用语。
4. `running-examples.md`：去掉决策编号；`make check` 仅留给改示例/补丁读者。
5. 附录证据标签按 D021 保留。

## 验证

`mdbook build book`；`python3 tools/check_release.py --require-built-book --json`
→ `ok=true`。

## 下一步

提交推送。
