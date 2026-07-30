# 项目架构

## 三层内容模型

```text
框架无关原理
    ↓ 用固定案例解释
Burn API 与训练/部署抽象
    ↓ 按需下钻
Burn IR/Fusion → CubeCL → CubeK/设备运行时
```

教材不能从上游 README 拼接而成。每节先定义问题、约束和成本，再用固定
版本源码验证实现，最后以章节示例让读者观察行为。

## 真相来源

- 内容范围：`planning/CHAPTER_MATRIX.md`
- 进行中工作：`planning/STATUS.md`
- 版本：`pins.toml`
- Burn 行为：固定 commit 的源码和测试
- OpenMLSys 来源：固定 commit 下的 v1 中文章节
- 可执行结论：`examples/` 中通过测试的代码

若文档、源码和上游在线页面冲突，以 `pins.toml` 对应源码为当前教材事实，
并将新版本差异记录为后续升级事项。

## 目录职责

- `book/` 只保存叙事、公式、图和对示例源码的 include。
- `examples/` 保存可编译代码及测试，是代码片段的唯一真相。
- `tools/` 保存项目级校验，不复制上游构建系统。
- `planning/` 保存跨会话协作状态，不承载正文。
- 五个上游目录用于研究和本地核验，始终保持独立。

## 版本升级流程

1. 新建版本升级决策记录。
2. 同时核对 Burn、CubeCL、CubeK、burn-onnx 的依赖 revision。
3. 更新 `pins.toml` 和示例 workspace dependency。
4. 运行完整检查，记录 API 和行为变化。
5. 只在一个写作里程碑边界合并升级。

