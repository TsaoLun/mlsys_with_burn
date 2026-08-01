# 跨章术语表

本表约束第 1–4 章及后续章节的统一用语。写作与审校时以本表为准；
历史对照可以提及旧名，但正文叙述应使用“统一用语”列。

完整写作规范见 [`AUTHORING.md`](AUTHORING.md)。

| 概念 | 统一用语 | 禁止或慎用 |
|---|---|---|
| 用户张量类型 | `Tensor<D, K>`；秩 `D`，类别 `K` | 旧文档 `Tensor<B, D>` 仅作历史对照 |
| 设备选择 | Device；运行时分派到 DispatchDevice | 不把 Device 称作 Backend |
| 后端实现契约 | Backend / BackendTypes（实现层） | 不对读者说“换 Backend 泛型” |
| 默认 CPU 路径 | Flex（eager） | 不暗示 Flex 走 Fusion |
| 自动微分表示 | autodiff tape（一阶反模式） | 不称“Burn 计算图” |
| 融合表示 | Burn IR / OperationIr；Fusion 计划 | 不与 tape 混称 |
| CubeCL 表示 | Scope / KernelDefinition / CubeCL IR | 不称“计算图” |
| 设备重放 | backend / device graph capture | 仅在 Runtime 支持时提及 |
| 并行拓扑 | Cube / Unit / Plane | 正文主用 CubeCL 词，CUDA 对照放表内 |
| 算子库路径 | CubeK Blueprint–Routine；Strategy/Launch | 不把 Guide 说成四层架构 |
| 同步边界 | read / `Device::sync` 为完成边界；flush 为提交/推进 | 不把 flush 写成设备完成 |
| 状态保存 | ModuleRecord | 不写旧 `Record<B>` |
| 切断依赖 | `detach()`（保留 require-grad 意图） | 不写“detach 后不可求导” |
| 负载分析 | workload card；计算、数据、设备、目标四元组 | 不用模型名称替代系统负载 |
| 算术强度 | arithmetic intensity；FLOP/字节（教学模型可用 FLOP/加载元素） | 不把算术强度直接当成实测性能 |
| 数据供给模型 | $F/P/G$；读取、变换、设备消费速率 | 不把有界队列当成长期瓶颈修复 |
| 强化学习终止 | `done`（自然终止）与 `truncated`（外部/时间截断） | 不无条件把两者当作同一 bootstrap 语义 |
| 策略关系 | behavior policy $\mu$ / target policy $\pi$；on-policy/off-policy | 不由“有 replay”单独判断算法类别 |
| 采样更新架构 | Actor–Learner；actor 采样，learner 更新 | 不把 DDP gradient collective 叫 Actor–Learner |

## 三张地图（符号约定）

全书重复出现的三张文本图应使用同一层名，避免各章重新发明：

1. **系统分层**（第 1 章）：模型与训练程序 → 张量执行与 autodiff tape →
   Burn IR / Fusion → CubeCL / CubeK Kernel → 设备 Runtime；横跨路径为
   数据管道、训练执行、部署与通信。
2. **三种表示**（第 2 章）：autodiff tape、Burn IR / Fusion 计划、
   device graph capture。它们描述相似操作，但节点、生命周期与目的不同。
3. **编译栈**（第 4 章）：capture/register → analysis → transformation →
   lowering → code generation → compile/cache → allocate/schedule/launch →
   read/sync。
