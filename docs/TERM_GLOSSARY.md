# 跨章术语表

本表约束第 1–4 章及后续章节的统一用语。写作与审校时以本表为准；
历史对照可以提及旧名，但正文叙述应使用“统一用语”列。

完整写作规范见 [`AUTHORING.md`](AUTHORING.md)。

| 概念 | 统一用语 | 禁止或慎用 |
|---|---|---|
| 用户张量类型 | `Tensor<D, K>`；秩 `D`，类别 `K` | 旧文档 `Tensor<B, D>` 仅作历史对照 |
| 设备选择 | Device；运行时分派到 DispatchDevice | 不把 Device 称作 Backend |
| 分派桥 | BridgeTensor / Dispatch；路由到具体后端 | 不把桥接层说成第三种数学语义 |
| 后端实现契约 | Backend / BackendTypes（实现层） | 不对读者说“换 Backend 泛型” |
| 默认 CPU 路径 | Flex / `Device::flex()`（eager） | 不暗示 Flex 走 Fusion/CubeCL |
| Fusion CPU 设备 | `Device::cpu()`（需 `cpu`+`fusion`） | 不与 Flex 混为同一条证据路径 |
| 主机参考实现 | host reference；普通 host 代码的可观察正确结果 | 不把 host reference 等同于 Host/Device 机器模型或某个 Runtime |
| 计算客户端 | ComputeClient；buffer / launch / read 入口 | 不把 host 返回当成设备完成 |
| 回退路径 | fallback；仍正确的较简实现 | 不把 fallback 写成“失败”或与高性能路径成本等同 |
| 自动调优 | autotune / tune key；当前设备测量并缓存 | 不把 CPU 缓存命中外推到其他 Runtime |
| 特化键 / 编译键 | specialization / compile key | 不把任意运行时标量都塞进 comptime 键 |
| 融合器 | fuser；Fusion 接受/拒绝操作块的组件 | 不把 Inspector 的 fuser 名当成设备 Kernel 名 |
| Fusion 流 | Fusion stream / `StreamId` | 不与 CUDA stream 或集群作业队列混称 |
| ExecutionStrategy 同名 | Fusion 块内策略 vs `burn-train` 训练策略 | 不把两处同名类型当成同一个 API |
| 自动微分表示 | autodiff tape（一阶反模式） | 不称“Burn 计算图” |
| 融合表示 | Burn IR / OperationIr；Fusion 计划 | 不与 tape 混称 |
| CubeCL 表示 | Scope / KernelDefinition / CubeCL IR | 不称“计算图” |
| 设备重放 | backend / device graph capture | 仅在 Runtime 支持时提及 |
| 并行拓扑 | Cube / Unit / Plane | 正文主用 CubeCL 词，CUDA 对照放表内 |
| 算子库路径 | CubeK Blueprint–Routine；Strategy/Launch；LocalTuner | 不把 Guide 说成四层架构 |
| 同步边界 | read / readback / `Device::sync` 为完成边界；flush 为提交/推进 | 不把 flush 写成设备完成 |
| 训练进程号 | rank（分布式副本序号） | 不与张量秩（rank）混用 |
| 状态保存 | ModuleRecord | 不写旧 `Record<B>` |
| 切断依赖 | `detach()`（保留 require-grad 意图） | 不写“detach 后不可求导” |
| 负载分析 | workload card；计算、数据、设备、目标四元组 | 不用模型名称替代系统负载 |
| 算术强度 | arithmetic intensity；FLOP/字节（教学模型可用 FLOP/加载元素） | 不把算术强度直接当成实测性能 |
| 数据供给模型 | $F/P/G$；读取、变换、设备消费速率 | 不把有界队列当成长期瓶颈修复 |
| 分片与提交 | deterministic shard/offset；epoch commit；reorder buffer | 不把数据守恒写成全局保序 |
| 加权集合通信 | weighted AllReduce；按样本数聚合局部梯度 | 不把等权平均用于不等 batch |
| 梯度新鲜度 | gradient staleness；当前参数版本与梯度版本的差距 | 不静默接受无限 stale update |
| 流水线空泡 | pipeline bubble；1F1B 的 warm-up/drain 空闲槽 | 不把 micro-batch 数当作 stage 数 |
| artifact 清单 | artifact manifest；version/payload length/checksum | 不把 checksum 单独当作供应链安全 |
| 读者综合实验 | `Dataset → autodiff → ModuleRecord → inference` 的端到端 CPU 路径；项目 planning 中可标记为 P1 | 正文标题不单独暴露 P1 代号 |
| 逐文件对照矩阵 | OpenMLSys 文件、本书章节、源码入口和证据层级的映射；英文可写 crosswalk | 读者入口在附录 `appendix-scope-and-evidence.md`；不写成 OpenMLSys 或 Burn 的官方术语 |
| 证据分类 | 源码核验、CPU 可运行验证、协议/成本模型、可选平台实验、未覆盖 | 仅见于附录；不写成 Burn 官方能力等级或平台 parity；旧称“证据状态 / 本章你能验证什么”章首块仅见于历史记录 |
| 默认 CPU 路径 | CPU 可运行路径（CPU-first）；用于默认示例和发布门禁 | 不把 CPU 验证外推成 GPU、网络或集群能力 |
| 动态 batching | dynamic batching；同 shape 请求按容量成批 | 不跨 shape 拼 batch 或把队列等待算进 kernel |
| 强化学习终止 | `done`（自然终止）与 `truncated`（外部/时间截断） | 不无条件把两者当作同一 bootstrap 语义 |
| 策略关系 | behavior policy $\mu$ / target policy $\pi$；on-policy/off-policy | 不由“有 replay”单独判断算法类别 |
| 策略新鲜度 | policy freshness；behavior/target version lag | 不把 mock policy 版本当作算法实现 |
| 采样更新架构 | Actor–Learner；actor 采样，learner 更新 | 不把 DDP gradient collective 叫 Actor–Learner |
| GPU 集群 | GPU cluster；由 GPU、节点、机柜和网络域组成的资源系统 | 不把多张 `Device` 直接称作集群 |
| 控制面/数据面 | control plane 负责作业与资源；training data plane 负责 rank 间通信 | 不把设备 runtime 的 queue 当作集群控制面 |
| 成组调度 | gang scheduling；同步作业要么获得完整资源，要么等待 | 不允许部分 rank 先启动 |
| 拓扑感知放置 | topology-aware placement；按节点、机柜、链路域选择资源 | 不只按 GPU 数量或连续 id 放置 |
| 超额认购 | oversubscription；峰值流量超过共享链路容量 | 不把链路峰值需求写成实测可用带宽 |
| 资源碎片 | resource fragmentation；总量足够但无法满足成组/显存/拓扑约束 | 不用总空闲 GPU 数代替可成组容量 |
| 故障域 | failure domain；节点、机柜、链路等共同失效范围 | 不把单个 worker 错误等同于整个集群故障 |
| 检查点提交 | checkpoint commit；写入、校验、版本确认和可恢复可见性 | 不把异步写文件等同于原子分布式提交 |
| 幂等重试 | idempotent retry；由 job/attempt/step/version 防止重复更新 | 不只靠进程重启恢复训练进度 |
| 集群遥测 | cluster telemetry；跨作业、rank、设备和链路的 metrics/tracing | 不把本地 kernel profiler 叫集群遥测 |
| trace schema | machine-readable trace；`schema_version`、event、job/attempt、time、placement、replay | 不把日志文本格式当成稳定 API |
| 队列等待 | queue wait；从提交/重新排队到成组准入的时间 | 不把 queue wait 混入 device compute |
| 作业调度与算子调度 | job scheduler 管资源租约；operator scheduler 管进程/设备内任务 | 不把 Fusion/stream scheduler 当作 GPU 集群调度器 |

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
