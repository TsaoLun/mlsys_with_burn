# 术语表

本表汇总全书反复使用的关键术语，按主题分组。正文在首次出现时给出英文
原名；本表用于回查。“章节”列指出该术语的主要讨论位置。

## 基础：张量、设备与自动微分

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 张量 | `Tensor<D, K>` | Burn 的张量类型；`D` 为秩，`K` 为类别 | [第 2 章](ch02-programming-and-graph.md) |
| 设备 | Device | 用户侧设备选择；运行时分派到 DispatchDevice | [第 2 章](ch02-programming-and-graph.md) |
| 后端契约 | Backend / BackendTypes | 后端实现层契约，不等于用户手里的 Device | [第 2 章](ch02-programming-and-graph.md) |
| 默认 CPU 后端 | Flex | 纯 Rust eager CPU 后端；默认实验路径不走 Fusion/CubeCL | [第 1 章](ch01-introduction.md) |
| 自动微分记录 | autodiff tape | 一阶反向模式动态 tape，位于 `burn-autodiff` | [第 2 章](ch02-programming-and-graph.md) |
| 切断依赖 | `detach()` | 切断旧图形成新叶子，保留 require-grad 意图 | [第 2 章](ch02-programming-and-graph.md) |
| 状态保存 | ModuleRecord | Burn 的参数 artifact 表示 | [第 7 章](ch07-model-serving.md) |

## 加速器与编译

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 并行拓扑 | Cube / Unit / Plane | CubeCL 的并行层次术语 | [第 3 章](ch03-accelerator.md) |
| 算子库路径 | Blueprint–Routine | CubeK 的算子组织方式；Strategy/Launch 负责选择与启动 | [第 3 章](ch03-accelerator.md) |
| 算术强度 | arithmetic intensity | FLOP/字节（教学模型可用 FLOP/加载元素）；是复用方向指标，不是实测性能 | [第 3 章](ch03-accelerator.md) |
| 融合表示 | Burn IR / OperationIr | Fusion 计划的中间表示；与 autodiff tape 不同层 | [第 4 章](ch04-compiler-and-runtime.md) |
| CubeCL 表示 | Scope / KernelDefinition | CubeCL 侧的 IR 对象，不称“计算图” | [第 4 章](ch04-compiler-and-runtime.md) |
| 设备重放 | graph capture | backend/device 级执行重放，仅在 Runtime 支持时存在 | [第 4 章](ch04-compiler-and-runtime.md) |
| 同步边界 | read / `Device::sync` | 完成边界；flush 只是提交/推进，不代表设备完成 | [第 4 章](ch04-compiler-and-runtime.md) |

## 数据与训练

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 负载卡片 | workload card | 计算、数据、设备、目标四元组描述的系统负载 | [第 1 章](ch01-introduction.md) |
| 数据供给模型 | $F/P/G$ | 读取、变换、设备消费三类速率 | [第 5 章](ch05-data-processing.md) |
| 分片与提交 | shard / epoch commit | 确定性分片 offset；epoch 边界的提交点 | [第 5 章](ch05-data-processing.md) |
| 加权集合通信 | weighted AllReduce | 按样本数聚合局部梯度，而非等权平均 | [第 6 章](ch06-training-systems.md) |
| 梯度新鲜度 | gradient staleness | 当前参数版本与梯度版本的差距 | [第 6 章](ch06-training-systems.md) |
| 流水线空泡 | pipeline bubble | 1F1B 的 warm-up/drain 空闲槽 | [第 6 章](ch06-training-systems.md) |

## 部署与强化学习

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| artifact 清单 | artifact manifest | version/payload length/checksum 等元数据 | [第 7 章](ch07-model-serving.md) |
| 动态批处理 | dynamic batching | 同 shape 请求按容量成批 | [第 7 章](ch07-model-serving.md) |
| 终止语义 | `done` / `truncated` | 自然终止与外部/时间截断，bootstrap 语义不同 | [第 8 章](ch08-rl-systems.md) |
| 策略关系 | behavior / target policy | $\mu$ 采样、$\pi$ 学习；on/off-policy 的区分依据 | [第 8 章](ch08-rl-systems.md) |
| 策略新鲜度 | policy freshness | behavior/target 的版本差 | [第 8 章](ch08-rl-systems.md) |
| 采样更新架构 | Actor–Learner | actor 采样、learner 更新的分工架构 | [第 8 章](ch08-rl-systems.md) |

## 集群

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 控制面 / 数据面 | control plane / data plane | 作业与资源管理 vs. rank 间训练通信 | [第 9 章](ch09-gpu-cluster.md) |
| 成组调度 | gang scheduling | 同步作业要么获得完整资源，要么等待 | [第 9 章](ch09-gpu-cluster.md) |
| 拓扑感知放置 | topology-aware placement | 按节点、机柜、链路域选择资源 | [第 9 章](ch09-gpu-cluster.md) |
| 超额认购 | oversubscription | 峰值流量超过共享链路容量 | [第 9 章](ch09-gpu-cluster.md) |
| 资源碎片 | resource fragmentation | 总量足够但无法满足成组/显存/拓扑约束 | [第 9 章](ch09-gpu-cluster.md) |
| 故障域 | failure domain | 节点、机柜、链路等共同失效范围 | [第 9 章](ch09-gpu-cluster.md) |
| 检查点提交 | checkpoint commit | 写入、校验、版本确认和可恢复可见性 | [第 9 章](ch09-gpu-cluster.md) |
| 幂等重试 | idempotent retry | 由 job/attempt/step/version 防止重复更新 | [第 9 章](ch09-gpu-cluster.md) |
| 集群遥测 | cluster telemetry | 跨作业、rank、设备和链路的 metrics/tracing | [第 9 章](ch09-gpu-cluster.md) |
| 机器可读轨迹 | machine-readable trace | 带 `schema_version` 的结构化事件流 | [第 9 章](ch09-gpu-cluster.md) |
| 队列等待 | queue wait | 从提交/重新排队到成组准入的时间 | [第 9 章](ch09-gpu-cluster.md) |

## 本书的阅读证据标签

`CPU 可运行验证`、`源码核验`、`协议/成本模型`、`可选平台实验`、
`未覆盖` 的定义见[逐文件对照矩阵导读](crosswalk-guide.md)。
