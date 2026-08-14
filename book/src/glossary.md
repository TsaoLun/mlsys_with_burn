# 术语表

本表汇总全书反复使用的关键术语，按主题分组。正文在首次出现时给出英文
原名；本表用于回查。“章节”列指出该术语的主要讨论位置。

## 基础：张量、设备与自动微分

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 张量 | `Tensor<D, K>` | Burn 的张量类型；`D` 为秩，`K` 为类别 | [第 2 章](ch02-programming-and-graph.md) |
| 设备 | Device | 用户侧设备选择；内部经 `DispatchDevice` 分派到具体后端 | [第 2 章](ch02-programming-and-graph.md) |
| 分派桥 | BridgeTensor / Dispatch | Tensor 与具体后端之间的运行时桥接层；不是第三种数学语义 | [第 2 章](ch02-programming-and-graph.md) |
| 后端契约 | Backend / BackendTypes | 后端实现层契约，不等于用户手里的 Device | [第 2 章](ch02-programming-and-graph.md) |
| 默认 CPU 后端 | Flex / `Device::flex()` | 纯 Rust eager CPU；默认实验路径，**不**走 Fusion/CubeCL | [第 1 章](ch01-introduction.md) |
| Fusion CPU 设备 | `Device::cpu()` | 启用 `cpu`+`fusion` 后走 CubeCL CPU Fusion；与 Flex 不同路径 | [第 4 章](ch04-compiler-and-runtime.md) |
| 自动微分记录 | autodiff tape | 一阶反向模式动态 tape，位于 `burn-autodiff` | [第 2 章](ch02-programming-and-graph.md) |
| 切断依赖 | `detach()` | 切断旧图形成新叶子，保留 require-grad 意图 | [第 2 章](ch02-programming-and-graph.md) |
| 状态保存 | ModuleRecord | Burn 的参数 artifact 表示 | [第 7 章](ch07-model-serving.md) |

## 加速器与编译

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 并行拓扑 | Cube / Unit / Plane | CubeCL 的并行层次术语 | [第 3 章](ch03-accelerator.md) |
| 主机参考实现 | host reference | 用普通 host 代码写出的可观察正确结果，用来对照 Kernel/Runtime 输出；**不是** Host/Device 机器模型本身 | [第 3 章](ch03-accelerator.md) |
| 计算客户端 | ComputeClient | host 向某 Runtime 创建 buffer、launch Kernel、读回结果的入口 | [第 3 章](ch03-accelerator.md) |
| 回退路径 | fallback | 高性能候选不可用或不合适时改走仍正确的较简实现；正确性相同 ≠ 成本相同 | [第 3 章](ch03-accelerator.md) |
| 自动调优 | autotune / tune key | 在当前设备上测量候选并按键缓存；CPU 命中不能直接搬到另一 Runtime | [第 3 章](ch03-accelerator.md) |
| 特化键 / 编译键 | specialization / compile key | 参与 Kernel 变体区分的 comptime 或编译配置输入；键变了可能触发重编译 | [第 3 章](ch03-accelerator.md) |
| 算子库路径 | Blueprint–Routine | CubeK 的算子组织方式；Strategy/Launch 负责选择与启动；LocalTuner 注册候选 | [第 3 章](ch03-accelerator.md) |
| 算术强度 | arithmetic intensity | FLOP/字节（教学模型可用 FLOP/加载元素）；是复用方向指标，不是实测性能 | [第 3 章](ch03-accelerator.md) |
| 融合表示 | Burn IR / OperationIr | Fusion 计划的中间表示；与 autodiff tape 不同层 | [第 4 章](ch04-compiler-and-runtime.md) |
| 融合器 | fuser | Fusion 里接受/拒绝一组操作并生成融合块的组件（如 ElementWise） | [第 4 章](ch04-compiler-and-runtime.md) |
| Fusion 流 | Fusion stream / `StreamId` | Fusion 延迟队列的隔离键；不是 CUDA stream，也不是集群作业队列 | [第 4 章](ch04-compiler-and-runtime.md) |
| CubeCL 表示 | Scope / KernelDefinition | CubeCL 侧的 IR 对象，不称“计算图” | [第 4 章](ch04-compiler-and-runtime.md) |
| 设备重放 | graph capture | backend/device 级执行重放，仅在 Runtime 支持时存在 | [第 4 章](ch04-compiler-and-runtime.md) |
| 同步边界 | read / readback / `Device::sync` | 完成边界；flush 只是提交/推进，不代表设备完成 | [第 4 章](ch04-compiler-and-runtime.md) |

## 数据与训练

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| 负载卡片 | workload card | 计算、数据、设备、目标四元组描述的系统负载 | [第 1 章](ch01-introduction.md) |
| 数据供给模型 | $F/P/G$ | 读取、变换、设备消费三类速率 | [第 5 章](ch05-data-processing.md) |
| 分片与提交 | shard / epoch commit | 确定性分片 offset；epoch 边界的提交点 | [第 5 章](ch05-data-processing.md) |
| 重排缓冲 | reorder buffer | 多 worker 乱序到达后按全局序号恢复可读顺序的缓冲 | [第 5 章](ch05-data-processing.md) |
| 加权集合通信 | weighted AllReduce | 按样本数聚合局部梯度，而非等权平均 | [第 6 章](ch06-training-systems.md) |
| 梯度新鲜度 | gradient staleness | 当前参数版本与梯度版本的差距 | [第 6 章](ch06-training-systems.md) |
| 流水线空泡 | pipeline bubble | 1F1B 的 warm-up/drain 空闲槽 | [第 6 章](ch06-training-systems.md) |
| 环形 AllReduce | ring AllReduce | 每设备约 \(2S\) 字节、α 步数 \(2(p-1)\) 的集合通信 | [第 6 章](ch06-training-systems.md) |
| 参数分片 | ZeRO / FSDP | 按级切开优化器状态、梯度、参数以换显存 | [第 6 章](ch06-training-systems.md) |
| 训练执行策略 | `ExecutionStrategy`（burn-train） | MultiDevice/DDP 等训练装配策略；**不是**第 4 章 Fusion 块内的同名搜索对象 | [第 6 章](ch06-training-systems.md) |
| 训练进程号 | rank | 分布式作业里的进程/副本序号；与张量的秩（rank）不是同一概念 | [第 9 章](ch09-gpu-cluster.md) |

## 部署与强化学习

| 术语 | 英文 | 含义 | 章节 |
|---|---|---|---|
| artifact 清单 | artifact manifest | version/payload length/checksum 等元数据 | [第 7 章](ch07-model-serving.md) |
| 动态批处理 | dynamic batching | 同 shape 请求按容量成批 | [第 7 章](ch07-model-serving.md) |
| 训练后量化 | post-training quantization, PTQ | 训练后用校准集确定量化参数，不改训练过程 | [第 7 章](ch07-model-serving.md) |
| 校准 | calibration | 用代表性数据估计张量数值范围，决定 scale/zero-point | [第 7 章](ch07-model-serving.md) |
| 量化参数 | scale / zero-point | 浮点区间到整数网格的仿射映射参数 | [第 7 章](ch07-model-serving.md) |
| 连续批处理 | continuous batching | 逐 token 调度：请求完成即退出、新请求随时并入 batch | [第 7 章](ch07-model-serving.md) |
| KV 缓存 | KV cache | 生成式推理保存注意力键值以免重算；容量构成 KV 预算 | [第 7 章](ch07-model-serving.md) |
| 首 token 延迟 | TTFT | time to first token：到达到第一个 decode token | [第 7 章](ch07-model-serving.md) |
| 每 token 间隔 | TPOT | time per output token：首 token 之后的平均出字间隔 | [第 7 章](ch07-model-serving.md) |
| 分块预填充 | chunked prefill | 把长 prompt 切成多步，以免独占连续批的一整步 | [第 7 章](ch07-model-serving.md) |
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
