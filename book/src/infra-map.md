# 从应用到集群

机器学习系统不是「框架 + 几张 GPU」。一次完整任务通常同时穿过下面几层；
本书九章分别打开其中一层。表中的产业名字只用来定位问题，不表示 Burn
实现了同名产品。

## 系统分层

| 你在产业里碰到的名字 | 它在解决什么 | 本书位置 |
|---|---|---|
| PyTorch / JAX / TensorFlow 的 Tensor、Module | 用什么接口写模型、参数和梯度从哪来 | [第 2 章](ch02-programming-and-graph.md) |
| CUDA / Triton / CUTLASS、GPU 存储层次 | 工作如何映射到计算阵列与内存 | [第 3 章](ch03-accelerator.md) |
| XLA / TVM / nvFuser、Kernel 选择 | 如何在不改语义的前提下变换计算 | [第 4 章](ch04-compiler-and-runtime.md) |
| DataLoader、`tf.data`、DALI、对象存储 | 如何让设备一直有数据可算 | [第 5 章](ch05-data-processing.md) |
| DDP、FSDP / ZeRO、Megatron TP/PP、GPipe | 参数、激活与梯度如何切分和同步 | [第 6 章](ch06-training-systems.md) |
| ONNX Runtime、Triton Server、vLLM | 权重如何落地、请求如何排队 | [第 7 章](ch07-model-serving.md) |
| 环境交互、replay、Actor–Learner | 数据由策略和环境共同产生时怎么训练 | [第 8 章](ch08-rl-systems.md)（可选） |
| Slurm、K8s 设备插件、作业队列、NCCL 拓扑 | 谁获得 GPU、通信走哪条链路、故障怎么恢复 | [第 9 章](ch09-gpu-cluster.md) |

大模型训练和推理并没有另起一套物理学：KV cache 是第 3 章存储层次和第
7 章队列的应用；张量并行是第 6 章通信成本的另一种切分；MoE 专家放置是
第 9 章拓扑感知的近亲。第 1 章会把这些杠杆标在地图上，各章再展开机制。

## 三条阅读顺序

**先把模型跑通：** 第 1、2、5、6、7 章，最后做[综合实验](capstone.md)。
需要自定义 Kernel 或看融合时，再补第 3、4 章。

**先看清基础设施：** 第 1 章 → 第 3、4 章（设备与编译）→ 第 5、6 章
（数据与训练）→ 第 7、9 章（服务与集群）。第 8 章可以跳过。打开源码时
配合[一次调用会经过哪些层](crate-map.md)。

**先写 Kernel：** 第 1、2 章之后进入第 3、4 章，并对照 CubeCL / CubeK
源码；训练和服务章节按需回看。
