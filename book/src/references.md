# 参考文献

本页收录各章原理与产业背景的论文、教材和官方文档，按章分组，并用一句
话说明它与正文的关系。两点使用提示：

- 这些文献解释**框架无关的原理与历史**；其中的 API 细节或性能数字属于
  各自的系统与年代，不能直接当作本书所用 Burn 版本的事实。
- 链接以 arXiv、DOI 与官方站点为主。个别未给链接的条目请按题名检索。

## 教材与总览

- OpenMLSys 团队，《机器学习系统：设计和实现》。
  [openmlsys.github.io](https://openmlsys.github.io/)
  ——本书的底本。推荐系统、联邦学习、可解释 AI 与机器人系统等本书未
  覆盖的主题，可直接阅读原作对应章节。
- Burn 官方文档与仓库：[burn.dev](https://burn.dev/)、
  [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
  ——在线文档跟随最新版本；与本书版本出现差异时，以书内示例为准。
- CubeCL 仓库：
  [github.com/tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)
  ——第 3、4 章 Kernel 与运行时叙事的上游项目。
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*,
  NeurIPS 2015——「训练代码只是系统的一小部分」这一判断的经典论证，
  可作第 1 章系统分层的动机阅读。

## 第 1 章 导论

- Hennessy & Patterson, *A New Golden Age for Computer Architecture*,
  Communications of the ACM, 2019——领域专用架构为何兴起；对应第 1 章
  「负载决定系统设计」的产业背景。
- Jouppi et al., *In-Datacenter Performance Analysis of a Tensor
  Processing Unit*, ISCA 2017。
  [arXiv:1704.04760](https://arxiv.org/abs/1704.04760)
  ——用真实数据中心负载论证专用加速器的设计取舍，与第 1 章 workload
  card 的思路互相印证。

## 第 2 章 编程接口与计算图

- Baydin, Pearlmutter, Radul & Siskind, *Automatic Differentiation in
  Machine Learning: a Survey*, JMLR 2018。
  [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
  ——自动微分四种实现路线的系统综述；第 2 章反向模式与 tape 的理论
  背景。
- Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep
  Learning Library*, NeurIPS 2019。
  [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
  ——eager 执行接口的代表设计，可与 Burn 的 eager + 运行时融合路线
  对照。
- Abadi et al., *TensorFlow: A System for Large-Scale Machine
  Learning*, OSDI 2016。
  [arXiv:1605.08695](https://arxiv.org/abs/1605.08695)
  ——静态数据流图的代表设计；第 2 章「先建图后执行」路线的原始论述。
- Griewank & Walther, *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation* (2nd ed.), SIAM 2008
  ——自动微分的标准教材，适合想深究 tape 与检查点策略的读者。

## 第 3 章 AI 加速器与编程

- Williams, Waterman & Patterson, *Roofline: An Insightful Visual
  Performance Model for Multicore Architectures*, Communications of
  the ACM, 2009。
  [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
  ——第 3 章算术强度与屋顶模型的原始出处。
- NVIDIA, *CUDA C++ Programming Guide*。
  [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  ——线程层次、共享内存与同步原语的权威定义；与 CubeCL 的
  Cube/Plane/Unit 拓扑对照阅读。
- Tillet, Kung & Cox, *Triton: An Intermediate Language and Compiler
  for Tiled Neural Network Computations*, MAPL 2019——与 CubeCL 同类
  的 tile 级 Kernel DSL，可比较两者对布局与调度的抽象方式。
- Jouppi et al., *In-Datacenter Performance Analysis of a Tensor
  Processing Unit*, ISCA 2017。
  [arXiv:1704.04760](https://arxiv.org/abs/1704.04760)
  ——脉动阵列与矩阵单元的设计动机，对应第 3 章矩阵指令一节。

## 第 4 章 AI 编译器与运行时系统

- Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for
  Deep Learning*, OSDI 2018。
  [arXiv:1802.04799](https://arxiv.org/abs/1802.04799)
  ——算子编译与自动调优的代表系统；第 4 章编译因果链的横向对照。
- Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain
  Specific Computation*, CGO 2021。
  [arXiv:2002.11054](https://arxiv.org/abs/2002.11054)
  ——多层 IR 基础设施的设计论述，可对照 Burn IR 与 CubeCL IR 的分层。
- Ragan-Kelley et al., *Halide: A Language and Compiler for Optimizing
  Parallelism, Locality, and Recomputation in Image Processing
  Pipelines*, PLDI 2013——「算法与调度分离」思想的出处，是理解现代
  Kernel 编译器的共同源头。
- OpenXLA 项目文档：[openxla.org/xla](https://openxla.org/xla)
  ——图级融合与后端 lowering 的另一工业实现，可对照第 4 章 Fusion。

## 第 5 章 数据处理系统

- Murray et al., *tf.data: A Machine Learning Data Processing
  Framework*, VLDB 2021。
  [arXiv:2101.12127](https://arxiv.org/abs/2101.12127)
  ——数据管道的组合子设计与自动调优；第 5 章惰性变换与背压模型的
  横向对照。
- Mohan et al., *Analyzing and Mitigating Data Stalls in DNN
  Training*, VLDB 2021。
  [arXiv:2007.06775](https://arxiv.org/abs/2007.06775)
  ——用测量方法定位「数据等待」瓶颈，与第 5 章生产/消费预算模型互补。
- NVIDIA DALI 文档：
  [docs.nvidia.com/deeplearning/dali](https://docs.nvidia.com/deeplearning/dali/)
  ——把解码与增广搬上 GPU 的工程路线，对应第 5 章扩展路径一节。

## 第 6 章 训练系统

- Li et al., *Scaling Distributed Machine Learning with the Parameter
  Server*, OSDI 2014。
  [usenix.org](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu)
  ——参数服务器协议的原始论文；第 6 章版本/陈旧梯度讨论的出处。
- Sergeev & Del Balso, *Horovod: Fast and Easy Distributed Deep
  Learning in TensorFlow*, 2018。
  [arXiv:1802.05799](https://arxiv.org/abs/1802.05799)
  ——把环形 AllReduce 引入主流训练框架的工程报告。
- Patarasuk & Yuan, *Bandwidth Optimal All-reduce Algorithms for
  Clusters of Workstations*, JPDC 2009——环形 AllReduce 每设备流量
  近似 $2S$ 的理论出处，对应第 6 章的推导。
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in
  1 Hour*, 2017。[arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
  ——大 batch 与学习率线性缩放的经典实验，训练系统与算法互动的样本。
- Huang et al., *GPipe: Efficient Training of Giant Neural Networks
  using Pipeline Parallelism*, NeurIPS 2019。
  [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
  ——micro-batch 流水线并行的出处；第 6 章空泡公式的背景。
- Narayanan et al., *PipeDream: Generalized Pipeline Parallelism for
  DNN Training*, SOSP 2019。
  [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)
  ——1F1B 调度的出处，与 GPipe 的填充-排空策略对照。
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models*, SC 2020。
  [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
  ——把优化器状态/梯度/参数分片的内存优化谱系。
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter
  Language Models Using Model Parallelism*, 2019。
  [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
  ——张量并行的工程范式，对应第 6 章并行策略分类。
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016。
  [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
  ——激活重计算（recomputation）的出处，对应第 6 章内存动机。
- Micikevicius et al., *Mixed Precision Training*, ICLR 2018。
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
  ——混合精度与 loss scaling 的标准做法。

## 第 7 章 模型服务

- ONNX 规范与算子集：[onnx.ai](https://onnx.ai/)
  ——第 7 章模型交换格式的权威定义。
- Jacob et al., *Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference*, CVPR 2018。
  [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
  ——scale/zero-point 量化方案的出处，对应第 7 章 PTQ 演算。
- Han, Mao & Dally, *Deep Compression: Compressing Deep Neural
  Networks with Pruning, Trained Quantization and Huffman Coding*,
  ICLR 2016。[arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
  ——剪枝、量化与编码组合压缩的经典工作。
- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural
  Network*, 2015。[arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
  ——知识蒸馏的出处。
- Olston et al., *TensorFlow-Serving: Flexible, High-Performance ML
  Serving*, 2017。
  [arXiv:1712.06139](https://arxiv.org/abs/1712.06139)
  ——模型版本、加载与批处理的服务系统样本。
- Crankshaw et al., *Clipper: A Low-Latency Online Prediction Serving
  System*, NSDI 2017。
  [usenix.org](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw)
  ——推理服务的延迟/吞吐权衡与自适应批处理。
- Dean & Barroso, *The Tail at Scale*, Communications of the ACM,
  2013——尾延迟为什么主导服务体验；第 7 章延迟指标的必读背景。
- Yu et al., *Orca: A Distributed Serving System for
  Transformer-Based Generative Models*, OSDI 2022。
  [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu)
  ——continuous batching 的出处；第 7 章的队列模拟实验
  `ch07-serving-queue-sim` 演示的正是这一机制，工程实现可从这篇
  读起。
- Kwon et al., *Efficient Memory Management for Large Language Model
  Serving with PagedAttention*, SOSP 2023。
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  ——KV cache 分页管理（vLLM）；同为大模型服务的延伸出口。

## 第 8 章 强化学习系统

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd
  ed.), MIT Press 2018。
  [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html)
  ——MDP、TD 与策略学习的标准教材；第 8 章符号体系与其一致。
- Mnih et al., *Human-Level Control through Deep Reinforcement
  Learning*, Nature 2015。
  [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
  ——DQN 与经验回放的出处，对应第 8 章 replay 一节。
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017。
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
  ——on-policy 训练的代表算法，可对照第 8 章 on/off-policy 边界。
- Horgan et al., *Distributed Prioritized Experience Replay*, ICLR
  2018。[arXiv:1803.00933](https://arxiv.org/abs/1803.00933)
  ——Actor 与 Learner 分离加共享 replay 的系统形态（Ape-X）。
- Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures*, ICML 2018。
  [arXiv:1802.01561](https://arxiv.org/abs/1802.01561)
  ——大规模 actor–learner 的策略滞后（policy lag）修正方案。
- Moritz et al., *Ray: A Distributed Framework for Emerging AI
  Applications*, OSDI 2018。
  [arXiv:1712.05889](https://arxiv.org/abs/1712.05889)
  ——支撑分布式 RL 的通用任务/Actor 运行时。
- Liang et al., *RLlib: Abstractions for Distributed Reinforcement
  Learning*, ICML 2018。
  [arXiv:1712.09381](https://arxiv.org/abs/1712.09381)
  ——RL 系统的组合抽象设计，可与 burn-rl 的组合点对照。

## 第 9 章 大规模 GPU 集群管理

- Verma et al., *Large-Scale Cluster Management at Google with Borg*,
  EuroSys 2015。
  [doi:10.1145/2741948.2741964](https://doi.org/10.1145/2741948.2741964)
  ——集群调度器的奠基论文；队列、配额与优先级机制的原型。
- Xiao et al., *Gandiva: Introspective Cluster Scheduling for Deep
  Learning*, OSDI 2018。
  [usenix.org](https://www.usenix.org/conference/osdi18/presentation/xiao)
  ——针对深度学习负载的时分复用与迁移调度。
- Gu et al., *Tiresias: A GPU Cluster Manager for Distributed Deep
  Learning*, NSDI 2019。
  [usenix.org](https://www.usenix.org/conference/nsdi19/presentation/gu)
  ——无先验作业时长下的 GPU 调度策略，对应第 9 章队列与放置。
- Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU Clusters for
  DNN Training Workloads*, USENIX ATC 2019。
  [usenix.org](https://www.usenix.org/conference/atc19/presentation/jeon)
  ——真实多租户集群的负载分析（Philly trace），本章故障与碎片讨论的
  实证背景。
- Weng et al., *MLaaS in the Wild: Workload Analysis and Scheduling
  in Large-Scale Heterogeneous GPU Clusters*, NSDI 2022。
  [usenix.org](https://www.usenix.org/conference/nsdi22/presentation/weng)
  ——更大规模异构集群的负载画像。
- Young, *A First Order Approximation to the Optimum Checkpoint
  Interval*, Communications of the ACM, 1974；Daly, *A Higher Order
  Estimate of the Optimum Checkpoint Interval for Restart Dumps*,
  FGCS 2006——第 9 章 checkpoint 间隔公式的一阶与高阶出处。
- NVIDIA NCCL 文档：
  [developer.nvidia.com/nccl](https://developer.nvidia.com/nccl)
  ——GPU 集合通信库的官方入口，对应第 6、9 章的通信数据面。
