# 7.4 压缩、精度与离线优化

## 压缩不是把文件压小

模型压缩可能改变四个不同对象：

1. 参数在磁盘上的表示；
2. 参数加载后的 dtype 或布局；
3. forward 期间的 activation 和算子；
4. 模型拓扑与参数数量。

只改变第一项通常减少下载和存储成本；只有同时改变 runtime 支持的计算
路径，才可能减少在线内存或延迟。比如把 float32 权重压成 int8 文件，
如果运行时每次都把它完整反量化成 float32，再调用原来的算子，收益可能
主要来自存储，而不是计算。

## 量化的基本模型

以线性量化为例，把浮点值 $r$ 映射为整数 $q$：

$$
q = \operatorname{clip}\left(
\operatorname{round}\left(\frac{r}{s}+z\right), q_{\min},q_{\max}
\right),
$$

其中 $s$ 是 scale，$z$ 是 zero point。对称量化通常取 $z=0$，非对称
量化可以更充分使用整数范围；逐层、逐通道和逐 block 决定 scale 的共享
粒度。粒度越细通常能减小误差，但会增加 metadata、加载逻辑和计算复杂度。

激活值量化需要代表性数据做校准，因为部署前并不知道每一层真实输入分布。
常见路径有：

- **PTQ（post-training quantization）**：训练完成后量化，工程成本较低，
  但精度依赖校准和模型结构；
- **QAT（quantization-aware training）**：训练中插入伪量化，让模型适应
  误差，成本更高；
- **权重量化**：只改变 weight，激活仍走浮点；
- **全量化**：weight、activation 和算子共同走低精度路径。

固定主线 Burn 的 crate 文档明确写出：当前不支持 QAT，部分 backend 在
开发中的 PTQ 路径支持有限的低精度表示。这个事实不能外推为“任意
`burn-onnx` 模型都能自动完成量化”。`burn-onnx` 中存在量化相关 ONNX
node 代码，也只说明 importer 有对应节点实现入口；仍需针对具体模型、
backend、dtype 和 reference 做验证。

## 稀疏、剪枝与蒸馏

剪枝将不重要的 weight、channel 或 block 置零或删除。非结构化稀疏的精度
损失可能较小，但不规则索引会带来分支、访存和负载不均；结构化稀疏更容易
映射到规则 kernel，却可能损失更多表达能力。因而“零值更多”不等于
“服务更快”，要看目标 backend 是否真正使用稀疏 kernel。

知识蒸馏则不一定改变现有模型的 runtime format。它训练一个较小的学生
模型，让学生同时接近真实标签和教师输出，例如：

$$
\mathcal{L} =
\mathcal{L}_{\mathrm{label}} +
\lambda\mathcal{L}_{\mathrm{teacher}}.
$$

蒸馏的成本主要发生在训练阶段，部署得到的是一个新的拓扑和参数 artifact。
不能把蒸馏当作加载时的 `ModuleRecord` 选项。

## 图级优化与算子级优化

固定 `burn-onnx` 的 simplify/codegen 路径适合做一部分离线图简化：
常量折叠、死节点消除、公共子表达式和某些 reshape/permute pattern。
部署阶段还可以做：

- Conv 与 BatchNorm 的权重折叠；
- layout/transpose 重排；
- 将裁切、reshape 或 binary elementwise 前移；
- 为目标 backend 选择 kernel、tile 和 workspace；
- 复用中间 buffer，减少峰值内存。

这些优化都有前提。BatchNorm 训练期的统计和参数不是部署前的常量；
只有在推理状态冻结后，某些折叠才保持语义。layout 重排需要同步输入、
权重和后续算子；workspace 复用需要证明活跃区间不重叠。

第 4 章的 Fusion/CubeCL 讨论运行时和设备侧表示。这里的“离线优化”
不应被写成“Burn 每次都会执行某个融合”：真正的行为取决于生成图、
Burn feature、backend、shape 和设备。

## 精度—延迟—内存的验证闭环

每个优化至少要记录三组证据：

```text
reference model + fixed inputs
        │
        ├── output error: max / mean / task metric
        ├── memory: artifact + peak runtime allocation
        └── latency: warmup + p50/p95 + batch/shape/device
```

对量化、剪枝和 layout 变换，测试不能只比较模型文件大小。至少要固定
输入 schema、batch、dtype、backend 和测量方式；如果使用校准集，还要
保存校准集版本或摘要。一个在 CPU Flex 上变快的结果不能自动外推到
WebGPU、CUDA 或嵌入式目标。

本项目固定快照没有为第 7 章加入量化 benchmark，因为当前要先验证主线
Record artifact 的恢复语义。压缩方案可以作为练习或后续章节扩展，不在
没有目标 backend 和 reference 数据时写成已完成的部署能力。
