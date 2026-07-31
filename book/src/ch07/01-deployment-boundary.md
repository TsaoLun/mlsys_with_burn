# 7.1 部署边界、artifact 与服务成本

## 先把“部署模型”拆开

“模型”在不同阶段不是同一个东西。训练期间的模型通常包括可更新的
参数、optimizer state、scheduler、随机数和数据迭代位置；部署时至少需要
固定的拓扑、推理参数和权重。一个服务还需要输入 schema、前处理、后处理、
版本标识和运行时配置。

可以把一次部署写成四个对象：

```text
Model artifact
  = topology + parameters + dtype/shape metadata + version

Runtime
  = artifact loader + backend + device + memory policy

Request contract
  = input schema + preprocessing + batching + timeout

Service policy
  = routing + authorization + metrics + rollout + recovery
```

这四个对象可能由不同项目负责。`burn-onnx` 主要处理 topology 到 Rust
source 的转换，并生成权重加载入口；Burn `ModuleRecord` 主要处理 module
参数状态；Remote 主要处理 tensor operation 的远端执行。它们都不会自动
变成一个带鉴权、限流和灰度发布的 HTTP 服务。

## 训练与推理的状态差异

训练需要保留梯度、autodiff tape 和 optimizer state；推理通常希望
`valid()` 模型不再建立训练 tape，并尽早释放临时张量。训练的目标是
time-to-accuracy，推理服务通常同时优化以下目标：

- **正确性**：相同版本、相同 dtype 和相同前/后处理时，输出满足容差；
- **延迟**：单请求或批请求从进入队列到返回的时间，常看 p50、p95、p99；
- **吞吐**：单位时间完成的样本或 token 数；
- **资源**：权重常驻内存、临时 workspace、设备占用和功耗；
- **可恢复性**：进程重启、模型回滚、设备错误和请求超时后的行为。

如果只比较一次 `forward` 的墙钟时间，测量可能漏掉模型加载、输入复制、
队列等待、设备 flush、结果 readback 和后处理。服务观测至少应把这些边界
分开记录。

## 一个最小延迟模型

对一个请求，可以用下面的分解开始分析：

$$
T_{\mathrm{request}} =
T_{\mathrm{queue}} + T_{\mathrm{pre}} +
T_{\mathrm{copy}} + T_{\mathrm{forward}} +
T_{\mathrm{readback}} + T_{\mathrm{post}}.
$$

批处理会改变其中多个项。若一次批包含 $b$ 个样本，固定调度和 kernel
启动成本可能被摊薄，吞吐上升；但队列需要等待更多请求，单请求延迟可能
上升。设备的计算时间也未必线性缩放，原因包括矩阵形状、缓存、带宽和
backend 的 kernel 选择。

Remote 场景还要增加网络传输和远端排队：

$$
T_{\mathrm{remote}} =
T_{\mathrm{client\ queue}} + T_{\mathrm{upload}} +
T_{\mathrm{remote\ queue}} + T_{\mathrm{compute}} +
T_{\mathrm{download}}.
$$

把模型权重在 peer 上常驻，可以避免每个请求重复传输权重；但输入、
中间张量或输出仍可能跨网络移动。只有在确认 operation batching、数据
大小和链路条件后，才能讨论 Remote 是否比本地执行更快。

## artifact 的兼容性不是文件能打开就够了

一个权重文件能被解析，只说明二进制格式和部分元数据可读。真正可用还要
检查：

1. 参数路径能否匹配目标 module；
2. shape、layout 和 dtype 是否一致；
3. 参数是否需要转置、重命名或其他 adapter；
4. 模型拓扑是否与权重版本匹配；
5. reference 输入下输出误差是否在约定容差内；
6. 前处理、后处理和类别/token 映射是否同步。

这也是为什么本章把 `ModuleRecord` round-trip 写成测试，而不是只断言
“文件生成成功”。在生产环境中，还应把 model revision、代码 revision、
backend、dtype、校准集摘要和 schema version 写入发布元数据。

## Burn 的位置

固定 Burn 主线的核心 `ModuleRecord` 记录 module 参数和 `ParamId`，可以用
内存 Burnpack bytes 恢复到一个新 module。它是一个很小、可测试的 artifact
边界，不等于完整部署 manifest。更丰富的 `burn-store` 再提供
`ModuleSnapshot`、SafeTensors/PyTorch adapter、过滤和 remap，但这些能力
要按 feature 启用并单独验证。

下一节转向更大的转换边界：ONNX 图如何变成 Burn 的 Rust source，以及
为什么固定 `burn-onnx` 的依赖 revision 必须先对齐。
