# 第 7 章 模型服务

训练结束不是系统结束。一个模型从训练进程走到用户请求之间，至少要经过
状态导出、格式或图转换、数值验证、运行时选择、输入输出契约和服务治理。
模型服务（model serving）关心的不只是“能不能算出一个 tensor”，还要回答
模型文件由谁解释、权重何时进入设备、请求怎样排队，以及延迟、吞吐、内存和
安全边界由哪一层负责。

## 本章问题

训练产物如何转换、验证、优化并部署到服务器、浏览器或受限设备？固定 Burn
快照中的 `ModuleRecord`、`burn-onnx`、Remote 和 WASM/no_std 分别解决哪一
段问题，又没有解决什么？

## 学习目标

完成本章后，你应该能够：

1. 把模型部署拆成模型产物（artifact）、执行运行时（runtime）、请求服务和安全治理四个边界；
2. 解释 ONNX 图到 Burn Rust source 的转换为什么不是简单的文件改名；
3. 区分模型拓扑、参数状态、权重格式、运行时 backend 和服务协议；
4. 使用 `ModuleRecord`/Burnpack 在 CPU 上保存并恢复参数，验证推理输出；
5. 区分 `burn-onnx` 的 `File`、`Embedded`、`Bytes` 和 `None` 加载策略；
6. 用 batch、队列和设备读回建立延迟/吞吐的基本模型；
7. 解释 Burn Remote 的 compute peer 边界，以及 WASM 异步连接的限制；
8. 识别固定快照中已验证、只完成源码核验和仍需额外工程的能力。

## 先修知识

建议先完成第 2 章的 Tensor、Device、Module 和 ModuleRecord，第 4 章的
IR/融合与同步边界，第 5 章的数据管道，以及第 6 章的训练状态。需要理解
Rust trait、所有权、二进制 artifact 和基本的延迟/吞吐概念。不要求先拥有
CUDA、浏览器或部署集群。

## 本章路线

先把部署问题放到框架无关的边界上，再进入固定源码：

![部署主路径：训练状态 → artifact → convert/validate/optimize → runtime model，再经 pre/post 与 batch/queue 服务边界](img/ch07-serving-pipeline.svg)

`ModuleRecord` 适合验证“参数能否恢复到一个 Burn module”；`burn-onnx` 负责
把 ONNX 图生成成 Burn Rust source，并为生成模型安排 Burnpack 权重加载；
Remote 把 tensor operation 送到 compute peer；服务端的路由、鉴权、版本、
限流和故障转移则仍然是应用系统的职责。

本章的固定版本警告很重要：`pins.toml` 中的 `burn-onnx` revision 的
manifest 仍指向较早的 Burn revision，与根 workspace 使用的 Burn revision 不同。
因而本章把 ONNX 源码核验和当前 workspace 的 `ModuleRecord` CPU 实验分开，不用一个未对齐
的依赖图制造“端到端已验证”的印象。

## 小节

1. [部署边界、artifact 与服务成本](ch07/01-deployment-boundary.md)
2. [ONNX、图转换与 Burn Rust 代码生成](ch07/02-onnx-and-codegen.md)
3. [ModuleRecord、Burnpack 与权重格式](ch07/03-record-and-artifacts.md)
4. [压缩、精度与离线优化](ch07/04-compression-and-optimization.md)
5. [推理 runtime、批处理与服务接口](ch07/05-inference-runtime-and-service.md)
6. [Remote、WASM/no_std 与部署边界](ch07/06-remote-wasm-and-nostd.md)
7. [实验：CPU 模型状态往返保存与恢复](ch07/07-record-roundtrip-lab.md)
8. [练习、延伸阅读与来源](ch07/08-exercises-and-sources.md)

示例代码位于 `examples/ch07-record-roundtrip`，使用当前项目固定 Burn
revision 的 Flex CPU。它验证的是 Burnpack 参数状态的内存导出/加载和输出
一致性，不下载 ONNX、不启动网络服务，也不把一次 CPU 测试外推为浏览器或
GPU 性能结论。

## 证据状态

以下标签是本书的阅读证据分类，不代表 Burn 官方能力等级；完整定义见
[逐文件对照矩阵导读](crosswalk-guide.md)。

- `CPU 可运行验证`：当前 workspace 的 `ModuleRecord`/Burnpack 参数往返保存与恢复，
  以及恢复后的 inference；
- `源码核验`：burn-onnx 的 graph/codegen/load strategy、Remote、
  WASM/no_std 和当前 workspace 的 artifact 入口；
- `协议/成本模型`：manifest、checksum、版本、rollback、batch/queue
  和安全威胁模型；
- `可选平台实验`：真实 ONNX fixture、服务治理、浏览器/Remote 部署和
  设备性能；
- `未覆盖`：burn-onnx 旧 revision 与当前 workspace Burn 的端到端混用。

对应 artifact manifest、rollback 和动态 batching 见[核心主题比较卡](comparison-cards.md#第-7-章模型部署)。

