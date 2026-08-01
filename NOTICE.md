# 来源与改编声明

## OpenMLSys

本教材的主题结构、部分论述和后续可能复用的图表源自：

- OpenMLSys Team，《机器学习系统：设计和实现》
- https://github.com/openmlsys/openmlsys
- https://openmlsys.github.io/
- 许可证：CC BY-NC-SA 4.0

本项目将原书内容重组为以 Burn 与 Rust 为主线的九章教材，并更新框架、
编译器、加速器和部署相关知识。每章应在“来源与改编说明”中记录实际引用、
改写和新增内容。未引用上游表达的原创段落也作为整本衍生教材的一部分，
采用 CC BY-NC-SA 4.0。

## Burn 生态

教材以 Tracel 维护的 Burn、CubeCL、CubeK 和 burn-onnx 为案例。项目会
引用其公开 API、源码路径和文档，但不代表 Tracel，也不是其官方教程。
Rust 示例依据相应项目许可证使用其公开依赖；本项目原创代码采用
MIT OR Apache-2.0，正文与规划文档采用 CC BY-NC-SA 4.0。Burn、CubeCL、
CubeK 与 burn-onnx 的具体许可证和版权归属以各固定上游 checkout 的
许可证文件为准。

## 固定快照与发布边界

本轮发布以 `pins.toml` 中的 `burn-0.22.0-pre.1` 写作快照为准。Burn、
CubeCL、CubeK、OpenMLSys 和 burn-onnx 均由完整 Git revision 标识；
burn-onnx 与主线 Burn 使用不同 revision，因此不进入默认端到端实验。
默认示例是 CPU-first；CUDA、NCCL、真实网络、ONNX fixture、DDP、DQN/
MARL 和 GPU 集群属于需要额外环境的可选比较轨道。

## 独立性声明

“MLSys with Burn”不是 OpenMLSys 或 Tracel 的官方项目，与二者均无
隶属、赞助或背书关系。OpenMLSys、Burn、CubeCL、CubeK 等名称归各自
权利人所有。

