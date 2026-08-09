# 许可、来源与独立性声明

本书改编自 OpenMLSys Team 的《机器学习系统：设计和实现》，原项目地址
为 <https://github.com/openmlsys/openmlsys>，采用 CC BY-NC-SA 4.0。
本书正文同样采用 CC BY-NC-SA 4.0。

改编工作包括重新组织章节、以 Burn/Rust 示例替换框架专用实现，并更新
编译器、加速器、训练和部署内容。各章文件级改编说明见
[来源与改编总录](appendix-sources.md)。

本书使用 Burn、CubeCL、CubeK 和 burn-onnx 作为源码案例，但不是
OpenMLSys 或 Tracel 的官方项目，与二者均无隶属关系。

九章默认在 CPU 上可运行；真实 GPU、NCCL、ONNX fixture、DDP、DQN/MARL
和集群控制面需要额外环境。固定版本号与 burn-onnx 的版本关系见
[范围、证据与对照附录](appendix-scope-and-evidence.md#固定版本)。

完整项目许可边界和改编声明见仓库根目录的 `LICENSE.md` 与 `NOTICE.md`。
