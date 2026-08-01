# 许可、来源与独立性声明

本书改编自 OpenMLSys Team 的《机器学习系统：设计和实现》，原项目地址
为 <https://github.com/openmlsys/openmlsys>，采用 CC BY-NC-SA 4.0。
本书正文同样采用 CC BY-NC-SA 4.0。

改编工作包括重新组织章节、以 Burn/Rust 示例替换框架专用实现，并更新
编译器、加速器、训练和部署内容。每章将单独列出实际使用的原始章节和
主要改动。

本书使用 Burn、CubeCL、CubeK 和 burn-onnx 作为源码案例，但不是
OpenMLSys 或 Tracel 的官方项目，与二者均无隶属关系。

本书首个候选版固定在 `pins.toml` 的 OpenMLSys revision
`9c289782ccbb165ac8ad7c960ecffc12942a5560`、Burn revision
`976aa9c5ec1d2dd3412710f99759e3c44bdff03d`、CubeCL revision
`be278a1e76aed881e2cc6b165414ee6103ca4634` 和 CubeK revision
`f82a6d07ebf35a1d446893b32712458744d80f13`；Burn 版本线是
`0.22.0-pre.1`。`burn-onnx` 使用 revision
`af2dfb43af43bf363dc2d7d858d933d86e2a65a8` 和不同的 Burn revision，
因此只作为源码证据和可选比较轨道，不进入主线 Cargo workspace。
九章主线默认是 CPU-first；真实 GPU、NCCL、ONNX fixture、DDP、DQN/MARL
和集群控制面需要额外环境或协议，不属于默认已验证能力。

完整项目许可边界和改编声明见仓库根目录的 `LICENSE.md` 与 `NOTICE.md`。

