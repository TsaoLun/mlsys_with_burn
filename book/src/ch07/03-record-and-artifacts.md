# ModuleRecord、Burnpack 与权重格式

## 参数状态与模型定义

部署 artifact 经常把两件事放在一个文件或一组文件里：

- **模型定义**：结构、节点、输入输出和 forward 逻辑；
- **参数状态**：weight、bias、normalization state 以及它们的 shape/dtype。

Burn 的 `Module` trait 让模型定义成为 Rust 类型；`ModuleRecord` 则把 module
树中的参数收集成一个非泛型记录。记录本身不携带一个可以独立执行的
`forward`。加载时仍然需要先创建目标 module，再把记录应用到它。

```text
Rust Model::new(device)  ── topology + initial params
          │
          ├── into_record() ── path + ParamId + TensorData
          │                         │
          │                  Burnpack bytes/file
          │                         │
          └── try_load_record(from_bytes) ◄── fresh Model::new(device)
```

这也是为什么一个“只有权重”的文件无法恢复任意模型：参数路径必须匹配，
shape 和 dtype 必须能加载，目标 module 的结构还必须存在。

## 核心 `ModuleRecord`

固定版本的 `burn-core/src/store/mod.rs` 给出的最小 API 是：

- `Module::into_record` 遍历 module 参数；
- `ModuleRecord::into_bytes` 将记录序列化为内存 Burnpack；
- `ModuleRecord::from_bytes` 从内存 Burnpack 重建记录；
- `Module::try_load_record` 应用记录并返回 `RecordError`；
- `Module::load_record` 是失败时 panic 的便利方法；
- `ModuleRecord::save/load` 在 `std` 下提供文件路径接口。

记录保存 parameter path、`ParamId`、shape、dtype 和 bytes。加载默认会检查
shape 是否匹配，以及目标 module 是否缺少记录中的参数；`allow_partial(true)`
和 `validate(false)` 可以改变行为，但应只在明确知道缺失或映射原因时使用。
教学代码和服务启动路径优先使用 `try_load_record`，把损坏 artifact 变成
可报告的错误，而不是在请求处理中突然 panic。

`ParamId` 不是用户给出的 layer name。它是参数状态关联的重要标识，固定
源码的 mapper 会在加载时恢复保存的 id。因此保存 model 与 optimizer state
时，不能随意丢掉它或只复制一份 tensor 数值。

## dtype policy 与布局

`DTypePolicy::FromRecord` 是默认行为：目标参数采用记录中的 dtype。
`CastToModule` 则按照目标 module 当前参数 dtype 转换记录。两者解决的是
“加载时的数据类型策略”，不是量化校准，也不能保证误差满足业务需求。

Module mapper 还可能把保存形态与 live 形态分开。例如 `Linear` 的某些
layout 会在保存和加载时做转置。只看一个 tensor 的 shape 而不看
`Module` 的 mapper，容易把“存储布局”和“forward 布局”混为一谈。

因此，跨设备或跨格式加载的检查顺序应是：

1. 读取 metadata、dtype、shape 和路径；
2. 选择目标 backend/device；
3. 让 module 的 mapper 处理布局；
4. 根据 policy 处理 dtype；
5. 用 reference input 比较输出；
6. 记录容差和失败 tensor。

## `burn-store` 提供的更大边界

当核心 `ModuleRecord` 不够用时，固定 `burn-store` 提供
`ModuleSnapshot`/`ModuleStore` 抽象。它把 module traversal 与具体存储格式
分开，并支持：

- `BurnpackStore`：面向 Burn snapshot 的 Burnpack 文件；
- `SafetensorsStore`：适合以 tensor metadata、offset 和 bytes 为中心的
  文件/内存存储；
- `PytorchStore`：在启用 PyTorch feature 时导入 `.pt/.pth`；
- `PathFilter`：只选择某些 module path；
- `KeyRemapper`：在不同命名规则之间重命名；
- `PyTorchToBurnAdapter` 与 `BurnToPyTorchAdapter`：处理 Linear 权重
  转置、BatchNorm/LayerNorm 参数命名等；
- `HalfPrecisionAdapter` 与 `FloatCastAdapter`：显式做存储或加载 dtype
  转换。

`ModuleSnapshot` 的 snapshot 可以延迟 materialize tensor data；这对检查
大模型的 path、shape 和 dtype 有用，但“lazy”不等于设备端 zero-copy。
固定源码也提示，文件 mmap、静态 bytes 和 backend 内部 tensor allocation
是不同层次的复制问题。

## 格式、拓扑和安全

SafeTensors 这类纯 tensor 数据格式可以避免加载时执行任意模型代码，但它
不能替代对模型拓扑、参数路径、版本和数值 reference 的检查。Burnpack 也
不是加密容器；如果模型是商业秘密，传输加密、静态加密、密钥管理和运行
时保护要单独设计。

同样，`Embedded` 把 `.bpk` 放进 binary 只改变分发和读取路径，不等于模型
被保护。攻击者仍可能分析 binary 或运行时内存。模型安全要先定义威胁：
防传输窃听、防文件替换、防内存转储，还是防模型结构分析；每个目标需要
不同机制。

可以把最小威胁模型写成四条边界：

- **artifact at rest**：文件被读取、替换或回滚；需要访问控制、完整性
  checksum/signature、版本和密钥轮换；
- **in transit**：模型或输入在 peer/服务之间被窃听或重放；需要传输
  加密、peer 身份和请求授权；
- **in use**：进程、日志、core dump 或设备内存暴露权重；需要最小权限、
  secret 生命周期、内存清理或 TEE 等额外机制；
- **model behavior**：恶意/不兼容 artifact 触发错误算子、资源耗尽或
  输出后门；需要 schema/算子 allowlist、资源上限、reference 和审计。

静态加密主要保护第一条，TLS/Iroh transport 主要保护第二条，TEE 试图
缩小第三条，模型签名和加载校验主要帮助第四条中的篡改检测；它们不能
互相替代。固定 Burn `ModuleRecord`、Remote endpoint 或
`PeerAuthorizer` 只提供可组合接口，不能单独构成上述安全产品。

本章实验刻意只覆盖最小边界：根 workspace 的 Burn Linear module 经过
`ModuleRecord → Burnpack bytes → ModuleRecord → module` 后，CPU forward
输出保持一致。下一节再讨论压缩和图优化为什么必须用精度/性能数据证明。
