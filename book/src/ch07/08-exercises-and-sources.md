# 练习、延伸阅读与来源

## 小结

模型服务把训练状态变成可验证、可加载、可执行和可治理的 artifact。
`ModuleRecord` 是 Burn 核心的参数记录：它保存 parameter path、`ParamId`、
shape、dtype 和 bytes，但不保存可以独立执行的 topology。`burn-store` 在
feature 开启后增加 snapshot、Burnpack/SafeTensors/PyTorch store、过滤、
重命名和 adapter。

固定 `burn-onnx` 的 `ModelGen` 将 ONNX graph 解析成 Burn graph，生成 Rust
source，并收集 Burnpack 权重。`File`、`Embedded`、`Bytes` 和 `None`
决定权重从哪里进入生成的 model。由于该仓库固定快照的 manifest 仍依赖旧
Burn revision，本章对 ONNX 的源码对照与本书 Record 实验保持分离。

Remote 负责把 tensor operation 送到 compute peer；WASM client 的连接
需要异步事件循环；no_std 只缩小标准库依赖，不自动提供文件、网络、线程
或所有 backend。HTTP/gRPC、版本发布、鉴权、限流、指标和故障恢复属于
应用服务层。

本章还有两个纯 Rust 机制实验：`ch07-ptq-calibration` 把量化校准的
scale/zero-point、粒度与离群值交易变成可复现数字；
`ch07-serving-queue-sim` 用虚拟时间队列演示连续批处理与 KV 预算如何
决定延迟和吞吐。它们验证的是协议与机制，不是任何 backend 的性能。

## 练习

练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。


## 概念题

1. 【基础】为什么 `ModuleRecord` 不能单独恢复一个任意模型？列出 topology、
   参数路径、shape 和 dtype 各自的职责。

<details>
<summary>提示</summary>

[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
开头的「参数状态与模型定义」示意图就是本题的骨架。再注意
`examples/ch07-record-roundtrip` 中 `from_bytes` 之后仍要用同一
`LinearConfig` 重新 `init` 一个新 module 才能 `try_load_record`。
想清楚：记录里的 path、shape、dtype 各自在加载时校验什么，而
`forward` 逻辑由谁提供。

</details>

2. 【基础】比较 `LoadStrategy::File`、`Embedded`、`Bytes` 和 `None` 的 artifact
   生命周期；哪一种会在生成代码中使用 `std::path::Path`？

<details>
<summary>提示</summary>

[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)的
「生成代码如何加载权重」逐条列出四种策略生成的入口；关于
`std::path::Path` 的那一问，线索在
[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)
对 `extern crate std` 的讨论里。按「权重字节在构建期还是运行期、
从哪个来源进入 model」给四种策略排一条时间线。

</details>

3. 【基础】为什么把 F32 权重存成 F16 不等于完成了低精度推理？列出加载、算子、
   activation 和 reference 校准还需要的条件。

<details>
<summary>提示</summary>

[「压缩、精度与离线优化」](04-compression-and-optimization.md)开头
把「压缩」拆成四个可以分别改变的对象，存盘表示只是第一个；加载侧
`FromRecord`/`CastToModule` 的语义见
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)。
沿「文件 dtype → 加载 dtype → 算子路径 → 误差证据」逐层追问：
F16 到哪一层就停了，剩下每层还缺什么条件。

</details>

4. 【基础】用 $T\_{\mathrm{queue}}$、$T\_{\mathrm{pre}}$、$T\_{\mathrm{copy}}$、
   $T\_{\mathrm{forward}}$、$T\_{\mathrm{readback}}$ 和
   $T\_{\mathrm{post}}$ 分解一次请求，说明动态 batching 可能改善和恶化
   哪些项。

<details>
<summary>提示</summary>

延迟分解式在
[「部署边界、artifact 与服务成本」](01-deployment-boundary.md)的
「一个最小延迟模型」；动态 batching 的三个合批条件在
[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)。
逐项判断哪些成本能按 batch 摊薄、哪一项必然随等待变长，再想
padding 与 shape bucket 又把无效计算加进了哪一项。

</details>

5. 【进阶】Remote peer、模型 registry、HTTP service 和授权系统分别负责什么？
   为什么它们不能仅由 `Device::remote_iroh` 代替？

<details>
<summary>提示</summary>

[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)
第一段就是起点：Remote 只移动 tensor operation 的执行位置。再拿
[「部署边界、artifact 与服务成本」](01-deployment-boundary.md)的
四对象分解（artifact、Runtime、请求契约、服务策略）当表格，把题中
四个组件各归入一格，看 `Device::remote_iroh` 只落在哪一格、其余
格子由谁补齐。

</details>

6. 【进阶】“生成的 model 可以 no_std”与“ONNX converter 可以在 no_std 目标运行”
   为什么是两个构建问题？

<details>
<summary>提示</summary>

[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)的
「`no_std` 的范围」把 converter 定位成 build-time/CLI 工具，并列出
它与生成 model 各自的依赖集合。分别写出两个阶段运行在哪台机器上、
各需要哪些宿主能力（ONNX parser、codegen、文件系统，对比 `alloc`
与目标 backend），答案自然分开。

</details>


## Rust 与 API 题

1. 【基础】把实验的 Linear 改成两个 Linear 组成的 `Module`，断言 record tensor
   数量和输出误差。

<details>
<summary>提示</summary>

`examples/ch07-record-roundtrip` 的 `run_round_trip` 里
`record_tensors = record.len()`，单个 Linear 是 2（weight 与
bias）。先推算两层后的期望值再动手改；记录收集的是整棵 module
树的参数，嵌套字段不需要手写遍历。沿用 `Initializer::Constant`
可以让输出误差断言保持确定性。

</details>

2. 【基础】使用 `try_load_record` 构造 shape mismatch，记录 `RecordError::Validation`
   的行为；再比较 `validate(false)` 和 `allow_partial(true)` 的差异。

<details>
<summary>提示</summary>

在示例里把恢复侧的 `LinearConfig` 维度改掉（比如 `new(3, 1)`）就能
制造 mismatch。两个开关的语义见
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的「核心 `ModuleRecord`」，实现在
`burn/crates/burn-core/src/store/mod.rs`。分别问：它们各跳过哪类
检查，跳过之后哪种错误会被推迟到什么时候才暴露。

</details>

3. 【进阶】让目标 module 使用不同 dtype，比较默认 `FromRecord` 与
   `cast_to_module_dtype` 的结果。

<details>
<summary>提示</summary>

两种 policy 谁说了算，
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的「dtype policy 与布局」各有一句话定义。在 round-trip 实验上加一个
不同 dtype 的目标 module，观察加载后参数 dtype 与 `max_abs_error`
相对 `1e-6` 容差的变化；记住这只是加载时的数据类型策略，不是量化
校准。

</details>

4. 【进阶】写一个服务 runner，让 model 只在启动阶段加载一次；用 Rust ownership
   明确 handler 的借用、锁或 actor 边界。

<details>
<summary>提示</summary>

[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)
的「推理 runtime 的状态」给出启动六步和「handler 只借用或经锁/actor
访问」的所有权约束。注意实验里 `into_record` 会消费 model——哪些
API 拿走所有权，决定了共享状态里能存放什么。先写清类型：谁持有
model、handler 拿到的是共享引用还是消息通道，再填实现。

</details>

5. 【进阶】给输入增加版本化 schema 和显式前处理函数，测试错误输入不会进入
   `forward`。

<details>
<summary>提示</summary>

[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)
开头强调前处理属于模型契约、schema 要与模型版本绑定；该节末尾测试
分层里的 contract test 就是本题要写的那一层。设计方向：让前处理
返回 `Result`，非法输入在构造 tensor 之前就被拒绝，`forward` 只
接受已通过校验的类型化 batch。

</details>

6. 【进阶】用一个 `Bytes` provider 模拟远端/固件来源，比较 `from_bytes` 和
   `from_file` 的错误处理。

<details>
<summary>提示</summary>

起点是实验页「从实验走向部署」的第 1 步：把内存 bytes 换成临时
`.bpk` 文件；`ModuleRecord::save/load` 的文件接口见
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)。
示例的 `inspect_burnpack_layout` 测试演示了截断与坏 magic 两类字节
层错误，比较时区分路径/IO 错误、格式错误与校验错误各在哪一层报出。
生成代码的 `from_file` 只需对照固定 `burn-onnx` 源码阅读，不要接入
本书 workspace。

</details>


## 源码题

1. 【进阶】阅读 `burn/crates/burn-core/src/store/mod.rs`，追踪
   `into_record → into_bytes → from_bytes → try_load_record` 的数据路径，
   并找出 `ParamId` 恢复位置。

<details>
<summary>提示</summary>

[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的「核心 `ModuleRecord`」列出了这条链上的每个方法，可当阅读地图；
每一步的可观察结果（`record.len()`、误差边界）在
`examples/ch07-record-roundtrip` 的 round-trip 测试里都有对照。找
`ParamId` 时盯住加载侧：正文说恢复保存 id 的是 module mapper，在
源码中定位这段逻辑。

</details>

2. 【进阶】阅读 `burn/crates/burn-core/src/module/base.rs`，比较
   `load_record` 与 `try_load_record` 的错误边界。

<details>
<summary>提示</summary>

正文称 `load_record` 是「失败时 panic 的便利方法」——便利方法通常
包装另一个入口，读源码时先找出谁调用谁、panic 消息从哪来。实验页
[「实验：CPU 模型状态往返保存与恢复」](07-record-roundtrip-lab.md)
解释了服务启动路径为什么选 `try_load_record`；把两者的返回类型和
失败时机放到「启动阶段」与「请求处理中」两个场景里比较。

</details>

3. 【进阶】阅读 `burn-onnx/crates/burn-onnx/src/model_gen.rs` 和
   `burn-onnx/crates/burn-onnx/src/burn/graph.rs`，画出 ONNX parser、
   graph simplification、codegen 和 Burnpack loader 的调用链。

<details>
<summary>提示</summary>

[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)的
「`ModelGen` 的转换路线」已给出四阶段骨架，把它当待验证的
假设：在源码里定位 `OnnxGraphBuilder`、`ParsedOnnxGraph::into_burn`、
`BurnGraph::codegen` 与 `register_burnpack_loaders`，补全箭头之间
省略的中间结构。该仓库依赖较早的 Burn revision，只读源码即可，
不要把它接入本书 workspace。

</details>

4. 【进阶】阅读 `burn-onnx/crates/burn-onnx/src/burn/graph.rs` 的
   `LoadStrategy` 测试，确认四种策略生成了哪些 constructor。

<details>
<summary>提示</summary>

先按[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)
「生成代码如何加载权重」的四条列表写下预测，再到该文件的测试里
核对生成 token 中出现的 constructor 名。特别留意 `Default` 在
`File` 与 `Embedded` 下为什么含义不同（生成时的文件路径，对比
`include_bytes!` 进 binary）。

</details>

5. 【进阶】阅读 `burn/crates/burn-store/src/traits.rs`、`adapter.rs` 和
   `tensor_snapshot.rs`，说明 lazy snapshot、filter、remap 和 adapter
   的边界。

<details>
<summary>提示</summary>

[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的「`burn-store` 提供的更大边界」逐项介绍了这些抽象。读源码时按
职责分组：谁决定选哪些参数、谁决定叫什么名字、谁改数值/布局、谁
决定何时 materialize tensor data；并找出正文「lazy 不等于设备端
zero-copy」对应的实现证据。

</details>

6. 【进阶】阅读 `burn/crates/burn-remote/src/lib.rs`、`server/builder.rs` 和
   `burn/crates/burn-tensor/src/device.rs`，比较 native Iroh、兼容
   WebSocket 和 WASM async device 的入口。

<details>
<summary>提示</summary>

三个入口名（`remote_iroh`、`remote_websocket`、`remote_iroh_async`）
在[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)
已点出。在 `device.rs` 里对比它们的条件编译差异，在
`server/builder.rs` 里看 `RemoteServerBuilder` 为何要求具体 backend
的非空设备列表；再想「浏览器主线程不能阻塞连接」这一约束如何落在
API 形状（同步返回，对比 async）上。

</details>

7. 【进阶】对照仓库版本钉扎、`burn-onnx/Cargo.toml` 和根 `Cargo.toml`，解释为什么
   两个 Burn revision 不能直接共用 `Tensor`/`Module` 类型。

<details>
<summary>提示</summary>

[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)末段
已给出结论（`976aa9...` 对 `78f10a...`，同名 package 不同 revision
即不同类型）；你的任务是把证据链补全：在两份 manifest 里找到各自
的 `rev` 字段，再从 Cargo 依赖解析的角度说明为什么它们是两个 crate
实例。这也是本书不把 ONNX 端到端接入同一依赖图的原因。

</details>


## 性能与系统题

1. 【进阶】测量 cold start、artifact load、warmup、forward、readback 和
   post-processing，报告设备、backend、dtype、shape 和同步边界。

<details>
<summary>提示</summary>

以 `examples/ch07-record-roundtrip` 的 `run_round_trip` 为骨架插入
分段计时，把 `from_bytes`、`try_load_record`、首次 forward 与稳态
forward 分开记录；启动六步与 warmup 的位置见
[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)。
注意首次 forward 可能触发 lazy allocation 等一次性成本，与稳态
混在一起会污染结论。

</details>

2. 【挑战】实现动态 batching，分别固定最大 batch size 和最大 queue delay，
   报告 throughput、p50、p95、p99 以及 queue wait。

<details>
<summary>提示</summary>

纯逻辑起点是示例里的 `dynamic_batch_groups`：按 shape 键分组并受
最大 batch size 限制，其测试给出了分组边界的预期，本题相当于给它
加上时间维度。合批三条件见
[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)。
思考两种上限各自主导哪个指标：batch 上限主要影响吞吐还是延迟，
queue delay 上限又会出现在哪个分位数里。

</details>

3. 【挑战】用固定 reference 输入比较 F32 与一种目标 dtype，分别记录模型大小、
   peak memory、输出误差和延迟；不要只报告压缩率。

<details>
<summary>提示</summary>

报告模板是
[「压缩、精度与离线优化」](04-compression-and-optimization.md)末尾
「精度—延迟—内存的验证闭环」：error、memory、latency 三组证据缺一
不可。加载侧入口可用
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的 dtype policy。固定输入 schema、batch、backend 与测量方式再比较；
CPU 上的结论不要外推到其他 backend。

</details>

4. 【挑战】设计模型版本发布协议，列出 topology checksum、weight checksum、
   schema version、code revision、backend 和回滚点。

<details>
<summary>提示</summary>

示例里的 `ArtifactManifest` 与 `rollback_allowed` 测试给了最小骨架
（checksum、版本号、回滚条件），本题是把它扩成完整字段表。字段
清单可对照
[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)末尾
「验证转换时建议记下什么」。给每个字段写一句「它能把哪类故障与
其他故障区分开」，协议就成形了。

</details>

5. 【挑战】设计 Remote 服务的网络故障测试：peer 断开、请求超时、重复提交、
   tensor transfer 失败和模型 reload。

<details>
<summary>提示</summary>

Remote 的成本与失效点列在
[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)；
把每种故障映射到
[「部署边界、artifact 与服务成本」](01-deployment-boundary.md)中
remote 延迟分解的某一项，再决定注入与观测手段。重复提交要考虑
幂等；真实 Iroh 网络属可选前提，故障矩阵与恢复协议可先用本地
模拟 peer 验证。

</details>

6. 【挑战】为 `Embedded`/`Bytes` 的嵌入式部署列出 binary、静态内存、堆、最大
   tensor shape 和算子覆盖预算。

<details>
<summary>提示</summary>

预算五项正文已点名：
[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)的
「`no_std` 的范围」末尾就是这份测量清单。结合
[「ONNX、图转换与 Burn Rust 代码生成」](02-onnx-and-codegen.md)对
`Embedded`（`include_bytes!` 进 binary）与 `Bytes`（调用者供字节）
的生命周期描述，分析权重进 binary 后每项预算与更新方式怎么变。

</details>

7. 【挑战】比较“模型文件加密”“transport authorization”“TEE”和“模型混淆”
   所保护的威胁，避免把它们当成同一个开关。

<details>
<summary>提示</summary>

对照表正文已备好：
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
的威胁四边界（at rest、in transit、in use、model behavior）加上
[「Remote、WASM/no_std 与部署边界」](06-remote-wasm-and-nostd.md)的
「安全边界」小节。先写清每种机制针对的攻击面，再找它们互不覆盖
的空白，「不是同一个开关」就有了具体证据。

</details>

8. 【挑战】为一个激活张量选择 PTQ 校准集，计算非对称量化的 scale/zero-point，
   比较逐层、逐通道和离群值裁剪的误差与 metadata 成本。

<details>
<summary>提示</summary>

从 `examples/ch07-ptq-calibration` 出发：它已实现 scale/zero-point
推导、min-max/百分位校准、per-channel 重建和主体/离群值分开的 MSE，
照着改校准策略与粒度即可。对照
[「压缩、精度与离线优化」](04-compression-and-optimization.md)的
带数字演算读输出；粒度对比时数一数逐通道需要多少组
scale/zero-point、metadata 随之怎么涨。实验是协议层演算，不依赖
低精度 backend kernel。

</details>

9. 【挑战】为一个线上模型写四层威胁模型：静态 artifact、传输、运行时内存和
   恶意行为；给每层列出验证证据，并说明 `ModuleRecord` 哪些问题不能
   单独解决。

<details>
<summary>提示</summary>

题面四层就是
[「ModuleRecord、Burnpack 与权重格式」](03-record-and-artifacts.md)
威胁模型段的展开。逐层清点手头已有的证据：示例的
`ArtifactManifest` checksum、Burnpack 头部读取器对截断/坏 magic 的
拒绝、固定源码内置的 metadata 与 tensor 数上限，各覆盖哪一层的哪个
子问题；剩下没有任何机制覆盖的空白，就是 `ModuleRecord` 单独解决
不了的部分。

</details>

10. 【挑战】给 `ch07-serving-queue-sim` 增加 chunked prefill：把大 prompt
    切成固定大小的块分多步处理，比较长 prompt 到达时其他请求的
    p95 延迟变化。

<details>
<summary>提示</summary>

当前模型在接纳步一次处理全部 prompt token（见
[「推理 runtime、批处理与服务接口」](05-inference-runtime-and-service.md)
「动手版」小节声明的简化），一条 512 token 的 prompt 会拖慢同一步里
所有请求——这正是要测的干扰。改法：`Running` 增加未完成的 prefill
余量，每步最多处理 `chunk` 个 prompt token；断言总 token 数守恒，
再比较 p95。

</details>


## 延伸阅读与固定源码入口

量化、蒸馏、推理服务与大模型服务的论文见附录
[参考文献](../references.md#第-7-章-模型服务)。
本书示例使用的 Burn：

- `burn/crates/burn-core/src/store/mod.rs`
- `burn/crates/burn-core/src/module/base.rs`
- `burn/crates/burn-pack/src/base.rs`（Burnpack 头部、对齐与容量上限）
- `burn/crates/burn-pack/src/writer.rs`（CBOR metadata 与 tensor 布局规划）
- `burn/crates/burn-store/src/traits.rs`
- `burn/crates/burn-store/src/burnpack/`
- `burn/crates/burn-store/src/safetensors/`
- `burn/crates/burn-store/src/adapter.rs`
- `burn/crates/burn-store/src/tensor_snapshot.rs`
- `burn/crates/burn/Cargo.toml`
- `burn/crates/burn-tensor/Cargo.toml`
- `burn/crates/burn-tensor/src/device.rs`
- `burn/crates/burn-remote/README.md`
- `burn/crates/burn-remote/src/lib.rs`
- `burn/crates/burn-remote/src/server/builder.rs`
- `burn/examples/remote-inference-web/README.md`

固定 `burn-onnx`：

- `burn-onnx/Cargo.toml`
- `burn-onnx/README.md`
- `burn-onnx/crates/burn-onnx/src/model_gen.rs`
- `burn-onnx/crates/burn-onnx/src/burn/graph.rs`
- `burn-onnx/crates/burn-onnx/src/bin/onnx2burn.rs`
- `burn-onnx/examples/onnx-inference/README.md`

OpenMLSys v1：

- `openmlsys/v1/zh_chapters/chapter_model_deployment/index.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_deployment_introduction.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_converter_and_optimizer.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_compression.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_inference.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/model_security.md`
- `openmlsys/v1/zh_chapters/chapter_model_deployment/summary.md`

## 本章系统结论

1. 部署闭环是：训练态 → artifact → 校验 → 推理入口 → 服务侧 batch/队列/版本。
2. `ModuleRecord`/Burnpack 管参数状态；ONNX/codegen 管图转换，二者版本不能混用。
3. 推理选用哪个 Device（CPU/WGPU/CUDA）与 artifact 格式正交，但影响延迟与批处理。
4. CPU 上你验证了 Linear 参数往返保存/恢复与输出误差边界、PTQ 校准的误差交易，以及连续批处理与 KV 预算的队列行为。
5. GPU 阅读线索：同一 record 加载到加速 Device 后的同步与 batch 合并；LLM 服务的机制模型见 `ch07-serving-queue-sim`，工程实现见参考文献。
6. 不能把本地 load 成功当成完整服务上线或 ONNX 端到端已在同一依赖图验证。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-7-章)。
