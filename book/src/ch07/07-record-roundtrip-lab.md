# 7.7 实验：CPU 模型状态 round-trip

## 实验目标与边界

实验位于 `examples/ch07-record-roundtrip`。它创建一个小型 `Linear` module，
把参数转换为内存 Burnpack bytes，再加载到一个新 module，最后比较两次
forward 的输出。

```text
主线 Burn Linear
      │
      ├── reference forward
      ├── Module::into_record
      ├── ModuleRecord::into_bytes
      ├── ModuleRecord::from_bytes
      └── fresh Linear::try_load_record
                    │
              restored forward
```

本实验验证的是参数 artifact 的保存、加载、shape 和数值等价性。它不验证
ONNX parser/codegen、HTTP/gRPC、Remote、WASM、SafeTensors、量化或 GPU
性能。把这些范围写出来，是为了避免把一个成功的本地 load 当成完整服务
上线证明。

## 1. 创建目标 module 和 reference

示例使用 `Initializer::Constant`，让模型参数不依赖随机初始化。输入是
一个三行、两列的 CPU tensor，输出应为三行、一列；具体输出值不是本实验
关注点，关注点是恢复前后的差异。

## 2. 保存并加载记录

下面的代码是示例源码的唯一正文真相：

```rust,ignore
{{#include ../../../examples/ch07-record-roundtrip/src/lib.rs:run_round_trip}}
```

`into_record` 消费 model，所以代码先用它计算 reference，再把 model 本身
转换为 record。`into_bytes` 返回内存中的 Burnpack；`from_bytes` 只重建
参数记录，不会凭空创建 topology。必须用同一 `LinearConfig` 创建一个新
module，再调用 `try_load_record`。

这里选择 `try_load_record` 而不是 `load_record`，因为服务启动时应把损坏
或不匹配的 artifact 变成可报告的错误。固定源码中的 validation 会检查
shape 和缺失 tensor；生产系统还应在加载前验证 checksum 和版本 metadata。

## 3. 测试可观察不变量

在项目根目录运行：

```bash
cargo test -p ch07-record-roundtrip
cargo run -p ch07-record-roundtrip
```

测试断言：

1. 记录包含两个参数 tensor，即 Linear 的 weight 和 bias；
2. 恢复后的输出 shape 为 `[3, 1]`；
3. reference 与 restored output 的最大绝对误差小于 `1e-6`。

主程序输出类似：

```text
record_tensors=2 output_shape=[3, 1] max_abs_error=0.000000e0
```

小数形式可能随 backend 或上游实现变化，但“参数数目、shape 和误差边界”
是本例的可观察协议。实验没有把 bytes 长度写成跨平台性能结论。

## 4. 从实验走向部署

可以按以下顺序扩展，而不混淆边界：

1. 把内存 bytes 改成临时 `.bpk` 文件，测试文件错误和 checksum；
2. 改用 `DTypePolicy::CastToModule`，比较 F32/F16 或其他目标 dtype；
3. 引入 `burn-store` 的 SafeTensorsStore，并记录路径 remap 和
   `ApplyResult`；
4. 为模型加一个版本 metadata 和固定 reference 输入；
5. 在 host 上用 `burn-onnx::ModelGen` 生成一个 fixture，并对齐它与
   当前主线 Burn revision 后再加入 workspace；
6. 把已加载的 model 放到服务 runner，单独测 queue、pre/post 和
   p95/p99；
7. 具备匹配网络和 backend 后，再尝试 Remote 或浏览器客户端。

第 5 步不能跳过：固定仓库的 `burn-onnx` 仍 pin 到较早 Burn revision，
本章没有把两个 revision 的 generated model 类型混进当前 workspace。

## 5. 接到第 5–6 章

若要从真实数据和训练状态进入 artifact，而不是手工初始化 Linear，请运行
[P1 贯穿实验：数据到推理](../capstone-p1.md)。它在本实验的
`ModuleRecord` round-trip 之前加入 Dataset split、SGD 更新和
`model.valid()`，并用错误 topology 验证加载失败语义。
