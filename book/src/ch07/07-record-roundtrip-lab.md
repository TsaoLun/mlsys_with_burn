# 实验：CPU 模型状态往返保存与恢复

## 你会学到什么

示例在 `examples/ch07-record-roundtrip`：创建一个小型 `Linear`，把参数
写成内存里的 Burnpack bytes，再加载到新 module，比较两次 forward。

```text
本书示例中的 Burn Linear
      │
      ├── reference forward
      ├── Module::into_record
      ├── ModuleRecord::into_bytes
      ├── ModuleRecord::from_bytes
      └── fresh Linear::try_load_record
                    │
              restored forward
```

你会观察参数保存、加载、shape 和数值是否一致。本实验刻意不做 ONNX
导入、HTTP/gRPC、Remote、WASM、SafeTensors、量化或 GPU 性能——本地
load 成功不等于服务已经上线。

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
或不匹配的 artifact 变成可报告的错误。源码中的 validation 会检查
shape 和缺失 tensor；生产系统还应在加载前验证 checksum 和版本 metadata。

## 3. 打开 Burnpack 字节

`from_bytes` 能跑通，只说明 API 层契约成立。部署时更底层的问题是：
这串字节是不是我期望的格式、什么版本、参数数据从哪里开始？固定版本
`burn-pack` 的容器布局很简单，可以不用任何序列化库直接读：

```text
header（10 字节，小端）
  magic u32        = 0x4255524E（"BURN"）
  version u16      = 1
  metadata_size u32
metadata（CBOR）：tensor 名 → dtype / shape / data_offsets / param_id
tensor data：起点对齐到 256 字节，每个 tensor 的起点也按 256 对齐
```

一个容易踩的细节：magic 常量写成 `"BURN"`，但小端落盘后文件头四个字节
是 `NRUB`——字母顺序反过来不是损坏，而是字节序的直接后果。示例按这份
规格手写一个最小读取器：

```rust,ignore
{{#include ../../../examples/ch07-record-roundtrip/src/lib.rs:burnpack_layout}}
```

256 字节对齐不是洁癖：它让文件里的绝对偏移可以直接用于 mmap 零拷贝
加载，也满足设备端读取的对齐偏好。代价可以用本实验的真实输出算出来：
两个参数总共只有 12 字节（weight 8 + bias 4），但因为每个 tensor 各占
一段 256 对齐的区域，容器总长是 `256（header+metadata 对齐）+ 256
（weight）+ 4（bias）= 516` 字节。小 artifact 上对齐开销巨大；参数以
GB 计时它又可以忽略——这个比例本身就说明格式在为谁优化。源码还
内置了防滥用的上限（metadata ≤ 100 MB、tensor 数 ≤ 100 000、CBOR 递归
≤ 128 层），加载不可信文件时这些上限就是第一道防线。

## 4. 运行并观察

在项目根目录运行：

```bash
cargo test -p ch07-record-roundtrip --locked
cargo run -p ch07-record-roundtrip --locked
```

你应能观察到：

1. 记录包含两个参数 tensor（Linear 的 weight 和 bias）；
2. 恢复后的输出 shape 为 `[3, 1]`；
3. reference 与 restored output 的最大绝对误差小于 `1e-6`；
4. Burnpack 头读起来是 `NRUB`、版本为 1，tensor 数据区从 256 的整数倍
   开始。

主程序输出类似：

```text
record_tensors=2 output_shape=[3, 1] max_abs_error=0.000000e0
burnpack magic=NRUB version=1 metadata_bytes=133 data_section_start=256 total_bytes=516
```

小数形式可能随 backend 变化；`metadata_bytes` 与 `total_bytes` 由当前
版本的 CBOR 字段决定，升级版本线时应重新核对。请抓住“参数数目、shape、
误差边界和头部契约”，不要把 bytes 长度当成性能结论。

## 5. 从实验走向部署

可以按以下顺序扩展，而不混淆边界：

1. 把内存 bytes 改成临时 `.bpk` 文件，测试文件错误和 checksum；
2. 改用 `DTypePolicy::CastToModule`，比较 F32/F16 或其他目标 dtype；
3. 引入 `burn-store` 的 SafeTensorsStore，并记录路径 remap 和
   `ApplyResult`；
4. 为模型加一个版本 metadata 和固定 reference 输入；
5. 在 host 上用 `burn-onnx::ModelGen` 生成一个 fixture，并先对齐它与
   本书示例使用的 Burn revision，再单独组织依赖；
6. 把已加载的 model 放到服务 runner，单独测 queue、pre/post 和
   p95/p99；
7. 具备匹配网络和 backend 后，再尝试 Remote 或浏览器客户端。

第 5 步不能跳过：`burn-onnx` 固定快照仍指向较早 Burn revision，
本章没有把两个 revision 的 generated model 类型混进同一依赖图。

## 6. 接到第 5–6 章

若要从真实数据和训练状态进入 artifact，而不是手工初始化 Linear，请运行
[综合实验：数据到推理](../capstone.md)。它在本实验的
`ModuleRecord` 往返保存与恢复之前加入 Dataset split、SGD 更新和
`model.valid()`，并用错误 topology 验证加载失败语义。
