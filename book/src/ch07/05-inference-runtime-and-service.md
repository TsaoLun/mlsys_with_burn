# 7.5 推理 runtime、批处理与服务接口

## 前处理和后处理属于模型契约

模型输入的 tensor shape 不等于用户输入。图像服务可能接收 JPEG，文本
服务可能接收 UTF-8 字符串，推荐系统可能接收带版本的 feature map。
前处理（pre-processing）把请求转换成模型期望的 dtype、shape、layout 和
范围；后处理（post-processing）把 logits、概率或检测框转换成 API 返回
的数据。

```text
request
  └── validate schema
      └── decode / normalize / tokenize
          └── Tensor on Device
              └── model.forward
                  └── read result
                      └── threshold / decode / label map
                          └── response
```

前处理和后处理必须和模型版本绑定。只升级权重而不升级 tokenizer、
normalization 常量或 label map，会产生“模型执行成功但业务结果错误”的
故障。

## 推理 runtime 的状态

服务启动时通常一次性完成：

1. 选择 backend 和 Device；
2. 创建 model topology；
3. 从 file、embedded bytes、内存 bytes 或其他 store 加载参数；
4. 进行一次 warmup，触发 lazy allocation、编译或 autotune；
5. 发布 readiness；
6. 收集 model revision、dtype、shape 和设备信息。

请求路径只应使用已经准备好的模型，避免每个请求重新读权重或重新初始化
runtime。若模型包含 autodiff backend，推理应切换到不建立训练 tape 的
有效模型路径；训练的 backward/optimizer state 不应留在请求 handler 中。

服务关闭时要考虑正在排队的请求、设备同步和流式响应。一个简单的 Rust
所有权约束是：启动阶段把 model 放入共享服务状态，handler 只借用或通过
明确的锁/actor 访问；不要在多个线程中隐式复制一个大型参数集合。

## batching 与队列

动态 batching 的基本流程是：

```text
incoming requests
       │
       ├── max batch size
       ├── max queue delay
       └── compatible shape/dtype/model version
                 ▼
              batcher
                 ▼
             one forward
                 ▼
           split responses
```

batcher 需要处理不同输入 shape、请求取消、超时和部分失败。如果请求
不能拼成同一 batch，应按模型版本、shape bucket 或最大 padding 比例
分组。padding 能提高 kernel 规则性，却增加无效计算；过细的 bucket
减少 padding，却降低合批率。

吞吐常可近似为：

$$
\mathrm{throughput} \approx
\frac{\mathrm{completed\ samples}}
{\mathrm{compute + queue + transfer\ time}}.
$$

但在 tail latency 受约束的服务中，最大化吞吐不是唯一目标。应同时报告
单请求与 batch 请求的 p50/p95/p99，以及 queue wait 和 forward 的分位数。

## worker pool、layout 与算子路径

服务进程通常还要在三个并发层次之间取舍：

- 请求线程或 async runtime 负责接收、校验和取消；
- batcher/worker pool 负责按 model version、shape bucket 和优先级组批；
- device stream 负责提交 kernel、读回和同步。

worker 数过少会让 CPU 前后处理成为瓶颈；过多会增加锁、上下文切换、
内存副本和设备 queue contention。一个“并发数更高”的结果如果没有区分
queue wait 和 forward time，不能说明模型算得更快。

layout 与算子优化也要放在端到端路径中判断。NCHW→NHWC、transpose、
padding 或 token packing 可能使单个 kernel 更适合向量化，却增加一次完整
内存搬运。只有当 layout 转换成本小于后续算子节省的时间，计划才可能
收益；这与第 4 章的 fusion、lifetime 和 fallback 条件相同。固定 Burn
提供 Tensor/Device/backend 的执行入口，但没有一个统一的生产线程池、
动态 batch 服务或自动 layout planner。

## Burn API 与服务框架的边界

Burn 提供 Tensor、Module、Device、backend 和部分 Remote server/client
入口；它不是一个固定的 HTTP/gRPC/REST 服务治理层。以下职责通常由应用
或平台实现：

- HTTP/gRPC/WebSocket 协议和 schema；
- 认证、授权、租户隔离和请求签名；
- rate limit、优先级、重试、熔断和 backpressure；
- model registry、灰度、回滚和版本路由；
- metrics、tracing、日志脱敏和审计；
- 多进程 worker、设备调度和故障转移。

把这些逻辑直接塞进 `forward` 会让模型不可复用，也难以测试。更好的边界
是让 service adapter 负责请求生命周期，让 model runner 只消费已经校验
的 typed batch，并返回 typed output。

## 正确性与性能测试

一个最小推理测试套件应分层：

- **artifact test**：状态可加载，path/shape/dtype 正确；
- **numerical test**：固定输入与 reference 比较；
- **contract test**：缺字段、错误 dtype、错误 batch 和超时行为；
- **runtime test**：warmup、冷启动、内存和设备同步；
- **load test**：并发、batch、queue delay、p95/p99 和错误率；
- **recovery test**：模型 reload、worker 重启和 backend error。

本章的 CPU 实验只覆盖第一层以及一次小型 numerical test。它没有伪装
成服务 load test，下一节也不会通过启动远端 server 来掩盖网络环境和
设备前提。
