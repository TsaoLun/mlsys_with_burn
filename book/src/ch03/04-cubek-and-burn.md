# CubeK 与 Burn 算子路径

CubeCL 解决“怎样表达和运行 Kernel”，CubeK 解决“怎样组织可复用的高性能
算子实现”。`burn-cubecl` 把二者接入 Burn Backend，并负责 shape 处理、
策略选择、fallback 与 autotune。

## 1. 从架构到调用接口

固定快照中的 CubeK Guide 明确提出 Blueprint–Routine 架构和 Autotuner；
结合 matmul 等 crate 的源码，还可以观察到 Strategy 与 Launch 接口：

1. **Blueprint**：描述会改变控制流或指令选择的编译期特化；
2. **Launch logic**：根据硬件与输入 shape/stride 决定向量化等约束；
3. **Routine**：接收这些约束，计算 cube 拓扑，生成 Blueprint 与
   LaunchSettings；
4. **Strategy**：在具体算子中表示 Naive、CpuGemm、CMMA/MMA、TMA
   等算法族；
5. **Launch API**：接收统一输入 binding，准备并提交具体 Kernel；
6. **Autotuner**：为当前问题寻找较好的 routine/strategy 组合。

这一分层避免把 dtype、布局、tile 和设备的笛卡尔积手写成大量独立函数。
它也把“算法不可用”变成可检测结果，例如设备没有 plane、矩阵指令或足够
共享内存时，候选策略应被过滤。

## 2. `matmul` 调用链

从用户视角看，矩阵乘只是 Tensor 方法。CubeCL Backend 下的核心路径可
概括为：

```text
Tensor::matmul
  → Burn dispatch / CubeCL Backend
  → burn-cubecl 的 shape、layout 与 dtype 准备
  → 候选策略或 LocalTuner
  → cubek::matmul::launch::launch_ref
  → CubeK Routine / Blueprint
  → CubeCL IR 与具体 Runtime
```

CubeK 同时包含朴素算法、CPU 友好的 blocking GEMM 和面向矩阵单元的变体。
不能由 crate 名推断某次调用一定使用 Tensor Core；实际策略取决于 feature、
设备属性、输入与 autotune。

## 3. 覆盖范围与边界

固定 Burn 快照会调用 CubeK 的 matmul、implicit-GEMM convolution、
reduce、attention forward、pool、interpolate、FFT、random 和 quantization
等模块。但这不表示所有算子都通过 CubeK：

- 大量逐元素、索引和 mask 操作直接使用 burn-cubecl 中的 CubeCL Kernel；
- 部分 direct convolution 和 transpose 路径由 burn-cubecl 自己实现；
- deformable convolution 会组合自定义 Kernel 与矩阵乘；
- attention 带 bias、softcap 或自定义 scale 时会走 fallback；
- CubeK 仓库中的 resample 没有被该 Burn 快照引用。

CubeK attention 目录中存在 backward 实现代码，但文档与测试仍带有未完成
标记，Burn 侧也没有调用路径。因此本书只把 FlashAttention forward 作为
已核验集成，不宣称 backward 已接入 Burn。

## 4. 为什么必须保留 fallback

fallback 不是失败的同义词，而是可移植系统的一部分。它可能在以下情况被
选择：

- Runtime 不支持候选 Kernel 所需特性；
- dtype、布局、head dimension 或 shape 不满足算法约束；
- 可选功能无法由 fused Kernel 表达；
- 对当前小输入，简单实现实测更快；
- 高性能实现返回配置不可用错误。

fallback 往往分配更多中间 Tensor 或执行更多 Kernel，因此正确性相同不
代表成本相同。第 7 章讨论服务延迟时会再次遇到这个差别。

## 5. 用户何时直接使用 CubeK

通常的模型开发者应从 Burn Tensor API 开始。直接调用 CubeK launch API
适合算子开发、Backend 集成和研究特定策略，需要自行承担 binding、layout、
device feature 与 unsafe 合约。第三章实验直接使用 CubeCL 是为了看清
Kernel 边界；它不把低层 API 建议为日常模型代码。

