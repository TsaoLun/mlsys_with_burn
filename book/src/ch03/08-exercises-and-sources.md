# 练习、延伸阅读与来源

## 小结

加速器性能来自并行执行与数据复用，而不是单一峰值数字。GPU 的 cube/unit/
plane 拓扑和寄存器/共享/全局存储层次共同约束 Kernel。CubeCL 用 Rust
风格 DSL、IR 与 Runtime 表达这套模型；raw buffer 和 unchecked launch
仍有必须显式证明的 unsafe 边界。

CubeK 以 Blueprint–Routine 架构和 Autotuner 组织算子，并在具体 crate
中提供 Strategy 与 Launch 接口。burn-cubecl 把它与纯 CubeCL Kernel、
fallback 和 LocalTuner 组合起来。
高性能策略受 shape、dtype 与设备能力约束，不是所有 Burn 算子都经过
CubeK。

## 练习


练习按难度标注为【基础】【进阶】【挑战】。折叠「提示」只给出方向
（正文小节、示例 crate 或书中给出的源码路径），不提供完整答案。
【挑战】题往往需要额外硬件、外部数据或自行设计，本书默认示例不覆盖。

### 概念题

1. 【基础】为什么峰值 FLOP/s 不能预测逐元素加法的性能？

<details>
<summary>提示</summary>

仿照[「工作负载与加速器设计」](01-workloads-and-design.md)对
`scale_kernel` 的算术强度演算，数一数逐元素加法每搬运一字节能做
几次运算，再想想这个强度落在 roofline 拐点的哪一侧：在带宽屋顶
之下，更高的峰值算力帮不上忙。

</details>

2. 【基础】比较 Cube、Unit 和 Plane 的协作范围。为什么 Plane 宽度不应写死？

<details>
<summary>提示</summary>

对照[「GPU 并行与存储模型」](02-gpu-machine-model.md)的层次配图与
CubeCL–CUDA 术语对照表，分别说清三者内部能共享什么、在哪一级同步；
再想想该表为何强调「认知映射而非 ABI」，CPU Runtime 的 plane 能力
与真实 GPU 的差别对「写死 32」意味着什么。

</details>

3. 【基础】Tiling 如何减少全局内存读取？它又会增加哪些资源与同步成本？

<details>
<summary>提示</summary>

收益侧对照[「GEMM 与优化阶梯」](05-gemm-optimization.md)的加载计数
推导与 `examples/ch03-tile-loads` 打印的 8192 对 1024；成本侧对照
[「GPU 并行与存储模型」](02-gpu-machine-model.md)的共享内存正确性
条件，想想 tile 增大时共享内存、寄存器与可驻留 cube 数如何变化。

</details>

4. 【基础】为什么 checked launch 不能证明 raw BufferArg 的长度真实？

<details>
<summary>提示</summary>

[「CubeCL 编程模型」](03-cubecl-programming.md)「边界检查与安全责任」
指出 `BufferArg::from_raw_parts` 是 `unsafe fn`；再读
`examples/ch03-cubecl-kernel` 中 `run_scale` 的 SAFETY 注释，想想
checked 模式做检查时依据的长度从哪里来、Runtime 有没有办法反向核实
这个声明。

</details>

5. 【进阶】fallback 在可移植算子库中承担什么职责？列出三种触发原因。

<details>
<summary>提示</summary>

[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)「为什么必须保留
fallback」列出了五种触发情形，挑三种并说明各自在哪一层被检测出来；
谈职责时留意正文的提醒：fallback 正确性相同不代表成本相同。

</details>

6. 【进阶】为什么 autotune 结果不能直接复制到另一台机器？

<details>
<summary>提示</summary>

[「算子编译、调优与生态」](06-compilation-and-tuning.md)把 autotune
定义为「在当前设备上测量候选并按 tune key 缓存」；顺着该节候选过滤、
测量、缓存的流程走一遍，想想候选集合、测量数值与缓存键各自绑定了
哪些只属于这台机器的事实。

</details>

7. 【进阶】对 `16×16×16` GEMM 手算朴素和 `8×8×8` tiled 的加载次数、FLOP
   数和简化算术强度；说明这个结果为什么还不是 Roofline 性能证明。

<details>
<summary>提示</summary>

按[「GEMM 与优化阶梯」](05-gemm-optimization.md)「Tiling 与共享内存」
的加载计数推导代入数字，并与 `examples/ch03-tile-loads` 测试注释里
的手算核对；第二问看同页「先用 Roofline 判断优化方向」对元素版强度
的限定：想想它没计入哪些真实成本、式子里有没有任何实测量。

</details>

8. 【进阶】为 naive → tile → thread tile → double buffer → matrix instruction
   的每一步写出新增资源、同步不变量和可能的回退条件。

<details>
<summary>提示</summary>

沿[「GEMM 与优化阶梯」](05-gemm-optimization.md)第 1–5 节逐级整理：
每级新增的存储形态（共享 tile、寄存器累加、双份 stage、矩阵
fragment）都带来新的同步或填充规则，双缓冲一节已列出四条流水线
正确性条件；回退条件可借
[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)的策略过滤视角
反推：设备缺哪种能力时这一级不可用。

</details>


### 已交付实验题

1. 【基础】修改 `scale_kernel` 为 `output[i] = input[i] * scale + bias`，先扩展
   host reference，再扩展 Kernel 和测试。

<details>
<summary>提示</summary>

先在 `examples/ch03-cubecl-kernel/src/lib.rs` 里改 `scale_reference`
把语义定下来，再动 Kernel 与测试；顺序依据是
[「CubeCL 编程模型」](03-cubecl-programming.md)「从正确 Kernel 到
高性能 Kernel」。bias 若也用 `#[comptime]`，会遇到实验节讲过的
特化键可哈希约束——先想清楚它该是编译期还是运行时参数。

</details>

2. 【基础】把输入长度改为 5，但将 CubeDim 向上取整到 8，验证 guard 仍使输出正确。

<details>
<summary>提示</summary>

`run_scale` 用 `div_ceil` 计算 cube 数，本就允许 unit 数超过元素数；
CubeDim 固定为 8 后，想想编号 5–7 的 unit 拿到的 `ABSOLUTE_POS`
是什么、Kernel 里哪一行挡住了它们，再用与 `scale_reference` 的对比
测试验证。背景是[「GPU 并行与存储模型」](02-gpu-machine-model.md)
说的「launch 的 unit 数可能大于元素数」。

</details>

3. 【进阶】为 `ch03-tile-loads` 增加另一组可整除尺寸并手算 `tiled_loads`。

<details>
<summary>提示</summary>

`tile_load_counts` 把 tiled 加载拆成 tile 网格数、K 方向 stage 数、
每 stage 加载量三个因子，现有测试注释里有 16/8 情形的完整手算可以
仿照；新尺寸必须通过整除检查（见 `rejects_non_divisible_tiles`），
推导在[「GEMM 与优化阶梯」](05-gemm-optimization.md)「Tiling 与
共享内存」。

</details>

4. 【进阶】在支持的机器上运行 `cargo test -p ch03-cubecl-kernel --features wgpu --locked`。

<details>
<summary>提示</summary>

前提、命令与预期输出都在
[「实验：CPU 上运行 CubeCL Kernel」](07-cpu-kernel-lab.md)「可选
GPU 路径」：需要 Metal/Vulkan/DX12 等可用 adapter，对应测试是
`wgpu_kernel_matches_reference_when_requested`。通过只说明同一
Kernel 在两类 Runtime 上与 host reference 一致，不含带宽或占用率
结论。

</details>


### 扩展 Kernel 题（未随章交付）

1. 【进阶】将 `scale` 从 comptime 整数改为运行时标量。记录生成 launch API 的差异。

<details>
<summary>提示</summary>

先对比 `scale_kernel` 签名里 `#[comptime] scale: u32` 与普通参数在
launch 调用处的传法；
[「实验：CPU 上运行 CubeCL Kernel」](07-cpu-kernel-lab.md)第 1、6 节
解释了 scale 为何取 `u32`、修改它为何产生新特化。改为运行时标量后
`f32` 不再受特化键约束，宏生成的 launch 入口如何接收这个参数、
重复 launch 是否仍触发新编译，正是要记录的差异。

</details>

2. 【挑战】为 raw buffer 长度不变量写一个 safe host helper，使 unsafe block 更小。

<details>
<summary>提示</summary>

`run_scale` 的 SAFETY 注释给出要证明的事实：两个 BufferArg 描述的
分配恰好容纳 `input.len()` 个 `f32`，Kernel 在索引前有 guard。让
helper 的签名把「handle 与长度出自同一个 slice」变成类型保证，
调用者便无法分别传入不匹配的 handle 与长度；对照
[「CubeCL 编程模型」](03-cubecl-programming.md)「边界检查与安全责任」
检查两项不变量是否都被覆盖。

</details>

3. 【挑战】增加 `Vector<f32, N>` 版本；验证长度不是 N 倍数时的尾部处理。

<details>
<summary>提示</summary>

对照 `examples/ch03-cubecl-kernel` 的边界 guard 与
[「CubeCL 编程模型」](03-cubecl-programming.md)「Slice、Vector 与
Tensor 参数」：向量宽度必须与 buffer 布局和元素总数一致；先决定
余数元素由谁处理，尾部访问不得越过 raw buffer 的长度不变量。

</details>

4. 【挑战】把 `ch03-gemm-ladder` 的阶梯延伸一级：把 tile 尺寸从 16 改为 8 或
   32 比较正确性与本机耗时，或让一个 unit 计算 2×2 输出（thread tile）。

<details>
<summary>提示</summary>

tiled Kernel 与计时协议的代码在
[「实验：CPU 上运行 CubeCL Kernel」](07-cpu-kernel-lab.md)第 8 节；
tile 尺寸受 max_units_per_cube 与共享内存容量约束（见
[「GPU 并行与存储模型」](02-gpu-machine-model.md)），thread tile 的
复用收益对应[「GEMM 与优化阶梯」](05-gemm-optimization.md)
「Thread tile 与向量化」的 $2mn/(m+n)$。任何改动先通过非整除形状的
正确性测试，再比较耗时。

</details>


### 源码题

1. 【进阶】在 CubeCL 中找到 `ABSOLUTE_POS` 的定义，解释它如何由 cube 和 unit
   坐标得到。

<details>
<summary>提示</summary>

定义在 `cubecl/crates/cubecl-core/src/frontend/topology.rs`；把
[「GPU 并行与存储模型」](02-gpu-machine-model.md)列出的
`CUBE_POS_X/Y/Z`、`UNIT_POS_X/Y/Z` 与 CubeDim 形状摆在一起，先猜
一个「三维坐标扁平化后再组合」的表达式，再回源码核对展开顺序。

</details>

2. 【进阶】找到 `BufferArg::from_raw_parts` 的 Safety 文档，列出调用者责任。

<details>
<summary>提示</summary>

Safety 文档随定义在
`cubecl/crates/cubecl-core/src/frontend/container/slice/launch.rs`；
逐条抄下调用者责任后，对照 `examples/ch03-cubecl-kernel` 中
`run_scale` 的 SAFETY 注释，看它对每条责任分别拿什么事实作答、
哪一条要靠 Kernel 里的 guard 兜底。

</details>

3. 【进阶】沿 Burn matmul 路径找到 `cubek::matmul::launch::launch_ref`，
   并说出六个决策点各自回答了什么问题。

<details>
<summary>提示</summary>

对照[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)的「逐层走查」
小节，按其中的文件路径在本书所用版本源码里各读一遍；每到一层，
先用一句话回答该层要解决的问题，再找出支撑这句话的函数或注释。

</details>

4. 【进阶】在 CubeK matmul 中比较 Naive、CpuGemm 与一个 CMMA Strategy 的可用条件。

<details>
<summary>提示</summary>

策略枚举在 `cubek/crates/cubek-matmul/src/strategy/strategy.rs`，
统一入口的转发在同 crate 的 `launch.rs`；比较时沿
[「GPU 并行与存储模型」](02-gpu-machine-model.md)「矩阵单元不是
通用乘法器」给的约束轴（Runtime、dtype、tile shape、设备 feature）
逐项过——正文提醒过，不能由 crate 名推断一定用上矩阵单元。

</details>

5. 【进阶】找到 attention 因 bias、softcap 或 scale 进入 fallback 的判断。

<details>
<summary>提示</summary>

从 `burn/crates/burn-cubecl/src/kernel/attention/` 入手，找策略
选择处对 bias、softcap 与自定义 scale 的条件分支；
[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)的「覆盖范围与
边界」「为什么必须保留 fallback」说明这类判断为何存在，以及
fallback 路径可能多付出的中间 Tensor 与 Kernel。

</details>

6. 【进阶】找出一个 Burn 直接实现的 CubeCL Kernel 和一个经 CubeK 实现的算子，
   比较两条路径。

<details>
<summary>提示</summary>

[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)「覆盖范围与边界」
点名逐元素、索引与 mask 类算子直接用 burn-cubecl 里的 CubeCL
Kernel；在 `burn/crates/burn-cubecl/src/kernel/` 下挑一个这样的
实现，与同目录 `matmul/` 经 CubeK 的路径对比策略选择、autotune 与
fallback 的有无。

</details>


### 性能实验题（仅概念/扩展，需自备设备）

1. 【进阶】为不同长度的 scale Kernel 分别报告首次运行与稳态运行时间，解释差异。

<details>
<summary>提示</summary>

差异来源对照
[「实验：CPU 上运行 CubeCL Kernel」](07-cpu-kernel-lab.md)「观察
编译边界」与[「算子编译、调优与生态」](06-compilation-and-tuning.md)
的编译与缓存讨论：首次运行含 IR 构建与 Runtime 编译，而长度经
CubeDim 参与编译配置，某些长度变化可能再次触发编译；计时要在真实
同步后停止，这类耗时观察也不能替代 GPU 带宽或吞吐结论。

</details>

2. 【挑战】在 GPU 上比较连续索引和跨距索引，记录有效带宽而不只记录耗时。

<details>
<summary>提示</summary>

推理工具在[「GPU 并行与存储模型」](02-gpu-machine-model.md)「合并、
向量化与同步」：步长增大时，同样多的请求散布到更宽的地址区间，
事务数随之增长；有效带宽应当用实际搬运的字节数除以同步后测得的
时间来算。本题需要额外 GPU 环境，记录设备与同步边界，勿外推为
默认 CPU 示例的性能结论。

</details>

3. 【挑战】为三个 GEMM shape 比较 Burn 默认策略和固定策略。报告 autotune 是否
   包含在计时内，不以单一 shape 宣布普遍胜负。

以上三题超出本章默认示例范围；若自行测量，须写明设备、同步边界与是否含编译。

<details>
<summary>提示</summary>

默认策略在哪一层注入，看
[「CubeK 与 Burn 算子路径」](04-cubek-and-burn.md)走查的第 2 步
（`burn/crates/burn-cubecl/src/ops/tensor.rs` 的 `float_matmul`）；
报告体例按[「算子编译、调优与生态」](06-compilation-and-tuning.md)
「测试、Benchmark 与性能声明」的五条来，尤其写明 autotune 搜索是否
计入首次计时、结论覆盖哪些 shape。

</details>


## 延伸阅读

Roofline、CUDA 与 tile 编译器的论文见附录
[参考文献](../references.md#第-3-章-ai-加速器与编程)。
本书所用版本源码中的权威入口：

- `cubecl/README.md`
- `cubecl/examples/gelu/`
- `cubecl/crates/cubecl-core/src/frontend/topology.rs`
- `cubecl/crates/cubecl-core/src/frontend/container/slice/launch.rs`
- `cubecl/crates/cubecl-runtime/src/runtime.rs`
- `cubecl/crates/cubecl-runtime/src/config/autotune.rs`
- `cubek/GUIDE.md`
- `cubek/crates/cubek-matmul/src/`
- `cubek/crates/cubek-matmul/src/launch.rs`（统一入口到 Strategy 的转发）
- `cubek/crates/cubek-matmul/src/strategy/strategy.rs`（策略空间枚举）
- `cubek/crates/cubek-reduce/src/`
- `burn/crates/burn-tensor/src/tensor/api/numeric.rs`（`Tensor::matmul` 的校验与 vec-mat 重解释）
- `burn/crates/burn-cubecl/src/ops/tensor.rs`（`float_matmul` 的策略选择）
- `burn/crates/burn-cubecl/src/kernel/matmul/`
- `burn/crates/burn-cubecl/src/kernel/attention/`
- `burn/crates/burn-cubecl/src/ops/module.rs`

CUDA、Triton、CUTLASS、TVM 与 MLIR 文档适合做生态对照；使用时应记录版本，
不能用最新在线签名替代本书所用版本的源码事实。

## 本章系统结论

1. 加速器收益来自并行度、算术强度与数据复用，而不是「有 GPU」四个字。
2. Cube/Unit/Plane 应对齐 GPU 的 block/thread/warp；存储层次决定搬运代价。
3. CubeCL 用同一套 Kernel IR 对接 CPU / WGPU / CUDA / HIP Runtime。
4. 实验对照 Kernel 与 host reference；有图形驱动时可看共享内存 tile 的相对差距。
5. 改 Kernel 或 matmul 策略时打开 CubeCL / CubeK，见[一次调用会经过哪些层](../crate-map.md)。
6. 正确性对照回答不了带宽排名或厂商 GEMM 谁更快。

## 来源与改编说明

OpenMLSys 文件对照与改编说明见[来源与改编总录](../appendix-sources.md#第-3-章)。
