# GPU 并行与存储模型

本节先使用抽象机器模型，再给出 CubeCL 和 CUDA 术语对照。不同厂商的精确
调度、缓存和矩阵单元并不相同，写 Kernel 时应以 Runtime 暴露的能力为准。

## 1. Host 与 Device

Host 程序分配 buffer、选择 launch 拓扑并提交 Kernel；Device 执行大量并行
工作。提交通常是异步的，只有读回、显式同步或依赖约束才要求 host 等待。
因此性能测量必须覆盖正确的同步边界，不能只计入“提交命令”的时间。

## 2. Cube、Unit 与 Plane

CubeCL 将一次 launch 描述为多个 cube，每个 cube 内含三维排列的 unit：

```text
CubeCount(x, y, z)
└─ 每个 Cube
   └─ CubeDim(x, y, z) 个 Unit
      └─ 若干相邻 Unit 可组成 Plane
```

粗略对照如下：

| CubeCL | CUDA 常用术语 | 含义 |
|---|---|---|
| CubeCount | grid 维度 | 本次 launch 有多少个工作组 |
| Cube | block | 可以协作并使用共享资源的一组 unit |
| CubeDim | block 维度 | 每个 cube 内 unit 的三维形状 |
| Unit | thread | 执行 Kernel 实例的逻辑工作项 |
| Plane | warp | 可执行协同操作的一组 unit |

这只是认知映射，不是 ABI 等价关系。Plane 大小不是语言层固定常数，CPU
Runtime 的 plane 能力也不同于真实 GPU。Kernel 不应凭习惯写死 32。

CubeCL 提供多种拓扑内建量：

- `UNIT_POS_X/Y/Z`：unit 在 cube 内的位置；
- `CUBE_POS_X/Y/Z`：cube 在 launch 中的位置；
- `ABSOLUTE_POS`：扁平化后的全局线性 unit 索引；
- `ABSOLUTE_POS_X/Y/Z`：全局三维位置。

元素级 Kernel 常用 `ABSOLUTE_POS`，矩阵和图像 Kernel 则更适合二维或三维
坐标。launch 的 unit 数可能大于元素数，所以索引前仍要处理边界。

## 3. 存储层次

可用下面的简化层次理解数据移动：

| 层次 | 可见范围 | 典型特点 |
|---|---|---|
| unit 私有值/寄存器 | 单个 unit | 容量最小、延迟低，过多会降低并发 |
| cube 共享内存 | 同一 cube | 显式协作和同步，可复用 tile |
| device 全局内存 | 整个 device | 容量大、延迟高，连续访问更易利用带宽 |
| host 内存/外部存储 | host 或系统 | 跨总线或 I/O，代价通常更高 |

缓存可能自动缓解一部分访问，但 Kernel 优化不能假设所有数据都“恰好在
cache”。常见策略是从全局内存批量加载连续 tile，在 cube 内复用，然后把
结果合并写回。

## 4. 合并、向量化与同步

相邻 unit 访问相邻地址，更容易形成合并访存。CubeCL 的
`Vector<F, N>` 表示连续元素，可让 Runtime 在合适时使用 SIMD；它不保证
任意 `N` 或任意地址都高效。

共享内存带来新的正确性条件：

1. 所有参与者先完成写入；
2. 在需要的位置执行 cube 级同步；
3. 同步后再读取其他 unit 写入的数据；
4. 避免多个 unit 无序写同一位置的数据竞争。

同步也有成本。过细的 tile 增加同步次数，过大的 tile 又可能耗尽寄存器或
共享内存。这个折中不能只由源代码表面决定。

## 5. 矩阵单元不是通用乘法器

现代 GPU 常提供矩阵乘加专用单元。CubeCL 用 CMMA 等抽象暴露相关能力，
CubeK 则提供使用这些能力的算法变体。可用性取决于 Runtime、dtype、tile
shape 和设备 feature；不支持时必须过滤策略或回退。

OpenMLSys 以 Volta Tensor Core 和 Ascend Cube 为案例解释这一思想。它们
仍有历史教学价值，但具体尺寸、吞吐与内存数字不能代表今天所有设备。本书
关注“矩阵指令对输入布局和 tile 有约束”这一长期原则。

