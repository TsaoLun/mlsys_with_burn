# 2026-08-13 深度批次五：CubeK 四层下钻与 autotune 键

## 背景

延续「机制彻底搞懂」定位（批次三/四），交付 STATUS 记录的两项
机制纵深：matmul 在 CubeK 内部的组件层次、autotune 特化键的端到端
构成。纯源码走读 + 正文深化，无新代码、无范围决策。

## 交付

1. **ch03/04「第七层：Routine 内部的四层组件」**（续接既有六决策点
   走查）：`cubek-matmul/src/components/` 的 batch/global/stage/tile
   四层职责表（batch 处理广播的多个独立 matmul；global「把块装进
   共享内存完成整条归约，负责搬运/边界/plane 特化」；stage 的
   `PartitionedStageMatmul` 分区给 plane；tile 为最小乘法单元），
   均按 pinned 模块注释意译。tile 五变体
   （Cmma/Mma/Register/PlaneVec/Interleaved）与
   `requires_accelerator()`；第 5 节优化阶梯映射为显式类型
   （1–2 级 = global 装载 + Register，5 级 = Cmma/Mma）；GEMM 阶梯
   实验的手写 kernel 定位为「global+Register 压平形态」；
   `routines/`（naive/gemm/cmma/gemv_unit_perpendicular/cpu_gemm/
   batch/selector）收尾整条链。
2. **ch03/06「一个 tune key 长什么样」**：`MatmulAutotuneKey =
   ProblemDefinition + Analysis` 摘录，逐字段机制来源——
   - `#[autotune(anchor)]`：cubecl `tune/util.rs` 的 anchor 把尺寸
     **向上取整到底数的幂**；autotune level 缩放底数（0→2.5 粗桶、
     1→2、3→精确匹配），控制键基数与首调成本；
   - stride/pow2 因子封顶 `2^10`：源码注释明示 128 字节 swizzle
     重复周期是最后影响性能的对齐档位；
   - analysis 的 512/2048 显式分桶（Small/Medium/Large）+
     `should_tune_double_buffering` 按键剪枝候选集合；
   - 存储侧核实 `LocalTuner<AK, ID>` 按设备 ID 各持一个 Tuner，
     Tuner 内按键缓存——换设备必重测、换 shape 跨桶才重测。
   结论：键设计 = 性能等价类的工程近似，为编译/测量次数设上界；
   与「level 不等于越高越快」互为机制根源。
3. 互链：ch03/05 矩阵指令段、ch03/07 阶梯实验段各一句指向第七层。

## 验证

- 全部断言以 `git show/grep <pin>`（cubek `f82a6d0`、cubecl
  `be278a1`）核实，含 anchor 实现、tile 枚举、tune_key 字段、
  LocalTuner 结构；
- `mdbook build/test book`、`check_release.py --require-built-book
  --json`（`ok=true`、`errors=[]`、`warnings=[]`）、
  `git diff --check` 通过；无 Rust 代码改动。

## 边界

- 四层职责为模块注释意译，标注来源；不宣称已在真机对比五种 tile
  变体的性能；blueprint 参数细节未展开（属 Strategy 枚举既有叙述）。

## 后续候选

- `#[derive(CubeType)]` 展开对照（宏黑箱的收尾件）；
- attention/reduce 在同一四层框架下的对照走读；
- Fusion on-the-fly 生成的 kernel 与 CubeK 预制 kernel 的选择边界。
