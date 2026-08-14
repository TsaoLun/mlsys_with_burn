//! 数据并行、张量并行、流水线与 ZeRO 的**整数成本模型**。
//!
//! 与第 9 章集群模拟器同类：把教材公式变成可断言的字节数与时间槽，
//! 不代表 NCCL、Megatron 或 FSDP 的真实 runtime。时间单位是「一步」
//! 或「一个 α」，带宽项按字节计；不做墙钟测量。

use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// 成本模型输入不满足整除或规模假设。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostError {
    /// AllReduce / AllGather 至少需要 2 个参与者。
    WorldTooSmall,
    /// 字节数无法被 world size 整除，整数公式会丢精度。
    NotDivisible { value: u64, divisor: u32 },
    /// stage 或 micro-batch 数必须为正。
    EmptyPipeline,
    /// ZeRO 分片需要至少一个设备。
    EmptyWorld,
}

impl Display for CostError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorldTooSmall => formatter.write_str("集合通信至少需要 2 个参与者"),
            Self::NotDivisible { value, divisor } => {
                write!(formatter, "{value} 不能被 world size {divisor} 整除")
            }
            Self::EmptyPipeline => formatter.write_str("流水线的 stage 与 micro-batch 数必须为正"),
            Self::EmptyWorld => formatter.write_str("设备数必须为正"),
        }
    }
}

impl Error for CostError {}

fn require_divisible(value: u64, divisor: u32) -> Result<u64, CostError> {
    let world = u64::from(divisor);
    if value % world != 0 {
        return Err(CostError::NotDivisible { value, divisor });
    }
    Ok(value / world)
}

// ANCHOR: ring
/// 环形 AllReduce 的每设备成本：两阶段 scatter-reduce + allgather。
///
/// 每阶段 \(p-1\) 步、每步传送 \(S/p\) 字节，因此
///
/// - α 步数 \(= 2(p-1)\)
/// - 每设备发送字节 \(= 2(p-1)S/p\)，大 \(p\) 时趋近 \(2S\)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingAllReduceCost {
    pub world_size: u32,
    pub payload: u64,
    pub alpha_steps: u64,
    pub bytes_sent: u64,
}

impl RingAllReduceCost {
    /// 相对「每设备 2S」这条常用近似的分子/分母：恰好是 \((p-1)/p\)。
    pub fn two_s_ratio(&self) -> (u64, u64) {
        let p = u64::from(self.world_size);
        (p - 1, p)
    }
}

/// 环形 AllReduce：`payload` 必须能被 `world_size` 整除。
pub fn ring_allreduce(payload: u64, world_size: u32) -> Result<RingAllReduceCost, CostError> {
    if world_size < 2 {
        return Err(CostError::WorldTooSmall);
    }
    let p = u64::from(world_size);
    let bytes_sent = 2 * (p - 1) * require_divisible(payload, world_size)?;
    Ok(RingAllReduceCost {
        world_size,
        payload,
        alpha_steps: 2 * (p - 1),
        bytes_sent,
    })
}

/// \(\alpha + \beta\) 时间：`alpha` 是一步启动开销，`beta_per_byte` 是每字节传送时间。
pub fn ring_allreduce_time(
    payload: u64,
    world_size: u32,
    alpha: u64,
    beta_per_byte: u64,
) -> Result<u64, CostError> {
    let cost = ring_allreduce(payload, world_size)?;
    Ok(cost.alpha_steps.saturating_mul(alpha) + cost.bytes_sent.saturating_mul(beta_per_byte))
}
// ANCHOR_END: ring

// ANCHOR: pipeline
/// 流水线虚拟时间：每个 forward 或 backward 槽占用 1 个时间单位。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineBubble {
    pub stages: u32,
    pub microbatches: u32,
    /// 从第一个槽到最后一个槽的跨度。
    pub span: u64,
    /// 真正在做计算的槽数（所有 stage × 每个 micro-batch 的工作）。
    pub busy_slots: u64,
    /// 空闲比例的分子：\(p-1\)。
    pub idle_numerator: u64,
    /// 空闲比例的分母：GPipe 为 \(m+p-1\)，1F1B 为 \(2m+p-1\)。
    pub idle_denominator: u64,
}

impl PipelineBubble {
    /// 交叉相乘比较空闲比例，避免浮点。`true` 表示 `self` 空闲比例严格更大。
    pub fn idle_worse_than(self, other: Self) -> bool {
        self.idle_numerator * other.idle_denominator > other.idle_numerator * self.idle_denominator
    }
}

fn pipeline_parts(stages: u32, microbatches: u32) -> Result<(u64, u64), CostError> {
    if stages == 0 || microbatches == 0 {
        return Err(CostError::EmptyPipeline);
    }
    Ok((u64::from(stages), u64::from(microbatches)))
}

/// GPipe 式 flush：只跑前向（或把一整段 F+B 看成一个槽）时，
/// 跨度 \(m+p-1\)，空闲比例 \((p-1)/(m+p-1)\)。
pub fn gpipe_flush(stages: u32, microbatches: u32) -> Result<PipelineBubble, CostError> {
    let (p, m) = pipeline_parts(stages, microbatches)?;
    Ok(PipelineBubble {
        stages,
        microbatches,
        span: m + p - 1,
        busy_slots: m * p,
        idle_numerator: p - 1,
        idle_denominator: m + p - 1,
    })
}

/// 简化 1F1B：每个 micro-batch 占用一个 F 槽和一个 B 槽，
/// 跨度 \(2m+p-1\)，空闲比例 \((p-1)/(2m+p-1)\)。
///
/// 真实 Megatron 1F1B 还要处理 F/B 耗时不等、激活保存与重计算；
/// 这里只保留「交错之后 warm-up/drain 仍在」这一条时序事实。
pub fn one_f_one_b(stages: u32, microbatches: u32) -> Result<PipelineBubble, CostError> {
    let (p, m) = pipeline_parts(stages, microbatches)?;
    Ok(PipelineBubble {
        stages,
        microbatches,
        span: 2 * m + p - 1,
        busy_slots: 2 * m * p,
        idle_numerator: p - 1,
        idle_denominator: 2 * m + p - 1,
    })
}
// ANCHOR_END: pipeline

// ANCHOR: zero
/// ZeRO 切到哪一级：切开的对象不同，每卡留下的字节就不同。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroStage {
    /// 参数、梯度、优化器状态全部复制。
    Replicated,
    /// 只分片优化器状态（ZeRO-1）。
    Optimizer,
    /// 再分片梯度（ZeRO-2）。
    Gradients,
    /// 参数也分片（ZeRO-3 / FSDP）。
    Parameters,
}

/// 一张卡上的参数 / 梯度 / 优化器状态字节。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroMemory {
    pub params: u64,
    pub grads: u64,
    pub optimizer: u64,
}

impl ZeroMemory {
    pub fn total(self) -> u64 {
        self.params + self.grads + self.optimizer
    }
}

/// 每卡显存。`params`/`grads`/`optimizer` 在对应 stage 必须能被 `world_size` 整除。
pub fn zero_per_device(
    params: u64,
    grads: u64,
    optimizer: u64,
    world_size: u32,
    stage: ZeroStage,
) -> Result<ZeroMemory, CostError> {
    if world_size == 0 {
        return Err(CostError::EmptyWorld);
    }
    let shard_params = matches!(stage, ZeroStage::Parameters);
    let shard_grads = matches!(stage, ZeroStage::Gradients | ZeroStage::Parameters);
    let shard_opt = !matches!(stage, ZeroStage::Replicated);
    Ok(ZeroMemory {
        params: if shard_params {
            require_divisible(params, world_size)?
        } else {
            params
        },
        grads: if shard_grads {
            require_divisible(grads, world_size)?
        } else {
            grads
        },
        optimizer: if shard_opt {
            require_divisible(optimizer, world_size)?
        } else {
            optimizer
        },
    })
}

/// 张量并行一层 AllGather：每设备发送 \((p-1)S/p\) 字节。
pub fn tensor_parallel_allgather(activation: u64, world_size: u32) -> Result<u64, CostError> {
    if world_size < 2 {
        return Err(CostError::WorldTooSmall);
    }
    let p = u64::from(world_size);
    Ok((p - 1) * require_divisible(activation, world_size)?)
}
// ANCHOR_END: zero

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_bytes_match_textbook_formula() {
        let cost = ring_allreduce(1024, 8).expect("1024 可被 8 整除");
        assert_eq!(cost.alpha_steps, 14);
        assert_eq!(cost.bytes_sent, 2 * 7 * 1024 / 8);
        assert_eq!(cost.two_s_ratio(), (7, 8));
    }

    #[test]
    fn ring_bytes_approach_two_s_as_world_grows() {
        let payload = 1024;
        let small = ring_allreduce(payload, 4).expect("可整除");
        let large = ring_allreduce(payload, 32).expect("可整除");
        assert_eq!(small.bytes_sent, 1536);
        assert_eq!(large.bytes_sent, 1984);
        assert!(large.bytes_sent > small.bytes_sent);
        assert!(large.bytes_sent < 2 * payload);
        assert_eq!(large.bytes_sent, 2 * payload * 31 / 32);
    }

    #[test]
    fn ring_rejects_indivisible_payload() {
        assert_eq!(
            ring_allreduce(1000, 8),
            Err(CostError::NotDivisible {
                value: 1000,
                divisor: 8
            })
        );
        assert_eq!(ring_allreduce(1024, 1), Err(CostError::WorldTooSmall));
    }

    #[test]
    fn ring_time_is_alpha_steps_plus_bytes_times_beta() {
        let time = ring_allreduce_time(1024, 8, 10, 2).expect("可整除");
        assert_eq!(time, 14 * 10 + 1792 * 2);
    }

    #[test]
    fn gpipe_idle_matches_formula() {
        let tight = gpipe_flush(3, 3).expect("正数");
        assert_eq!(tight.span, 5);
        assert_eq!(tight.busy_slots, 9);
        assert_eq!((tight.idle_numerator, tight.idle_denominator), (2, 5));

        let wide = gpipe_flush(3, 16).expect("正数");
        assert_eq!((wide.idle_numerator, wide.idle_denominator), (2, 18));
        assert!(tight.idle_worse_than(wide));
    }

    #[test]
    fn one_f_one_b_has_smaller_idle_than_gpipe() {
        let gpipe = gpipe_flush(4, 8).expect("正数");
        let interleaved = one_f_one_b(4, 8).expect("正数");
        assert_eq!(interleaved.span, 16 + 4 - 1);
        assert_eq!(interleaved.busy_slots, 2 * 8 * 4);
        assert_eq!(
            (interleaved.idle_numerator, interleaved.idle_denominator),
            (3, 19)
        );
        assert!(gpipe.idle_worse_than(interleaved));
    }

    #[test]
    fn pipeline_rejects_empty() {
        assert_eq!(gpipe_flush(0, 4), Err(CostError::EmptyPipeline));
        assert_eq!(one_f_one_b(4, 0), Err(CostError::EmptyPipeline));
    }

    #[test]
    fn zero_stages_shard_the_right_terms() {
        let params = 16;
        let grads = 16;
        let optimizer = 32;
        let n = 8;
        let z0 = zero_per_device(params, grads, optimizer, n, ZeroStage::Replicated)
            .expect("复制不要求整除到 1/n 以外");
        let z1 =
            zero_per_device(params, grads, optimizer, n, ZeroStage::Optimizer).expect("可整除");
        let z2 =
            zero_per_device(params, grads, optimizer, n, ZeroStage::Gradients).expect("可整除");
        let z3 =
            zero_per_device(params, grads, optimizer, n, ZeroStage::Parameters).expect("可整除");
        assert_eq!(z0.total(), 64);
        assert_eq!(
            z1,
            ZeroMemory {
                params: 16,
                grads: 16,
                optimizer: 4
            }
        );
        assert_eq!(
            z2,
            ZeroMemory {
                params: 16,
                grads: 2,
                optimizer: 4
            }
        );
        assert_eq!(
            z3,
            ZeroMemory {
                params: 2,
                grads: 2,
                optimizer: 4
            }
        );
        assert_eq!(z3.total(), (params + grads + optimizer) / u64::from(n));
    }

    #[test]
    fn tensor_parallel_allgather_is_one_phase() {
        assert_eq!(tensor_parallel_allgather(1024, 8).expect("可整除"), 896);
        assert_eq!(
            tensor_parallel_allgather(1024, 1),
            Err(CostError::WorldTooSmall)
        );
    }
}
