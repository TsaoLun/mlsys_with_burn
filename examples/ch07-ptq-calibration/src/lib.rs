//! 训练后量化（PTQ）校准的可运行版本：int8 仿射量化、min-max 与
//! 百分位两种校准策略、per-tensor 与 per-channel 粒度，以及一个
//! i32 累加的 int8 矩阵乘。
//!
//! 全部纯 Rust 标量演算，验证的是第 7 章讲的**协议与误差结构**：
//! scale/zero-point 怎么来、离群值怎样毁掉 min-max、粒度为什么值得
//! 花元数据。它不代表任何低精度 backend 的执行路径或速度。

// ANCHOR: params
/// int8 仿射量化参数：`real ≈ (q - zero_point) * scale`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantParams {
    /// 由校准得到的实数区间 `[low, high]` 推出参数，并保证实数 0
    /// 恰好落在整数网格上（零填充/零跳过依赖这一点）。
    pub fn from_range(low: f32, high: f32) -> Self {
        // 区间必须覆盖 0，否则 0 无法精确表示。
        let low = low.min(0.0);
        let high = high.max(0.0);
        let scale = ((high - low) / 255.0).max(f32::MIN_POSITIVE);
        let zero_point = (-128.0 - low / scale).round() as i32;
        Self { scale, zero_point }
    }

    pub fn quantize(&self, value: f32) -> i8 {
        let q = (value / self.scale).round() as i32 + self.zero_point;
        q.clamp(-128, 127) as i8
    }

    pub fn dequantize(&self, q: i8) -> f32 {
        (q as i32 - self.zero_point) as f32 * self.scale
    }
}
// ANCHOR_END: params

// ANCHOR: calibration
/// min-max 校准：区间取观测到的最小/最大值，不丢任何点，但离群值
/// 会把 scale 拉大，牺牲所有正常值的分辨率。
pub fn minmax_params(data: &[f32]) -> QuantParams {
    let low = data.iter().copied().fold(f32::INFINITY, f32::min);
    let high = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    QuantParams::from_range(low, high)
}

/// 百分位校准：把区间夹在 `[p, 1-p]` 分位数，牺牲离群值的精确表示，
/// 换取主体分布的分辨率。
pub fn percentile_params(data: &[f32], tail: f32) -> QuantParams {
    assert!((0.0..0.5).contains(&tail), "tail 应位于 [0, 0.5)");
    let mut sorted = data.to_vec();
    sorted.sort_by(f32::total_cmp);
    let index = |fraction: f32| -> usize {
        let position = fraction * (sorted.len() - 1) as f32;
        position.round() as usize
    };
    QuantParams::from_range(sorted[index(tail)], sorted[index(1.0 - tail)])
}
// ANCHOR_END: calibration

/// 量化再反量化后的重建值。
pub fn reconstruct(data: &[f32], params: QuantParams) -> Vec<f32> {
    data.iter()
        .map(|&value| params.dequantize(params.quantize(value)))
        .collect()
}

/// 均方误差：校准策略比较的统一指标。
pub fn mse(reference: &[f32], candidate: &[f32]) -> f64 {
    assert_eq!(reference.len(), candidate.len());
    reference
        .iter()
        .zip(candidate)
        .map(|(&r, &c)| {
            let diff = (r - c) as f64;
            diff * diff
        })
        .sum::<f64>()
        / reference.len() as f64
}

/// 行主序矩阵按行（输出通道）分别校准量化，再整体重建。
pub fn reconstruct_per_channel(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    let mut result = Vec::with_capacity(weights.len());
    for row in 0..rows {
        let slice = &weights[row * cols..(row + 1) * cols];
        result.extend(reconstruct(slice, minmax_params(slice)));
    }
    result
}

// ANCHOR: int8_matmul
/// int8 × int8 → i32 累加 → 反量化的整数矩阵乘（per-tensor 参数）。
/// 与第 7 章正文的公式一致：先在整数域累加，最后一次性回到实数域。
pub fn int8_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    a_params: QuantParams,
    b_params: QuantParams,
) -> Vec<f32> {
    let qa: Vec<i8> = a.iter().map(|&v| a_params.quantize(v)).collect();
    let qb: Vec<i8> = b.iter().map(|&v| b_params.quantize(v)).collect();
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for i in 0..k {
                let qa_centered = qa[row * k + i] as i32 - a_params.zero_point;
                let qb_centered = qb[i * n + col] as i32 - b_params.zero_point;
                acc += qa_centered * qb_centered;
            }
            out[row * n + col] = acc as f32 * a_params.scale * b_params.scale;
        }
    }
    out
}
// ANCHOR_END: int8_matmul

/// f32 参考矩阵乘，量化误差以它为基准。
pub fn matmul_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * b[i * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// 确定性「近正态」样本：12 个均匀数求和（Irwin–Hall 近似），均值 0。
pub fn synthetic_activations(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..len)
        .map(|_| (0..12).map(|_| uniform()).sum::<f32>() - 6.0)
        .collect()
}

/// 离群值的确定性位置，供注入与「主体/离群」分开计误差使用。
pub fn outlier_positions(len: usize, count: usize) -> Vec<usize> {
    let stride = len / count.max(1);
    (0..count).map(|index| index * stride).collect()
}

/// 向样本注入固定数量的大离群值（模拟个别激活通道的尖峰）。
pub fn inject_outliers(data: &mut [f32], count: usize, magnitude: f32) {
    for (index, position) in outlier_positions(data.len(), count).into_iter().enumerate() {
        let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
        data[position] = sign * magnitude;
    }
}

/// 排除给定位置后的 MSE：把「主体误差」与「离群值误差」分开看。
pub fn mse_excluding(reference: &[f32], candidate: &[f32], excluded: &[usize]) -> f64 {
    assert_eq!(reference.len(), candidate.len());
    let mut skip = vec![false; reference.len()];
    for &position in excluded {
        skip[position] = true;
    }
    let mut sum = 0.0f64;
    let mut kept = 0usize;
    for index in 0..reference.len() {
        if skip[index] {
            continue;
        }
        let diff = (reference[index] - candidate[index]) as f64;
        sum += diff * diff;
        kept += 1;
    }
    sum / kept as f64
}

pub fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 在校准区间内，round-trip 误差不超过半个 scale——量化格点的
    /// 基本几何。
    #[test]
    fn round_trip_error_is_bounded_by_half_step() {
        let data = synthetic_activations(4096, 7);
        let params = minmax_params(&data);
        let rebuilt = reconstruct(&data, params);
        let bound = params.scale / 2.0 + 1e-6;
        for (&original, &restored) in data.iter().zip(&rebuilt) {
            assert!(
                (original - restored).abs() <= bound,
                "{original} 重建为 {restored}，超出半步界 {bound}"
            );
        }
    }

    /// 实数 0 必须精确表示：zero_point 的存在意义。
    #[test]
    fn zero_is_exactly_representable() {
        for seed in [1, 2, 3] {
            let mut data = synthetic_activations(1024, seed);
            data[0] = -3.0;
            let params = minmax_params(&data);
            assert_eq!(params.dequantize(params.quantize(0.0)), 0.0);
        }
    }

    /// 校准是一笔明码交易：99.5% 分位把**主体**分辨率提高一个数量级
    /// 以上（scale 缩小约 22 倍），代价是被裁剪的离群值贡献巨大误差，
    /// **整体** MSE 反而变差。哪边划算取决于任务是否需要离群值——
    /// 这就是「校准指标必须与任务对齐」。
    #[test]
    fn percentile_trades_outlier_error_for_body_resolution() {
        let mut data = synthetic_activations(8192, 11);
        let outliers = outlier_positions(data.len(), 82);
        inject_outliers(&mut data, 82, 80.0);

        let with_minmax = reconstruct(&data, minmax_params(&data));
        let with_clip = reconstruct(&data, percentile_params(&data, 0.005));

        let body_minmax = mse_excluding(&data, &with_minmax, &outliers);
        let body_clip = mse_excluding(&data, &with_clip, &outliers);
        let full_minmax = mse(&data, &with_minmax);
        let full_clip = mse(&data, &with_clip);

        // 主体：分位校准至少好一个数量级。
        assert!(
            body_clip < body_minmax / 10.0,
            "主体 MSE 应显著改善：clip={body_clip:.8} minmax={body_minmax:.8}"
        );
        // 整体：被裁剪的离群值把分位校准的 MSE 抬到 min-max 之上。
        assert!(
            full_clip > full_minmax,
            "含离群值的整体 MSE 应暴露裁剪代价：clip={full_clip:.6} minmax={full_minmax:.6}"
        );
    }

    /// 无离群值时 min-max 不吃亏：策略选择依赖分布，不存在恒优方案。
    #[test]
    fn minmax_is_fine_without_outliers() {
        let data = synthetic_activations(8192, 13);
        let minmax = mse(&data, &reconstruct(&data, minmax_params(&data)));
        let clipped = mse(&data, &reconstruct(&data, percentile_params(&data, 0.005)));
        assert!(minmax <= clipped * 1.5);
    }

    /// 两行动态范围差 100 倍的权重：per-tensor 的误差被宽行主导，
    /// 整体 MSE 几乎看不出差别，但**窄行**在 per-tensor 下分辨率
    /// 全毁——粒度收益要按通道看。
    #[test]
    fn per_channel_rescues_the_narrow_channel() {
        let cols = 512;
        let wide = synthetic_activations(cols, 17);
        let narrow: Vec<f32> = synthetic_activations(cols, 19)
            .into_iter()
            .map(|v| v * 0.01)
            .collect();
        let mut weights = wide;
        weights.extend(narrow.clone());

        let per_tensor = reconstruct(&weights, minmax_params(&weights));
        let per_channel = reconstruct_per_channel(&weights, 2, cols);

        let narrow_tensor = mse(&narrow, &per_tensor[cols..]);
        let narrow_channel = mse(&narrow, &per_channel[cols..]);
        assert!(
            narrow_channel < narrow_tensor / 1000.0,
            "窄行 MSE：per-channel={narrow_channel:.10} per-tensor={narrow_tensor:.10}"
        );

        // 宽行两种粒度同参数，误差一致；整体差距因此远小于窄行差距。
        let wide_tensor = mse(&weights[..cols], &per_tensor[..cols]);
        let wide_channel = mse(&weights[..cols], &per_channel[..cols]);
        assert!((wide_tensor - wide_channel).abs() < wide_tensor * 0.5);
    }

    /// int8 矩阵乘与 f32 参考的偏差有界，且确实有损（量化不是空转）。
    #[test]
    fn int8_matmul_tracks_reference_within_tolerance() {
        let (m, n, k) = (8, 6, 64);
        let a = synthetic_activations(m * k, 23);
        let b = synthetic_activations(k * n, 29);
        let reference = matmul_reference(&a, &b, m, n, k);
        let quantized = int8_matmul(&a, &b, m, n, k, minmax_params(&a), minmax_params(&b));

        let worst = max_abs_diff(&reference, &quantized);
        let scale = reference
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()))
            .max(1.0);
        assert!(
            worst / scale < 0.02,
            "int8 GEMM 相对误差应在 2% 内，实际 {}",
            worst / scale
        );
        assert!(worst > 0.0, "量化路径不应与 f32 完全一致");
    }
}
