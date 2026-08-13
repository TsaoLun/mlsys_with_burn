use ch07_ptq_calibration::{
    inject_outliers, int8_matmul, matmul_reference, max_abs_diff, minmax_params, mse,
    mse_excluding, outlier_positions, percentile_params, reconstruct, reconstruct_per_channel,
    synthetic_activations,
};

// ANCHOR: walkthrough
fn main() {
    // 1) 含 1% 大离群值的激活：校准是「主体分辨率 vs 离群值误差」的交易。
    let mut activations = synthetic_activations(8192, 11);
    let outliers = outlier_positions(activations.len(), 82);
    inject_outliers(&mut activations, 82, 80.0);

    let minmax = minmax_params(&activations);
    let clipped = percentile_params(&activations, 0.005);
    let with_minmax = reconstruct(&activations, minmax);
    let with_clip = reconstruct(&activations, clipped);
    println!("校准策略（8192 样本，1% 离群值 ±80）：");
    println!(
        "  min-max  scale={:.5}  主体 MSE={:.8}  整体 MSE={:.5}",
        minmax.scale,
        mse_excluding(&activations, &with_minmax, &outliers),
        mse(&activations, &with_minmax)
    );
    println!(
        "  p99.5    scale={:.5}  主体 MSE={:.8}  整体 MSE={:.5}",
        clipped.scale,
        mse_excluding(&activations, &with_clip, &outliers),
        mse(&activations, &with_clip)
    );
    println!("  → 分位校准主体好一个数量级以上，整体反而更差：指标必须与任务对齐。");

    // 2) 动态范围差 100 倍的两行权重：粒度收益要按通道看。
    let cols = 512;
    let mut weights = synthetic_activations(cols, 17);
    let narrow: Vec<f32> = synthetic_activations(cols, 19)
        .into_iter()
        .map(|value| value * 0.01)
        .collect();
    weights.extend(narrow.clone());
    let per_tensor = reconstruct(&weights, minmax_params(&weights));
    let per_channel = reconstruct_per_channel(&weights, 2, cols);
    println!("\n量化粒度（两行权重，范围差 100 倍）——只看窄行：");
    println!(
        "  per-tensor  窄行 MSE={:.10}",
        mse(&narrow, &per_tensor[cols..])
    );
    println!(
        "  per-channel 窄行 MSE={:.10}",
        mse(&narrow, &per_channel[cols..])
    );

    // 3) int8 矩阵乘（i32 累加）与 f32 参考。
    let (m, n, k) = (8, 6, 64);
    let a = synthetic_activations(m * k, 23);
    let b = synthetic_activations(k * n, 29);
    let reference = matmul_reference(&a, &b, m, n, k);
    let quantized = int8_matmul(&a, &b, m, n, k, minmax_params(&a), minmax_params(&b));
    println!(
        "\nint8 GEMM 与 f32 参考的最大绝对误差：{:.4}",
        max_abs_diff(&reference, &quantized)
    );
    println!("以上为纯 Rust 协议演算，不代表任何低精度 backend 的执行路径或速度。");
}
// ANCHOR_END: walkthrough
