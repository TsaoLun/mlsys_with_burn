use ch03_gemm_ladder::{deterministic_matrix, gemm_blocked, gemm_reference, max_abs_diff};

fn main() {
    let (m, n, k) = (17, 9, 33);
    let a = deterministic_matrix(m, k, 1);
    let b = deterministic_matrix(k, n, 2);
    let reference = gemm_reference(&a, &b, m, n, k);
    let blocked = gemm_blocked(&a, &b, m, n, k, 4);
    println!(
        "纯 Rust 分块 GEMM 与朴素参考的最大误差（M={m} N={n} K={k}，tile=4）：{}",
        max_abs_diff(&reference, &blocked)
    );

    #[cfg(not(feature = "wgpu"))]
    println!(
        "GPU 阶梯未启用；对比实验运行：cargo run -p ch03-gemm-ladder --features wgpu --locked"
    );

    #[cfg(feature = "wgpu")]
    gpu_ladder();
}

// ANCHOR: ladder_main
#[cfg(feature = "wgpu")]
fn gpu_ladder() {
    use ch03_gemm_ladder::gpu::{Ladder, run_gemm, runtime_name, time_gemm};
    use cubecl::wgpu::WgpuRuntime;

    let device = Default::default();
    println!(
        "\nRuntime：{}（同一协议只在本机内比较两级阶梯）",
        runtime_name::<WgpuRuntime>(&device)
    );

    // 先验证正确性，再谈时间。
    let (m, n, k) = (65, 33, 47);
    let a = deterministic_matrix(m, k, 3);
    let b = deterministic_matrix(k, n, 4);
    let reference = gemm_reference(&a, &b, m, n, k);
    for ladder in [Ladder::Naive, Ladder::Tiled] {
        let output =
            run_gemm::<WgpuRuntime>(&device, ladder, &a, &b, m, n, k).expect("WGPU GEMM 应可执行");
        println!(
            "{ladder:?} 与 host 参考最大误差：{:.2e}",
            max_abs_diff(&reference, &output)
        );
    }

    // 计时协议：预热 1 次 + 32 次 launch + 读回作为完成边界。
    println!(
        "\n{:>6}  {:>12}  {:>12}  {:>6}",
        "size", "naive µs", "tiled µs", "加速比"
    );
    for size in [256usize, 512, 1024] {
        let naive =
            time_gemm::<WgpuRuntime>(&device, Ladder::Naive, size, 32).expect("naive 计时应可执行");
        let tiled =
            time_gemm::<WgpuRuntime>(&device, Ladder::Tiled, size, 32).expect("tiled 计时应可执行");
        println!(
            "{size:>6}  {naive:>12.1}  {tiled:>12.1}  {:>6.2}",
            naive / tiled
        );
    }
    println!("\n数字只描述本机此次运行；跨设备结论需按第 3 章测量协议另行记录。");
}
// ANCHOR_END: ladder_main
