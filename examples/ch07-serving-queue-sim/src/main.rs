use ch07_serving_queue_sim::{
    DEFAULT_COST, mixed_workload, simulate_chunked_prefill, simulate_continuous, simulate_static,
    uniform_workload,
};

// ANCHOR: walkthrough
fn main() {
    let kv_budget = 16_384;
    let requests = mixed_workload(64, 5);

    println!("64 条混合长度请求（prompt 32–512，decode 16–256），KV 预算 {kv_budget} token：");
    println!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "调度", "平均 ms", "p95 ms", "p95 TTFT", "平均 TPOT", "总时长 ms", "空转槽步"
    );
    let static_batch = simulate_static(&requests, DEFAULT_COST, kv_budget, 8);
    let continuous = simulate_continuous(&requests, DEFAULT_COST, kv_budget);
    let chunked = simulate_chunked_prefill(&requests, DEFAULT_COST, kv_budget, 32);
    for (name, trace) in [
        ("静态批(8)", &static_batch),
        ("连续批处理", &continuous),
        ("分块 prefill", &chunked),
    ] {
        println!(
            "{:>12}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}  {:>8}",
            name,
            trace.mean_latency_us() / 1e3,
            trace.p95_latency_us() / 1e3,
            trace.p95_ttft_us() / 1e3,
            trace.mean_tpot_us() / 1e3,
            trace.makespan_us / 1e3,
            trace.idle_token_steps
        );
    }

    let uniform = uniform_workload(64);
    let static_uniform = simulate_static(&uniform, DEFAULT_COST, kv_budget, 8);
    let continuous_uniform = simulate_continuous(&uniform, DEFAULT_COST, kv_budget);
    println!(
        "\n对照：decode 长度全部相同时，静态批空转槽步 = {}，平均延迟差距收窄为 {:.2} 倍",
        static_uniform.idle_token_steps,
        static_uniform.mean_latency_us() / continuous_uniform.mean_latency_us()
    );

    println!("\nKV 预算对连续批处理吞吐的约束（同一负载）：");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>12}",
        "预算 tok", "总时长 ms", "tok/s", "峰值驻留 tok"
    );
    for budget in [2_048u64, 4_096, 8_192, 16_384, 32_768] {
        let trace = simulate_continuous(&requests, DEFAULT_COST, budget);
        println!(
            "{:>10}  {:>10.1}  {:>10.0}  {:>12}",
            budget,
            trace.makespan_us / 1e3,
            trace.throughput_tokens_per_s(&requests),
            trace.peak_resident_tokens
        );
    }
    println!("\n虚拟时间协议模型：解释机制，不代表任何真实服务 runtime 的性能。");
}
// ANCHOR_END: walkthrough
