//! 生成式服务的队列协议模型：静态批处理 vs 连续批处理（continuous
//! batching），外加 KV cache 容量预算对并发的约束。
//!
//! 这是与第 9 章集群模拟器同类的**纯 Rust 虚拟时间协议模型**：它解释
//! Orca/vLLM 一系的机制为什么有效，不代表任何真实服务 runtime 的
//! 性能。两点刻意简化：prefill 在被接纳的那一步一次性处理全部
//! prompt token（不做 chunked prefill）；KV 预算按「prompt + 全部
//! decode」预留，不做抢占与换出。

// ANCHOR: model
/// 一条生成请求：到达时刻、prompt 长度与要生成的 token 数。
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub arrival_us: u64,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
}

/// 一步（iteration）的耗时模型：固定开销 + 本步处理的 token 数。
/// decode 中的序列每步贡献 1 个 token；prefill 在接纳步贡献全部
/// prompt token——大 prompt 会拖慢同一步里的所有请求。
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub step_overhead_us: f64,
    pub per_token_us: f64,
}

impl CostModel {
    fn step_us(&self, tokens_this_step: u64) -> f64 {
        self.step_overhead_us + self.per_token_us * tokens_this_step as f64
    }
}
// ANCHOR_END: model

/// 一次模拟的汇总指标。
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// 每条请求从到达到最后一个 token 完成的延迟（µs），按输入顺序。
    pub latency_us: Vec<f64>,
    /// 全部请求完成的时刻（µs）。
    pub makespan_us: f64,
    /// 模拟中同时驻留的 KV token 峰值。
    pub peak_resident_tokens: u64,
    /// 空转 token 槽位数：静态批中已完成序列等待整批结束的步数。
    pub idle_token_steps: u64,
}

impl Trace {
    pub fn mean_latency_us(&self) -> f64 {
        self.latency_us.iter().sum::<f64>() / self.latency_us.len() as f64
    }

    pub fn p95_latency_us(&self) -> f64 {
        let mut sorted = self.latency_us.clone();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        sorted[index]
    }

    pub fn throughput_tokens_per_s(&self, requests: &[Request]) -> f64 {
        let tokens: u64 = requests.iter().map(|r| r.decode_tokens as u64).sum();
        tokens as f64 / (self.makespan_us / 1e6)
    }
}

fn kv_cost(request: &Request) -> u64 {
    (request.prompt_tokens + request.decode_tokens) as u64
}

// ANCHOR: continuous
/// 连续批处理：每一步开始时，只要 KV 预算允许就接纳队首请求；每条
/// 运行中的序列本步 decode 一个 token，完成即离开并归还 KV 预算。
pub fn simulate_continuous(requests: &[Request], cost: CostModel, kv_budget: u64) -> Trace {
    #[derive(Clone, Copy)]
    struct Running {
        index: usize,
        remaining: u32,
        resident: u64,
    }

    let mut latency = vec![0.0f64; requests.len()];
    let mut now = 0.0f64;
    let mut next_arrival = 0usize;
    let mut running: Vec<Running> = Vec::new();
    let mut resident: u64 = 0;
    let mut peak = 0u64;

    while next_arrival < requests.len() || !running.is_empty() {
        // 空转推进：没有可运行请求时，时间跳到下一个到达。
        if running.is_empty() && next_arrival < requests.len() {
            now = now.max(requests[next_arrival].arrival_us as f64);
        }

        // 接纳：按到达顺序（FCFS），预算按 prompt+decode 预留。
        let mut prefill_tokens: u64 = 0;
        while next_arrival < requests.len()
            && requests[next_arrival].arrival_us as f64 <= now
            && (resident + kv_cost(&requests[next_arrival]) <= kv_budget
                // 单条请求超出预算时也要在空闲时单独跑，避免永久卡死。
                || running.is_empty() && prefill_tokens == 0)
        {
            let request = requests[next_arrival];
            running.push(Running {
                index: next_arrival,
                remaining: request.decode_tokens.max(1),
                resident: kv_cost(&request),
            });
            resident += kv_cost(&request);
            prefill_tokens += request.prompt_tokens as u64;
            next_arrival += 1;
        }
        peak = peak.max(resident);

        // 一步：prefill token + 每条运行序列 1 个 decode token。
        let decode_tokens = running.len() as u64;
        now += cost.step_us(prefill_tokens + decode_tokens);

        // 完成的序列立即离开，释放预算给下一步的接纳。
        let mut index = 0;
        while index < running.len() {
            running[index].remaining -= 1;
            if running[index].remaining == 0 {
                let done = running.swap_remove(index);
                latency[done.index] = now - requests[done.index].arrival_us as f64;
                resident -= done.resident;
            } else {
                index += 1;
            }
        }
    }

    Trace {
        latency_us: latency,
        makespan_us: now,
        peak_resident_tokens: peak,
        idle_token_steps: 0,
    }
}
// ANCHOR_END: continuous

// ANCHOR: static_batching
/// 静态批处理：凑一个批（至多 `batch_size` 条、受同一 KV 预算约束），
/// 整批 prefill、整批 decode；先完成的序列占着槽位空转，直到批内
/// 最长的序列结束才换下一批。
pub fn simulate_static(
    requests: &[Request],
    cost: CostModel,
    kv_budget: u64,
    batch_size: usize,
) -> Trace {
    let mut latency = vec![0.0f64; requests.len()];
    let mut now = 0.0f64;
    let mut next = 0usize;
    let mut peak = 0u64;
    let mut idle_token_steps = 0u64;

    while next < requests.len() {
        now = now.max(requests[next].arrival_us as f64);

        // 组批：FCFS，直到批满或预算不够。
        let mut batch: Vec<usize> = Vec::new();
        let mut resident: u64 = 0;
        while next < requests.len()
            && batch.len() < batch_size
            && requests[next].arrival_us as f64 <= now
            && resident + kv_cost(&requests[next]) <= kv_budget
        {
            resident += kv_cost(&requests[next]);
            batch.push(next);
            next += 1;
        }
        if batch.is_empty() {
            // 预算装不下队首请求也要单独跑，否则永远卡死。
            resident = kv_cost(&requests[next]);
            batch.push(next);
            next += 1;
        }
        peak = peak.max(resident);

        // 整批 prefill 一步。
        let prefill: u64 = batch
            .iter()
            .map(|&index| requests[index].prompt_tokens as u64)
            .sum();
        now += cost.step_us(prefill + batch.len() as u64);

        // decode：批内最长的序列决定批何时结束。
        let longest = batch
            .iter()
            .map(|&index| requests[index].decode_tokens)
            .max()
            .unwrap_or(0);
        let mut remaining: Vec<(usize, u32)> = batch
            .iter()
            .map(|&index| (index, requests[index].decode_tokens.saturating_sub(1)))
            .collect();
        for _ in 1..longest {
            let active = remaining.iter().filter(|(_, left)| *left > 0).count() as u64;
            idle_token_steps += remaining.len() as u64 - active;
            now += cost.step_us(remaining.len() as u64);
            for entry in &mut remaining {
                entry.1 = entry.1.saturating_sub(1);
            }
        }
        for (index, _) in &remaining {
            latency[*index] = now - requests[*index].arrival_us as f64;
        }
    }

    Trace {
        latency_us: latency,
        makespan_us: now,
        peak_resident_tokens: peak,
        idle_token_steps,
    }
}
// ANCHOR_END: static_batching

/// 确定性混合负载：prompt 与 decode 长度都高度不均匀。
pub fn mixed_workload(count: usize, seed: u64) -> Vec<Request> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = move |range: u32| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % range
    };
    let mut arrival = 0u64;
    (0..count)
        .map(|_| {
            arrival += next(3_000) as u64;
            Request {
                arrival_us: arrival,
                prompt_tokens: 32 + next(480),
                decode_tokens: 16 + next(240),
            }
        })
        .collect()
}

/// 长度全部相同的对照负载：连续批处理的收益应当显著缩小。
pub fn uniform_workload(count: usize) -> Vec<Request> {
    (0..count)
        .map(|index| Request {
            arrival_us: index as u64 * 1_000,
            prompt_tokens: 128,
            decode_tokens: 96,
        })
        .collect()
}

pub const DEFAULT_COST: CostModel = CostModel {
    step_overhead_us: 200.0,
    per_token_us: 4.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    const KV: u64 = 16_384;

    /// 两种调度都必须服务全部请求（守恒），且延迟为正。
    #[test]
    fn both_schedulers_serve_every_request() {
        let requests = mixed_workload(64, 5);
        for trace in [
            simulate_continuous(&requests, DEFAULT_COST, KV),
            simulate_static(&requests, DEFAULT_COST, KV, 8),
        ] {
            assert_eq!(trace.latency_us.len(), requests.len());
            assert!(trace.latency_us.iter().all(|&latency| latency > 0.0));
        }
    }

    /// 混合长度下，连续批处理的平均延迟与总完成时间都应明显更好；
    /// 静态批的空转 token 槽位解释了差距来自哪里。
    #[test]
    fn continuous_batching_wins_on_mixed_lengths() {
        let requests = mixed_workload(64, 5);
        let continuous = simulate_continuous(&requests, DEFAULT_COST, KV);
        let static_batch = simulate_static(&requests, DEFAULT_COST, KV, 8);

        assert!(
            continuous.mean_latency_us() < static_batch.mean_latency_us() * 0.8,
            "平均延迟应至少好 20%：continuous={:.0} static={:.0}",
            continuous.mean_latency_us(),
            static_batch.mean_latency_us()
        );
        assert!(continuous.makespan_us < static_batch.makespan_us);
        assert!(static_batch.idle_token_steps > 0);
        assert_eq!(continuous.idle_token_steps, 0);
    }

    /// decode 长度全部相同时，静态批没有「等最长序列」的浪费，
    /// 两者差距应当收窄——收益来自长度方差，不是魔法。
    #[test]
    fn uniform_lengths_shrink_the_gap() {
        let requests = uniform_workload(64);
        let continuous = simulate_continuous(&requests, DEFAULT_COST, KV);
        let static_batch = simulate_static(&requests, DEFAULT_COST, KV, 8);

        assert_eq!(static_batch.idle_token_steps, 0);
        assert!(static_batch.mean_latency_us() < continuous.mean_latency_us() * 2.0);
    }

    /// KV 预算越小，可同时驻留的序列越少，总完成时间单调变差；
    /// 峰值驻留永不超过预算。
    #[test]
    fn kv_budget_throttles_concurrency() {
        let requests = mixed_workload(64, 5);
        let mut previous_makespan = 0.0f64;
        for budget in [2_048u64, 8_192, 32_768] {
            let trace = simulate_continuous(&requests, DEFAULT_COST, budget);
            assert!(trace.peak_resident_tokens <= budget);
            if previous_makespan > 0.0 {
                assert!(
                    trace.makespan_us <= previous_makespan,
                    "预算 {budget} 的完成时间不应差于更小预算"
                );
            }
            previous_makespan = trace.makespan_us;
        }
    }

    /// 同一负载与参数重复模拟，轨迹完全一致（确定性）。
    #[test]
    fn simulation_is_deterministic() {
        let requests = mixed_workload(48, 9);
        let first = simulate_continuous(&requests, DEFAULT_COST, KV);
        let second = simulate_continuous(&requests, DEFAULT_COST, KV);
        assert_eq!(first, second);
    }
}
