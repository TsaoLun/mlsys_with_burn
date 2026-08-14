//! 生成式服务的队列协议模型：静态批处理 vs 连续批处理（continuous
//! batching），外加 KV cache 容量预算、TTFT/TPOT 与 chunked prefill。
//!
//! 这是与第 9 章集群模拟器同类的**纯 Rust 虚拟时间协议模型**：它解释
//! Orca/vLLM 一系的机制为什么有效，不代表任何真实服务 runtime 的
//! 性能。KV 预算按「prompt + 全部 decode」预留，不做抢占与换出。

// ANCHOR: model
/// 一条生成请求：到达时刻、prompt 长度与要生成的 token 数。
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub arrival_us: u64,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
}

/// 一步（iteration）的耗时模型：固定开销 + 本步处理的 token 数。
/// decode 中的序列每步贡献 1 个 token；prefill 按本步实际处理的
/// prompt token 计——大 prompt 若一次吃完，会拖慢同一步里的所有请求。
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
    /// 每条请求的 TTFT：到达到第一个 decode token 完成（µs）。
    pub ttft_us: Vec<f64>,
    /// 每条请求的 TPOT：首 token 之后每个后续 decode token 的平均间隔。
    /// `decode_tokens <= 1` 时为 0。
    pub tpot_us: Vec<f64>,
    /// 全部请求完成的时刻（µs）。
    pub makespan_us: f64,
    /// 模拟中同时驻留的 KV token 峰值。
    pub peak_resident_tokens: u64,
    /// 空转 token 槽位数：静态批中已完成序列等待整批结束的步数。
    pub idle_token_steps: u64,
}

impl Trace {
    pub fn mean_latency_us(&self) -> f64 {
        mean(&self.latency_us)
    }

    pub fn p95_latency_us(&self) -> f64 {
        percentile(&self.latency_us, 0.95)
    }

    pub fn mean_ttft_us(&self) -> f64 {
        mean(&self.ttft_us)
    }

    pub fn p95_ttft_us(&self) -> f64 {
        percentile(&self.ttft_us, 0.95)
    }

    pub fn mean_tpot_us(&self) -> f64 {
        mean(&self.tpot_us)
    }

    pub fn throughput_tokens_per_s(&self, requests: &[Request]) -> f64 {
        let tokens: u64 = requests.iter().map(|r| r.decode_tokens as u64).sum();
        tokens as f64 / (self.makespan_us / 1e6)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * pct) as usize).min(sorted.len() - 1);
    sorted[index]
}

fn kv_cost(request: &Request) -> u64 {
    (request.prompt_tokens + request.decode_tokens) as u64
}

fn tpot_us(finish: f64, first_token: f64, decode_tokens: u32) -> f64 {
    let subsequent = decode_tokens.saturating_sub(1);
    if subsequent == 0 {
        0.0
    } else {
        (finish - first_token) / f64::from(subsequent)
    }
}

// ANCHOR: continuous
/// 连续批处理：每一步开始时，只要 KV 预算允许就接纳队首请求；每条
/// 运行中的序列本步 decode 一个 token，完成即离开并归还 KV 预算。
/// Prefill 在接纳步一次处理完（等价于 chunk = `u32::MAX` 的分块 prefill）。
pub fn simulate_continuous(requests: &[Request], cost: CostModel, kv_budget: u64) -> Trace {
    simulate_chunked_prefill(requests, cost, kv_budget, u32::MAX)
}
// ANCHOR_END: continuous

// ANCHOR: chunked
/// 分块 prefill：每条仍在吃 prompt 的序列，每步至多处理 `chunk_tokens`
/// 个 prompt token；已经进入 decode 的序列仍每步产出 1 个 token。
/// 长 prompt 因此不会独占整整一步。`chunk_tokens == 0` 按 1 处理。
pub fn simulate_chunked_prefill(
    requests: &[Request],
    cost: CostModel,
    kv_budget: u64,
    chunk_tokens: u32,
) -> Trace {
    #[derive(Clone, Copy)]
    struct Running {
        index: usize,
        remaining_prefill: u32,
        remaining_decode: u32,
        resident: u64,
        first_token_us: Option<f64>,
    }

    let chunk = chunk_tokens.max(1);
    let mut latency = vec![0.0f64; requests.len()];
    let mut ttft = vec![0.0f64; requests.len()];
    let mut tpot = vec![0.0f64; requests.len()];
    let mut now = 0.0f64;
    let mut next_arrival = 0usize;
    let mut running: Vec<Running> = Vec::new();
    let mut resident: u64 = 0;
    let mut peak = 0u64;

    while next_arrival < requests.len() || !running.is_empty() {
        if running.is_empty() && next_arrival < requests.len() {
            now = now.max(requests[next_arrival].arrival_us as f64);
        }

        while next_arrival < requests.len()
            && requests[next_arrival].arrival_us as f64 <= now
            && (resident + kv_cost(&requests[next_arrival]) <= kv_budget || running.is_empty())
        {
            let request = requests[next_arrival];
            running.push(Running {
                index: next_arrival,
                remaining_prefill: request.prompt_tokens,
                remaining_decode: request.decode_tokens.max(1),
                resident: kv_cost(&request),
                first_token_us: None,
            });
            resident += kv_cost(&request);
            next_arrival += 1;
        }
        peak = peak.max(resident);

        let mut work = 0u64;
        let mut first_decode = Vec::new();
        for (slot, job) in running.iter_mut().enumerate() {
            if job.remaining_prefill > 0 {
                let take = job.remaining_prefill.min(chunk);
                work += u64::from(take);
                job.remaining_prefill -= take;
            }
            if job.remaining_prefill == 0 && job.remaining_decode > 0 {
                work += 1;
                job.remaining_decode -= 1;
                if job.first_token_us.is_none() {
                    first_decode.push(slot);
                }
            }
        }
        now += cost.step_us(work);
        for slot in first_decode {
            running[slot].first_token_us = Some(now);
        }

        let mut index = 0;
        while index < running.len() {
            if running[index].remaining_prefill == 0 && running[index].remaining_decode == 0 {
                let done = running.swap_remove(index);
                let first = done.first_token_us.unwrap_or(now);
                latency[done.index] = now - requests[done.index].arrival_us as f64;
                ttft[done.index] = first - requests[done.index].arrival_us as f64;
                tpot[done.index] = tpot_us(now, first, requests[done.index].decode_tokens);
                resident -= done.resident;
            } else {
                index += 1;
            }
        }
    }

    Trace {
        latency_us: latency,
        ttft_us: ttft,
        tpot_us: tpot,
        makespan_us: now,
        peak_resident_tokens: peak,
        idle_token_steps: 0,
    }
}
// ANCHOR_END: chunked

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
    let mut ttft = vec![0.0f64; requests.len()];
    let mut tpot = vec![0.0f64; requests.len()];
    let mut now = 0.0f64;
    let mut next = 0usize;
    let mut peak = 0u64;
    let mut idle_token_steps = 0u64;

    while next < requests.len() {
        now = now.max(requests[next].arrival_us as f64);

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
            resident = kv_cost(&requests[next]);
            batch.push(next);
            next += 1;
        }
        peak = peak.max(resident);

        let prefill: u64 = batch
            .iter()
            .map(|&index| requests[index].prompt_tokens as u64)
            .sum();
        now += cost.step_us(prefill + batch.len() as u64);
        let first_now = now;

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
        for &index in &batch {
            latency[index] = now - requests[index].arrival_us as f64;
            ttft[index] = first_now - requests[index].arrival_us as f64;
            tpot[index] = tpot_us(now, first_now, requests[index].decode_tokens);
        }
    }

    Trace {
        latency_us: latency,
        ttft_us: ttft,
        tpot_us: tpot,
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

    /// 两种调度都必须服务全部请求（守恒），且延迟、TTFT 为正。
    #[test]
    fn both_schedulers_serve_every_request() {
        let requests = mixed_workload(64, 5);
        for trace in [
            simulate_continuous(&requests, DEFAULT_COST, KV),
            simulate_static(&requests, DEFAULT_COST, KV, 8),
        ] {
            assert_eq!(trace.latency_us.len(), requests.len());
            assert_eq!(trace.ttft_us.len(), requests.len());
            assert!(trace.latency_us.iter().all(|&latency| latency > 0.0));
            assert!(trace.ttft_us.iter().all(|&ttft| ttft > 0.0));
            for (ttft, latency) in trace.ttft_us.iter().zip(&trace.latency_us) {
                assert!(*ttft <= *latency + 1e-9);
            }
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
        let chunked = simulate_chunked_prefill(&requests, DEFAULT_COST, KV, 32);
        assert_eq!(
            chunked,
            simulate_chunked_prefill(&requests, DEFAULT_COST, KV, 32)
        );
    }

    /// 单条请求的 TTFT / TPOT 必须与成本模型逐步相加一致。
    #[test]
    fn ttft_is_first_decode_step_tpot_is_the_rest() {
        let requests = [Request {
            arrival_us: 0,
            prompt_tokens: 10,
            decode_tokens: 5,
        }];
        let trace = simulate_continuous(&requests, DEFAULT_COST, KV);
        assert!((trace.ttft_us[0] - 244.0).abs() < 1e-9);
        assert!((trace.tpot_us[0] - 204.0).abs() < 1e-9);
        assert!((trace.latency_us[0] - (244.0 + 4.0 * 204.0)).abs() < 1e-9);
    }

    /// `chunk = u32::MAX` 必须与原来的连续批处理逐步重合。
    #[test]
    fn unbounded_chunk_matches_continuous() {
        let requests = mixed_workload(32, 3);
        let continuous = simulate_continuous(&requests, DEFAULT_COST, KV);
        let chunked = simulate_chunked_prefill(&requests, DEFAULT_COST, KV, u32::MAX);
        assert_eq!(continuous, chunked);
    }

    /// 长 prompt 到达时，分块 prefill 保护已经在 decode 的序列：
    /// 它们不必等 512 个 prompt token 在同一步里处理完。
    #[test]
    fn chunked_prefill_protects_inflight_decode() {
        let requests = [
            Request {
                arrival_us: 0,
                prompt_tokens: 4,
                decode_tokens: 40,
            },
            Request {
                arrival_us: 800,
                prompt_tokens: 512,
                decode_tokens: 8,
            },
        ];
        let unchunked = simulate_continuous(&requests, DEFAULT_COST, 100_000);
        let chunked = simulate_chunked_prefill(&requests, DEFAULT_COST, 100_000, 16);
        assert!(
            chunked.latency_us[0] < unchunked.latency_us[0],
            "在飞 decode 的端到端延迟应下降：chunked={:.0} unchunked={:.0}",
            chunked.latency_us[0],
            unchunked.latency_us[0]
        );
        assert!(
            chunked.tpot_us[0] < unchunked.tpot_us[0],
            "在飞 decode 的 TPOT 应下降：chunked={:.1} unchunked={:.1}",
            chunked.tpot_us[0],
            unchunked.tpot_us[0]
        );
        assert_eq!(chunked.latency_us.len(), 2);
        assert!(chunked.ttft_us[1] > 0.0);
    }
}
