use std::error::Error;

use ch09_cluster_simulator::{
    Cluster, Job, NetworkModel, PlacementPolicy, SimulationConfig, simulate,
};

fn main() -> Result<(), Box<dyn Error>> {
    let cluster = Cluster::uniform_interleaved(2, 1, 2, 16_000)?;
    let jobs = vec![
        Job::new(1, 2, 4_000, 6, 100, 4_000, 2).with_failure_step(3),
        Job::new(2, 1, 4_000, 4, 80, 2_000, 2),
        Job::new(3, 2, 4_000, 3, 120, 3_000, 0),
    ];

    for policy in [PlacementPolicy::Fifo, PlacementPolicy::TopologyAware] {
        let result = simulate(
            &cluster,
            jobs.clone(),
            SimulationConfig {
                placement_policy: policy,
                network: NetworkModel::new(50, 1, 4),
                checkpoint_cost_us: 25,
                max_retries: 1,
            },
        )?;

        println!(
            "policy={policy} jobs={} completed={} makespan_ms={} \
queue_wait_ms={} p95_queue_wait_ms={} cross_rack_bytes={} \
collective_ms={} retries={} peak_allocated_gpus={}",
            result.jobs.len(),
            result
                .jobs
                .iter()
                .filter(|job| job.completed_step > 0)
                .count(),
            to_ms(result.makespan_us),
            to_ms(result.total_queue_wait_us),
            to_ms(result.p95_queue_wait_us),
            result.cross_rack_bytes,
            to_ms(result.collective_time_us),
            result.retries,
            result.peak_allocated_gpus,
        );

        for job in &result.jobs {
            println!(
                "job={} attempts={} retries={} queue_wait_ms={} \
checkpoint_replay_steps={} placements={:?}",
                job.id,
                job.attempts,
                job.retries,
                to_ms(job.queue_wait_us),
                job.checkpoint_replay_steps,
                job.placements,
            );
        }
    }

    Ok(())
}

fn to_ms(microseconds: u64) -> u64 {
    microseconds.div_ceil(1_000)
}
