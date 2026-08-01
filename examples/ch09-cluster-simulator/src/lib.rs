use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    num::NonZeroU32,
};

// ANCHOR: cluster_model
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpu {
    pub id: usize,
    pub node: usize,
    pub rack: usize,
    pub memory_mb: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cluster {
    gpus: Vec<Gpu>,
}

impl Cluster {
    pub fn new(gpus: Vec<Gpu>) -> Result<Self, SimulationError> {
        if gpus.is_empty() {
            return Err(SimulationError::InvalidCluster(
                "cluster must contain at least one GPU",
            ));
        }

        let mut ids = BTreeSet::new();
        for gpu in &gpus {
            if gpu.memory_mb == 0 {
                return Err(SimulationError::InvalidCluster(
                    "GPU memory capacity must be positive",
                ));
            }
            if !ids.insert(gpu.id) {
                return Err(SimulationError::DuplicateGpuId(gpu.id));
            }
        }

        Ok(Self { gpus })
    }

    /// Creates an intentionally interleaved rack layout.
    ///
    /// Consecutive IDs belong to different racks. This makes the difference
    /// between first-fit and topology-aware placement visible in a small
    /// deterministic fixture.
    pub fn uniform_interleaved(
        rack_count: usize,
        nodes_per_rack: usize,
        gpus_per_node: usize,
        memory_mb: u64,
    ) -> Result<Self, SimulationError> {
        if rack_count == 0 || nodes_per_rack == 0 || gpus_per_node == 0 {
            return Err(SimulationError::InvalidCluster(
                "rack, node, and GPU counts must be positive",
            ));
        }

        let mut gpus = Vec::new();
        let mut id = 0;
        for slot in 0..nodes_per_rack * gpus_per_node {
            for rack in 0..rack_count {
                gpus.push(Gpu {
                    id,
                    node: slot / gpus_per_node,
                    rack,
                    memory_mb,
                });
                id += 1;
            }
        }

        Self::new(gpus)
    }

    pub fn gpus(&self) -> &[Gpu] {
        &self.gpus
    }

    fn gpu(&self, id: usize) -> Option<&Gpu> {
        self.gpus.iter().find(|gpu| gpu.id == id)
    }
}
// ANCHOR_END: cluster_model

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub id: usize,
    pub gpu_count: usize,
    pub memory_mb: u64,
    pub steps: u32,
    pub compute_us_per_step: u64,
    pub gradient_bytes: u64,
    pub checkpoint_interval: u32,
    pub failure_step: Option<u32>,
}

impl Job {
    pub fn new(
        id: usize,
        gpu_count: usize,
        memory_mb: u64,
        steps: u32,
        compute_us_per_step: u64,
        gradient_bytes: u64,
        checkpoint_interval: u32,
    ) -> Self {
        Self {
            id,
            gpu_count,
            memory_mb,
            steps,
            compute_us_per_step,
            gradient_bytes,
            checkpoint_interval,
            failure_step: None,
        }
    }

    pub fn with_failure_step(mut self, step: u32) -> Self {
        self.failure_step = Some(step);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementPolicy {
    Fifo,
    TopologyAware,
}

impl fmt::Display for PlacementPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fifo => formatter.write_str("fifo"),
            Self::TopologyAware => formatter.write_str("topology-aware"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkModel {
    pub alpha_us: u64,
    pub beta_ns_per_byte: u64,
    pub cross_rack_multiplier: u64,
}

impl NetworkModel {
    pub const fn new(alpha_us: u64, beta_ns_per_byte: u64, cross_rack_multiplier: u64) -> Self {
        Self {
            alpha_us,
            beta_ns_per_byte,
            cross_rack_multiplier,
        }
    }
}

impl Default for NetworkModel {
    fn default() -> Self {
        Self::new(50, 1, 4)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommunicationCost {
    pub total_us: u64,
    pub cross_rack_bytes: u64,
    pub rounds: u64,
}

pub fn communication_cost(
    cluster: &Cluster,
    placement: &[usize],
    gradient_bytes: u64,
    network: NetworkModel,
) -> CommunicationCost {
    if placement.len() < 2 {
        return CommunicationCost {
            total_us: 0,
            cross_rack_bytes: 0,
            rounds: 0,
        };
    }

    let participants = placement.len() as u128;
    let rounds = 2 * (placement.len() - 1) as u64;
    let logical_bytes = u128::from(gradient_bytes) * (participants - 1);
    let base_transfer_us =
        nanos_to_micros(u128::from(network.beta_ns_per_byte).saturating_mul(logical_bytes));
    let base_latency_us = u128::from(network.alpha_us) * u128::from(rounds);

    let mut cross_rack_pairs = 0_u128;
    for (left_index, left_id) in placement.iter().enumerate() {
        for right_id in placement.iter().skip(left_index + 1) {
            let Some(left) = cluster.gpu(*left_id) else {
                continue;
            };
            let Some(right) = cluster.gpu(*right_id) else {
                continue;
            };
            if left.rack != right.rack {
                cross_rack_pairs += 1;
            }
        }
    }

    let cross_rack_bytes = u128::from(gradient_bytes).saturating_mul(cross_rack_pairs);
    let extra_multiplier = network.cross_rack_multiplier.saturating_sub(1);
    let cross_rack_penalty_us = nanos_to_micros(
        u128::from(network.beta_ns_per_byte)
            .saturating_mul(cross_rack_bytes)
            .saturating_mul(u128::from(extra_multiplier)),
    );

    CommunicationCost {
        total_us: saturating_u128_to_u64(
            base_latency_us
                .saturating_add(base_transfer_us)
                .saturating_add(cross_rack_penalty_us),
        ),
        cross_rack_bytes: saturating_u128_to_u64(cross_rack_bytes),
        rounds,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationConfig {
    pub placement_policy: PlacementPolicy,
    pub network: NetworkModel,
    pub checkpoint_cost_us: u64,
    pub max_retries: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            placement_policy: PlacementPolicy::TopologyAware,
            network: NetworkModel::default(),
            checkpoint_cost_us: 25,
            max_retries: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    JobStarted {
        job_id: usize,
        attempt: usize,
        at_us: u64,
        placement: Vec<usize>,
        resume_step: u32,
    },
    JobFailed {
        job_id: usize,
        attempt: usize,
        at_us: u64,
        failed_step: u32,
        checkpoint_step: u32,
        replayed_steps: u32,
    },
    JobCompleted {
        job_id: usize,
        attempt: usize,
        at_us: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobReport {
    pub id: usize,
    pub placements: Vec<Vec<usize>>,
    pub queue_wait_us: u64,
    pub attempts: usize,
    pub retries: usize,
    pub completed_step: u32,
    pub checkpoint_replay_steps: u32,
    pub collective_time_us: u64,
    pub cross_rack_bytes: u64,
    pub start_us: Option<u64>,
    pub end_us: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationResult {
    pub policy: PlacementPolicy,
    pub jobs: Vec<JobReport>,
    pub makespan_us: u64,
    pub total_queue_wait_us: u64,
    pub p95_queue_wait_us: u64,
    pub cross_rack_bytes: u64,
    pub collective_time_us: u64,
    pub retries: usize,
    pub peak_allocated_gpus: usize,
    pub free_gpu_count: usize,
    pub trace: Vec<TraceEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimulationError {
    InvalidCluster(&'static str),
    DuplicateGpuId(usize),
    DuplicateJobId(usize),
    InvalidJob { id: usize, reason: &'static str },
    JobDoesNotFit(usize),
    RetryLimitExceeded(usize),
    MissingReport(usize),
    SchedulerDeadlock,
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCluster(reason) => write!(formatter, "invalid cluster: {reason}"),
            Self::DuplicateGpuId(id) => write!(formatter, "duplicate GPU id: {id}"),
            Self::DuplicateJobId(id) => write!(formatter, "duplicate job id: {id}"),
            Self::InvalidJob { id, reason } => {
                write!(formatter, "invalid job {id}: {reason}")
            }
            Self::JobDoesNotFit(id) => write!(formatter, "job {id} cannot fit the cluster"),
            Self::RetryLimitExceeded(id) => write!(formatter, "job {id} exceeded retry limit"),
            Self::MissingReport(id) => write!(formatter, "missing report for job {id}"),
            Self::SchedulerDeadlock => formatter.write_str("scheduler made no progress"),
        }
    }
}

impl std::error::Error for SimulationError {}

#[derive(Clone, Debug)]
struct PendingJob {
    job: Job,
    queued_at_us: u64,
    resume_step: u32,
    attempt: usize,
    failure_injected: bool,
}

#[derive(Clone, Debug)]
struct RunningJob {
    pending: PendingJob,
    placement: Vec<usize>,
    finish_us: u64,
    failure_us: Option<u64>,
    communication: CommunicationCost,
}

// ANCHOR: simulator_api
pub fn simulate(
    cluster: &Cluster,
    jobs: Vec<Job>,
    config: SimulationConfig,
) -> Result<SimulationResult, SimulationError> {
    let mut pending = VecDeque::new();
    let mut reports = BTreeMap::new();

    for job in jobs {
        validate_job(cluster, &job)?;
        if reports.insert(job.id, empty_report(job.id)).is_some() {
            return Err(SimulationError::DuplicateJobId(job.id));
        }
        pending.push_back(PendingJob {
            job,
            queued_at_us: 0,
            resume_step: 0,
            attempt: 0,
            failure_injected: false,
        });
    }

    let mut free = cluster
        .gpus()
        .iter()
        .map(|gpu| gpu.id)
        .collect::<BTreeSet<_>>();
    let mut running = Vec::new();
    let mut trace = Vec::new();
    let mut now_us = 0;
    let mut peak_allocated_gpus = 0;
    let mut total_retries = 0;
    let mut total_collective_time_us = 0;
    let mut total_cross_rack_bytes = 0;

    while !pending.is_empty() || !running.is_empty() {
        admit_jobs(
            cluster,
            &mut pending,
            &mut running,
            &mut free,
            &mut reports,
            &mut trace,
            &config,
            now_us,
            &mut peak_allocated_gpus,
        )?;

        if running.is_empty() {
            return Err(SimulationError::SchedulerDeadlock);
        }

        let Some((event_index, event_time, is_failure)) = running
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let failure = item.failure_us;
                let time = failure.map_or(item.finish_us, |failure_time| {
                    failure_time.min(item.finish_us)
                });
                (index, time, failure == Some(time))
            })
            .min_by_key(|(_, time, is_failure)| (*time, !*is_failure))
        else {
            return Err(SimulationError::SchedulerDeadlock);
        };

        now_us = event_time;
        let running_job = running.swap_remove(event_index);
        release(&mut free, &running_job.placement);

        if is_failure {
            handle_failure(
                &mut pending,
                &mut reports,
                &mut trace,
                &running_job,
                &config,
                now_us,
                &mut total_retries,
            )?;
        } else {
            handle_completion(
                &mut reports,
                &mut trace,
                &running_job,
                now_us,
                &mut total_collective_time_us,
                &mut total_cross_rack_bytes,
            )?;
        }
    }

    let mut jobs = reports.into_values().collect::<Vec<_>>();
    jobs.sort_by_key(|report| report.id);
    let queue_waits = jobs
        .iter()
        .map(|report| report.queue_wait_us)
        .collect::<Vec<_>>();
    let total_queue_wait_us = queue_waits.iter().sum();

    Ok(SimulationResult {
        policy: config.placement_policy,
        jobs,
        makespan_us: now_us,
        total_queue_wait_us,
        p95_queue_wait_us: percentile_95(&queue_waits),
        cross_rack_bytes: total_cross_rack_bytes,
        collective_time_us: total_collective_time_us,
        retries: total_retries,
        peak_allocated_gpus,
        free_gpu_count: free.len(),
        trace,
    })
}
// ANCHOR_END: simulator_api

fn validate_job(cluster: &Cluster, job: &Job) -> Result<(), SimulationError> {
    if job.gpu_count == 0 {
        return Err(SimulationError::InvalidJob {
            id: job.id,
            reason: "GPU count must be positive",
        });
    }
    if job.gpu_count > cluster.gpus().len() {
        return Err(SimulationError::JobDoesNotFit(job.id));
    }
    if job.memory_mb == 0 {
        return Err(SimulationError::InvalidJob {
            id: job.id,
            reason: "memory requirement must be positive",
        });
    }
    if job.steps == 0 {
        return Err(SimulationError::InvalidJob {
            id: job.id,
            reason: "step count must be positive",
        });
    }
    if job.compute_us_per_step == 0 {
        return Err(SimulationError::InvalidJob {
            id: job.id,
            reason: "compute time must be positive",
        });
    }
    if let Some(step) = job.failure_step
        && (step == 0 || step > job.steps)
    {
        return Err(SimulationError::InvalidJob {
            id: job.id,
            reason: "failure step must be within the job",
        });
    }

    let eligible_gpus = cluster
        .gpus()
        .iter()
        .filter(|gpu| gpu.memory_mb >= job.memory_mb)
        .count();
    if eligible_gpus < job.gpu_count {
        return Err(SimulationError::JobDoesNotFit(job.id));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn admit_jobs(
    cluster: &Cluster,
    pending: &mut VecDeque<PendingJob>,
    running: &mut Vec<RunningJob>,
    free: &mut BTreeSet<usize>,
    reports: &mut BTreeMap<usize, JobReport>,
    trace: &mut Vec<TraceEvent>,
    config: &SimulationConfig,
    now_us: u64,
    peak_allocated_gpus: &mut usize,
) -> Result<(), SimulationError> {
    loop {
        let Some(front) = pending.front() else {
            return Ok(());
        };
        let Some(placement) = choose_placement(cluster, free, &front.job, config.placement_policy)
        else {
            return Ok(());
        };
        let Some(pending_job) = pending.pop_front() else {
            return Ok(());
        };

        let communication = communication_cost(
            cluster,
            &placement,
            pending_job.job.gradient_bytes,
            config.network,
        );
        let finish_us = now_us
            .saturating_add(work_time(
                &pending_job.job,
                pending_job.resume_step,
                pending_job.job.steps,
                config.checkpoint_cost_us,
            ))
            .saturating_add(communication.total_us);
        let failure_us = if pending_job.failure_injected {
            None
        } else {
            pending_job.job.failure_step.map(|step| {
                now_us.saturating_add(work_time(
                    &pending_job.job,
                    pending_job.resume_step,
                    step,
                    config.checkpoint_cost_us,
                ))
            })
        };

        for id in &placement {
            free.remove(id);
        }
        *peak_allocated_gpus = (*peak_allocated_gpus).max(cluster.gpus().len() - free.len());

        let Some(report) = reports.get_mut(&pending_job.job.id) else {
            return Err(SimulationError::MissingReport(pending_job.job.id));
        };
        report.queue_wait_us = report
            .queue_wait_us
            .saturating_add(now_us.saturating_sub(pending_job.queued_at_us));
        report.attempts += 1;
        report.placements.push(placement.clone());
        if report.start_us.is_none() {
            report.start_us = Some(now_us);
        }

        trace.push(TraceEvent::JobStarted {
            job_id: pending_job.job.id,
            attempt: pending_job.attempt,
            at_us: now_us,
            placement: placement.clone(),
            resume_step: pending_job.resume_step,
        });
        running.push(RunningJob {
            pending: pending_job,
            placement,
            finish_us,
            failure_us,
            communication,
        });
    }
}

fn choose_placement(
    cluster: &Cluster,
    free: &BTreeSet<usize>,
    job: &Job,
    policy: PlacementPolicy,
) -> Option<Vec<usize>> {
    let eligible = free
        .iter()
        .copied()
        .filter(|id| {
            cluster
                .gpu(*id)
                .is_some_and(|gpu| gpu.memory_mb >= job.memory_mb)
        })
        .collect::<Vec<_>>();
    if eligible.len() < job.gpu_count {
        return None;
    }

    match policy {
        PlacementPolicy::Fifo => Some(eligible.into_iter().take(job.gpu_count).collect()),
        PlacementPolicy::TopologyAware => {
            let mut by_rack = BTreeMap::<usize, Vec<usize>>::new();
            for id in eligible {
                if let Some(gpu) = cluster.gpu(id) {
                    by_rack.entry(gpu.rack).or_default().push(id);
                }
            }

            if let Some(group) = by_rack.values().find(|group| group.len() >= job.gpu_count) {
                return Some(group.iter().copied().take(job.gpu_count).collect());
            }

            let mut candidates = by_rack.into_values().flatten().collect::<Vec<_>>();
            candidates.sort_by_key(|id| {
                cluster
                    .gpu(*id)
                    .map(|gpu| (gpu.rack, gpu.node, gpu.id))
                    .unwrap_or((usize::MAX, usize::MAX, *id))
            });
            Some(candidates.into_iter().take(job.gpu_count).collect())
        }
    }
}

fn handle_failure(
    pending: &mut VecDeque<PendingJob>,
    reports: &mut BTreeMap<usize, JobReport>,
    trace: &mut Vec<TraceEvent>,
    running: &RunningJob,
    config: &SimulationConfig,
    now_us: u64,
    total_retries: &mut usize,
) -> Result<(), SimulationError> {
    let job = &running.pending.job;
    let failed_step = job.failure_step.unwrap_or(running.pending.resume_step);
    let checkpoint_step = NonZeroU32::new(job.checkpoint_interval)
        .map_or(0, |interval| failed_step / interval.get() * interval.get());
    let replayed_steps = failed_step.saturating_sub(checkpoint_step);
    let Some(report) = reports.get_mut(&job.id) else {
        return Err(SimulationError::MissingReport(job.id));
    };
    report.retries += 1;
    report.checkpoint_replay_steps = report
        .checkpoint_replay_steps
        .saturating_add(replayed_steps);
    *total_retries += 1;
    trace.push(TraceEvent::JobFailed {
        job_id: job.id,
        attempt: running.pending.attempt,
        at_us: now_us,
        failed_step,
        checkpoint_step,
        replayed_steps,
    });

    if report.retries > config.max_retries {
        return Err(SimulationError::RetryLimitExceeded(job.id));
    }

    pending.push_back(PendingJob {
        job: job.clone(),
        queued_at_us: now_us,
        resume_step: checkpoint_step,
        attempt: running.pending.attempt + 1,
        failure_injected: true,
    });
    Ok(())
}

fn handle_completion(
    reports: &mut BTreeMap<usize, JobReport>,
    trace: &mut Vec<TraceEvent>,
    running: &RunningJob,
    now_us: u64,
    total_collective_time_us: &mut u64,
    total_cross_rack_bytes: &mut u64,
) -> Result<(), SimulationError> {
    let job_id = running.pending.job.id;
    let Some(report) = reports.get_mut(&job_id) else {
        return Err(SimulationError::MissingReport(job_id));
    };
    report.completed_step = running.pending.job.steps;
    report.collective_time_us = report
        .collective_time_us
        .saturating_add(running.communication.total_us);
    report.cross_rack_bytes = report
        .cross_rack_bytes
        .saturating_add(running.communication.cross_rack_bytes);
    report.end_us = Some(now_us);
    *total_collective_time_us =
        total_collective_time_us.saturating_add(running.communication.total_us);
    *total_cross_rack_bytes =
        total_cross_rack_bytes.saturating_add(running.communication.cross_rack_bytes);
    trace.push(TraceEvent::JobCompleted {
        job_id,
        attempt: running.pending.attempt,
        at_us: now_us,
    });
    Ok(())
}

fn release(free: &mut BTreeSet<usize>, placement: &[usize]) {
    free.extend(placement.iter().copied());
}

fn work_time(job: &Job, from_step: u32, to_step: u32, checkpoint_cost_us: u64) -> u64 {
    if to_step <= from_step {
        return 0;
    }
    let step_count = u64::from(to_step - from_step);
    let checkpoint_count = if let Some(interval) = NonZeroU32::new(job.checkpoint_interval) {
        u64::from(to_step / interval.get()).saturating_sub(u64::from(from_step / interval.get()))
    } else {
        0
    };
    step_count
        .saturating_mul(job.compute_us_per_step)
        .saturating_add(checkpoint_count.saturating_mul(checkpoint_cost_us))
}

fn empty_report(id: usize) -> JobReport {
    JobReport {
        id,
        placements: Vec::new(),
        queue_wait_us: 0,
        attempts: 0,
        retries: 0,
        completed_step: 0,
        checkpoint_replay_steps: 0,
        collective_time_us: 0,
        cross_rack_bytes: 0,
        start_us: None,
        end_us: None,
    }
}

fn percentile_95(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * 95).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn nanos_to_micros(nanoseconds: u128) -> u128 {
    nanoseconds.div_ceil(1_000)
}

fn saturating_u128_to_u64(value: u128) -> u64 {
    value.min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_cluster() -> Cluster {
        Cluster::uniform_interleaved(2, 1, 2, 16_000)
            .expect("fixture cluster must have valid topology")
    }

    fn fixture_job(id: usize, gpu_count: usize) -> Job {
        Job::new(id, gpu_count, 4_000, 4, 100, 4_000, 2)
    }

    #[test]
    fn gang_admission_does_not_start_partial_jobs() {
        let cluster = Cluster::uniform_interleaved(1, 1, 2, 16_000)
            .expect("fixture cluster must have valid topology");
        let result = simulate(
            &cluster,
            vec![fixture_job(1, 2), fixture_job(2, 1)],
            SimulationConfig::default(),
        )
        .expect("simulation should complete");

        let starts = result
            .trace
            .iter()
            .filter_map(|event| match event {
                TraceEvent::JobStarted { job_id, at_us, .. } => Some((*job_id, *at_us)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0].0, 1);
        assert!(starts[1].1 > starts[0].1);
        assert_eq!(result.free_gpu_count, cluster.gpus().len());
        assert_eq!(result.peak_allocated_gpus, 2);
    }

    #[test]
    fn topology_aware_placement_reduces_cross_rack_bytes() {
        let cluster = fixture_cluster();
        let job = fixture_job(1, 2);
        let fifo = simulate(
            &cluster,
            vec![job.clone()],
            SimulationConfig {
                placement_policy: PlacementPolicy::Fifo,
                ..SimulationConfig::default()
            },
        )
        .expect("FIFO simulation should complete");
        let topology = simulate(
            &cluster,
            vec![job],
            SimulationConfig {
                placement_policy: PlacementPolicy::TopologyAware,
                ..SimulationConfig::default()
            },
        )
        .expect("topology-aware simulation should complete");

        assert!(topology.cross_rack_bytes < fifo.cross_rack_bytes);
        assert_eq!(topology.jobs[0].placements, vec![vec![0, 2]]);
        assert_eq!(fifo.jobs[0].placements, vec![vec![0, 1]]);
    }

    #[test]
    fn communication_cost_is_monotonic_in_message_size() {
        let cluster = fixture_cluster();
        let placement = vec![0, 1];
        let small = communication_cost(&cluster, &placement, 1_000, NetworkModel::default());
        let large = communication_cost(&cluster, &placement, 2_000, NetworkModel::default());
        assert!(large.total_us > small.total_us);
        assert!(large.cross_rack_bytes > small.cross_rack_bytes);
    }

    #[test]
    fn repeated_runs_have_identical_trace_and_metrics() {
        let cluster = fixture_cluster();
        let jobs = vec![fixture_job(1, 2).with_failure_step(3), fixture_job(2, 1)];
        let first = simulate(&cluster, jobs.clone(), SimulationConfig::default())
            .expect("first simulation should complete");
        let second =
            simulate(&cluster, jobs, SimulationConfig::default()).expect("second simulation");
        assert_eq!(first, second);
    }

    #[test]
    fn failure_replays_from_latest_checkpoint_and_releases_resources() {
        let cluster = fixture_cluster();
        let result = simulate(
            &cluster,
            vec![fixture_job(1, 2).with_failure_step(3)],
            SimulationConfig::default(),
        )
        .expect("failure should be recoverable");
        let report = &result.jobs[0];

        assert_eq!(report.retries, 1);
        assert_eq!(report.checkpoint_replay_steps, 1);
        assert_eq!(report.completed_step, 4);
        assert_eq!(report.attempts, 2);
        assert_eq!(result.retries, 1);
        assert_eq!(result.free_gpu_count, cluster.gpus().len());
    }
}
