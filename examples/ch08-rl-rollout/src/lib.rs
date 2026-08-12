use burn::{
    rl::{Environment, StepResult, TransitionBuffer},
    tensor::{Device, Tensor},
};
use std::fmt::{Display, Formatter};

/// A small deterministic state used by the CPU rollout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterState {
    /// Position in the one-dimensional environment.
    pub position: i32,
    /// Number of steps since the last reset.
    pub step: usize,
}

/// Actions accepted by [`CounterEnv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterAction {
    /// Move one position to the left and receive no reward.
    Left,
    /// Move one position to the right and receive reward one.
    Right,
}

/// A deterministic environment with both terminal and truncation boundaries.
#[derive(Clone, Debug, Default)]
pub struct CounterEnv {
    position: i32,
    step: usize,
}

impl CounterEnv {
    /// Create an environment at its initial state.
    pub fn new() -> Self {
        Self::default()
    }
}

// ANCHOR: environment
impl Environment for CounterEnv {
    type State = CounterState;
    type Action = CounterAction;

    const MAX_STEPS: usize = 4;

    fn state(&self) -> Self::State {
        CounterState {
            position: self.position,
            step: self.step,
        }
    }

    fn step(&mut self, action: Self::Action) -> StepResult<Self::State> {
        self.position += match action {
            CounterAction::Left => -1,
            CounterAction::Right => 1,
        };
        self.step += 1;

        let done = self.position >= 2;
        let truncated = !done && self.step >= Self::MAX_STEPS;
        let reward = if matches!(action, CounterAction::Right) {
            1.0
        } else {
            0.0
        };

        StepResult {
            next_state: self.state(),
            reward,
            done,
            truncated,
        }
    }

    fn reset(&mut self) {
        self.position = 0;
        self.step = 0;
    }
}
// ANCHOR_END: environment

/// Errors detected before sampling a rollout.
#[derive(Debug, PartialEq, Eq)]
pub enum RolloutError {
    /// No environment steps were requested.
    NoSteps,
    /// A circular buffer needs a positive capacity.
    ZeroCapacity,
    /// A replay batch needs at least one item.
    ZeroSampleBatch,
    /// Replay-driven learning needs at least one update round.
    ZeroUpdateRounds,
    /// The requested replay batch is larger than the collected buffer.
    SampleTooLarge {
        /// Number of requested samples.
        requested: usize,
        /// Number of available samples.
        available: usize,
    },
    /// A stored tensor could not be read back to the host.
    Readback(String),
}

impl Display for RolloutError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSteps => write!(formatter, "rollout requires at least one step"),
            Self::ZeroCapacity => write!(formatter, "replay capacity must be positive"),
            Self::ZeroSampleBatch => write!(formatter, "sample batch must be positive"),
            Self::ZeroUpdateRounds => write!(formatter, "replay learning needs a round"),
            Self::SampleTooLarge {
                requested,
                available,
            } => write!(
                formatter,
                "sample batch {requested} exceeds available transitions {available}"
            ),
            Self::Readback(error) => write!(formatter, "tensor readback failed: {error}"),
        }
    }
}

impl std::error::Error for RolloutError {}

/// Observable values produced by the rollout and update loop.
#[derive(Debug, PartialEq)]
pub struct RolloutReport {
    /// Number of environment steps collected.
    pub transitions_collected: usize,
    /// Number of transitions retained by the circular buffer.
    pub buffer_len: usize,
    /// Number of transitions terminated by the environment itself.
    pub done_transitions: usize,
    /// Number of transitions cut off by the external step limit.
    pub truncated_transitions: usize,
    /// Shape of sampled state tensors.
    pub sampled_state_shape: [usize; 2],
    /// Shape of sampled action tensors.
    pub sampled_action_shape: [usize; 2],
    /// Shape of sampled reward tensors.
    pub sampled_reward_shape: [usize; 2],
    /// Shape of sampled done tensors.
    pub sampled_done_shape: [usize; 2],
    /// Q value for the initial state and right action after online TD updates.
    pub initial_right_q: f32,
}

/// Compute one tabular Q-learning target.
// ANCHOR: td_target
pub fn td_target(reward: f32, next_max_q: f32, done: bool, gamma: f32) -> f32 {
    reward + if done { 0.0 } else { gamma * next_max_q }
}
// ANCHOR_END: td_target

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MockPolicy {
    pub version: u64,
    pub action: i32,
}

impl MockPolicy {
    pub fn action_for(&self, _state: CounterState) -> i32 {
        self.action
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicySampleMetadata {
    pub behavior_version: u64,
    pub target_version: u64,
}

pub fn policy_is_fresh(metadata: PolicySampleMetadata, max_lag: u64) -> bool {
    metadata
        .target_version
        .checked_sub(metadata.behavior_version)
        .is_some_and(|lag| lag <= max_lag)
}

#[derive(Clone, Debug, PartialEq)]
pub struct JointTransition {
    pub actions: Vec<i32>,
    pub rewards: Vec<f32>,
}

pub fn joint_transition(actions: [i32; 2], rewards: [f32; 2]) -> JointTransition {
    JointTransition {
        actions: actions.into(),
        rewards: rewards.into(),
    }
}

fn state_tensor(state: CounterState, device: &Device) -> Tensor<2> {
    Tensor::from_data([[state.position as f32, state.step as f32]], device)
}

fn action_tensor(action: CounterAction, device: &Device) -> Tensor<2> {
    let value = match action {
        CounterAction::Left => -1.0,
        CounterAction::Right => 1.0,
    };
    Tensor::from_data([[value]], device)
}

fn state_index(position: i32) -> usize {
    position.clamp(0, 2) as usize
}

/// Deterministic action schedule shared by the online and replay-driven paths.
///
/// The first episode is `Right, Right`, reaching position 2 and producing a
/// natural `done`; later episodes keep alternating and eventually hit the
/// truncation limit.
fn scheduled_action(index: usize) -> CounterAction {
    match index {
        1 => CounterAction::Right,
        _ if index.is_multiple_of(2) => CounterAction::Right,
        _ => CounterAction::Left,
    }
}

fn action_index(action: CounterAction) -> usize {
    match action {
        CounterAction::Left => 0,
        CounterAction::Right => 1,
    }
}

/// Collect transitions, sample replay data for shape inspection, and apply online TD updates.
///
/// The TD updates use transitions as they are collected; the replay batch is sampled
/// after collection and is intentionally not fed back into the table. Keeping those
/// two paths separate prevents this small example from pretending to be a full
/// off-policy DQN learner.
// ANCHOR: rollout
pub fn run_rollout(
    steps: usize,
    capacity: usize,
    sample_size: usize,
) -> Result<RolloutReport, RolloutError> {
    if steps == 0 {
        return Err(RolloutError::NoSteps);
    }
    if capacity == 0 {
        return Err(RolloutError::ZeroCapacity);
    }
    if sample_size == 0 {
        return Err(RolloutError::ZeroSampleBatch);
    }

    let available = steps.min(capacity);
    if sample_size > available {
        return Err(RolloutError::SampleTooLarge {
            requested: sample_size,
            available,
        });
    }

    let device = Device::flex();
    let mut environment = CounterEnv::new();
    let mut buffer = TransitionBuffer::<Tensor<2>, Tensor<2>>::new(capacity, &device);
    let mut q_values = [[0.0f32; 2]; 3];
    let gamma = 0.9;
    let learning_rate = 0.5;
    let mut done_transitions = 0;
    let mut truncated_transitions = 0;

    for index in 0..steps {
        let state = environment.state();
        let action = scheduled_action(index);
        let result = environment.step(action);
        if result.done {
            done_transitions += 1;
        }
        if result.truncated {
            truncated_transitions += 1;
        }
        let terminal = result.done || result.truncated;

        buffer.push(
            state_tensor(state, &device),
            state_tensor(result.next_state, &device),
            action_tensor(action, &device),
            result.reward as f32,
            terminal,
        );

        let state_row = state_index(state.position);
        let next_row = state_index(result.next_state.position);
        let action_column = action_index(action);
        let next_max_q = q_values[next_row]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let target = td_target(result.reward as f32, next_max_q, terminal, gamma);
        let current = q_values[state_row][action_column];
        q_values[state_row][action_column] = current + learning_rate * (target - current);

        if terminal {
            environment.reset();
        }
    }

    let batch = buffer.sample(sample_size);

    Ok(RolloutReport {
        transitions_collected: steps,
        buffer_len: buffer.len(),
        done_transitions,
        truncated_transitions,
        sampled_state_shape: batch.states.dims(),
        sampled_action_shape: batch.actions.dims(),
        sampled_reward_shape: batch.rewards.dims(),
        sampled_done_shape: batch.dones.dims(),
        initial_right_q: q_values[0][1],
    })
}
// ANCHOR_END: rollout

/// Observable values produced by replay-driven updates.
#[derive(Debug, PartialEq)]
pub struct ReplayUpdateReport {
    /// Number of environment steps collected before learning.
    pub transitions_collected: usize,
    /// Number of transitions retained by the circular buffer.
    pub buffer_len: usize,
    /// Number of sampled transitions applied to the table.
    pub updates_applied: usize,
    /// Q value for the initial state and right action after replay-driven updates.
    pub initial_right_q: f32,
}

/// Collect transitions without learning, then update the table from replay samples.
///
/// Phase 1 stores the same deterministic rollout used by [`run_rollout`], but
/// leaves the Q table untouched. Phase 2 repeatedly samples the buffer and
/// applies the same TD rule to the sampled rows. With `capacity = 1` the ring
/// retains only the newest transition, so the learner can never see the initial
/// state again: this makes the effect of replay capacity directly observable.
pub fn run_replay_driven(
    steps: usize,
    capacity: usize,
    sample_size: usize,
    update_rounds: usize,
) -> Result<ReplayUpdateReport, RolloutError> {
    if steps == 0 {
        return Err(RolloutError::NoSteps);
    }
    if capacity == 0 {
        return Err(RolloutError::ZeroCapacity);
    }
    if sample_size == 0 {
        return Err(RolloutError::ZeroSampleBatch);
    }
    if update_rounds == 0 {
        return Err(RolloutError::ZeroUpdateRounds);
    }
    let available = steps.min(capacity);
    if sample_size > available {
        return Err(RolloutError::SampleTooLarge {
            requested: sample_size,
            available,
        });
    }

    // Phase 1: collect only. No Q update happens while the environment runs.
    let buffer = collect_rollout(steps, capacity)?;

    let gamma = 0.9;
    let learning_rate = 0.5;
    let mut q_values = [[0.0f32; 2]; 3];

    // ANCHOR: replay_update
    for _ in 0..update_rounds {
        let batch = buffer.sample(sample_size);
        let states = read_rows(&batch.states)?;
        let next_states = read_rows(&batch.next_states)?;
        let actions = read_rows(&batch.actions)?;
        let rewards = read_rows(&batch.rewards)?;
        let dones = read_rows(&batch.dones)?;

        for row in 0..sample_size {
            let state_row = state_index(states[row][0] as i32);
            let next_row = state_index(next_states[row][0] as i32);
            let action_column = if actions[row][0] < 0.0 { 0 } else { 1 };
            let next_max_q = q_values[next_row]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let target = td_target(rewards[row][0], next_max_q, dones[row][0] != 0.0, gamma);
            let current = q_values[state_row][action_column];
            q_values[state_row][action_column] = current + learning_rate * (target - current);
        }
    }
    // ANCHOR_END: replay_update

    Ok(ReplayUpdateReport {
        transitions_collected: steps,
        buffer_len: buffer.len(),
        updates_applied: update_rounds * sample_size,
        initial_right_q: q_values[0][1],
    })
}

/// Run the deterministic rollout without any learning and return the buffer.
fn collect_rollout(
    steps: usize,
    capacity: usize,
) -> Result<TransitionBuffer<Tensor<2>, Tensor<2>>, RolloutError> {
    if capacity == 0 {
        return Err(RolloutError::ZeroCapacity);
    }
    let device = Device::flex();
    let mut environment = CounterEnv::new();
    let mut buffer = TransitionBuffer::<Tensor<2>, Tensor<2>>::new(capacity, &device);

    for index in 0..steps {
        let state = environment.state();
        let action = scheduled_action(index);
        let result = environment.step(action);
        buffer.push(
            state_tensor(state, &device),
            state_tensor(result.next_state, &device),
            action_tensor(action, &device),
            result.reward as f32,
            result.done || result.truncated,
        );
        if result.done || result.truncated {
            environment.reset();
        }
    }

    Ok(buffer)
}

/// Read a `[rows, columns]` tensor back into per-row vectors.
fn read_rows(tensor: &Tensor<2>) -> Result<Vec<Vec<f32>>, RolloutError> {
    let [rows, columns] = tensor.dims();
    let flat = tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| RolloutError::Readback(error.to_string()))?;
    debug_assert_eq!(flat.len(), rows * columns);
    Ok(flat.chunks(columns).map(Vec::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_resets_after_truncation() {
        let mut environment = CounterEnv::new();

        for _ in 0..3 {
            let result = environment.step(CounterAction::Left);
            assert!(!result.done);
            assert!(!result.truncated);
        }
        let result = environment.step(CounterAction::Left);
        assert!(!result.done);
        assert!(result.truncated);

        environment.reset();
        assert_eq!(
            environment.state(),
            CounterState {
                position: 0,
                step: 0
            }
        );
    }

    #[test]
    fn environment_reports_natural_done_before_step_limit() {
        let mut environment = CounterEnv::new();
        let first = environment.step(CounterAction::Right);
        let second = environment.step(CounterAction::Right);

        assert!(!first.done);
        assert!(!first.truncated);
        assert!(second.done);
        assert!(!second.truncated);
    }

    #[test]
    fn rollout_samples_shapes_and_updates_q_value() {
        let report = run_rollout(6, 4, 2).expect("valid deterministic rollout");

        assert_eq!(report.transitions_collected, 6);
        assert_eq!(report.buffer_len, 4);
        assert_eq!(report.done_transitions, 1);
        assert_eq!(report.truncated_transitions, 1);
        assert_eq!(report.sampled_state_shape, [2, 2]);
        assert_eq!(report.sampled_action_shape, [2, 1]);
        assert_eq!(report.sampled_reward_shape, [2, 1]);
        assert_eq!(report.sampled_done_shape, [2, 1]);
        assert!(
            (report.initial_right_q - 1.2125).abs() < 1e-6,
            "unexpected fixed Q reference: {}",
            report.initial_right_q
        );
    }

    #[test]
    fn td_target_stops_bootstrapping_at_terminal_state() {
        assert_eq!(td_target(1.0, 5.0, true, 0.9), 1.0);
        assert_eq!(td_target(1.0, 5.0, false, 0.9), 5.5);
    }

    #[test]
    fn invalid_rollout_configuration_is_reported() {
        assert_eq!(run_rollout(0, 4, 1), Err(RolloutError::NoSteps));
        assert_eq!(run_rollout(1, 0, 1), Err(RolloutError::ZeroCapacity));
        assert_eq!(run_rollout(1, 4, 0), Err(RolloutError::ZeroSampleBatch));
        assert_eq!(
            run_rollout(1, 4, 2),
            Err(RolloutError::SampleTooLarge {
                requested: 2,
                available: 1
            })
        );
    }

    #[test]
    fn unit_capacity_retains_only_the_latest_aligned_transition() {
        let buffer = collect_rollout(6, 1).expect("valid collection");

        assert_eq!(buffer.len(), 1);
        let batch = buffer.sample(1);
        // The last scheduled transition is state=[1,3] --Left--> next=[0,4]
        // with reward 0. It hits the step limit, so the stored replay `done`
        // flag is 1 even though the environment reported truncated, not done.
        // Sampling a 1-element ring is deterministic, so every column must
        // come from that same retained transition.
        assert_eq!(
            read_rows(&batch.states).expect("states"),
            vec![vec![1.0, 3.0]]
        );
        assert_eq!(
            read_rows(&batch.next_states).expect("next_states"),
            vec![vec![0.0, 4.0]]
        );
        assert_eq!(
            read_rows(&batch.actions).expect("actions"),
            vec![vec![-1.0]]
        );
        assert_eq!(read_rows(&batch.rewards).expect("rewards"), vec![vec![0.0]]);
        assert_eq!(read_rows(&batch.dones).expect("dones"), vec![vec![1.0]]);
    }

    #[test]
    fn replay_driven_with_unit_capacity_cannot_update_initial_q() {
        let report = run_replay_driven(6, 1, 1, 8).expect("valid replay-driven run");

        assert_eq!(report.transitions_collected, 6);
        assert_eq!(report.buffer_len, 1);
        assert_eq!(report.updates_applied, 8);
        // The retained transition touches state 1 / Left only, and its target is
        // zero; the initial state value stays at exactly zero.
        assert_eq!(report.initial_right_q, 0.0);
    }

    #[test]
    fn replay_driven_rejects_invalid_configuration() {
        assert_eq!(run_replay_driven(0, 1, 1, 1), Err(RolloutError::NoSteps));
        assert_eq!(
            run_replay_driven(1, 0, 1, 1),
            Err(RolloutError::ZeroCapacity)
        );
        assert_eq!(
            run_replay_driven(1, 1, 0, 1),
            Err(RolloutError::ZeroSampleBatch)
        );
        assert_eq!(
            run_replay_driven(1, 1, 1, 0),
            Err(RolloutError::ZeroUpdateRounds)
        );
        assert_eq!(
            run_replay_driven(4, 2, 3, 1),
            Err(RolloutError::SampleTooLarge {
                requested: 3,
                available: 2
            })
        );
    }

    #[test]
    fn protocol_card_checks_policy_freshness_and_joint_credit() {
        let policy = MockPolicy {
            version: 7,
            action: 1,
        };
        assert_eq!(
            policy.action_for(CounterState {
                position: 0,
                step: 0
            }),
            1
        );
        assert!(policy_is_fresh(
            PolicySampleMetadata {
                behavior_version: 5,
                target_version: 7
            },
            2
        ));
        assert!(!policy_is_fresh(
            PolicySampleMetadata {
                behavior_version: 4,
                target_version: 7
            },
            2
        ));

        assert_eq!(
            joint_transition([0, 1], [1.0, -1.0]),
            JointTransition {
                actions: vec![0, 1],
                rewards: vec![1.0, -1.0]
            }
        );
    }
}
