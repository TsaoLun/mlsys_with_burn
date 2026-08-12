use ch08_rl_rollout::{run_replay_driven, run_rollout};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_rollout(6, 4, 2)?;

    println!(
        "phase=online transitions={} buffer_len={} done_transitions={} \
         truncated_transitions={} state_shape={:?} action_shape={:?} \
         reward_shape={:?} done_shape={:?} initial_right_q={:.4}",
        report.transitions_collected,
        report.buffer_len,
        report.done_transitions,
        report.truncated_transitions,
        report.sampled_state_shape,
        report.sampled_action_shape,
        report.sampled_reward_shape,
        report.sampled_done_shape,
        report.initial_right_q,
    );

    let unit = run_replay_driven(6, 1, 1, 8)?;
    println!(
        "phase=replay capacity={} sample={} rounds={} updates={} buffer_len={} \
         initial_right_q={:.4}",
        1, 1, 8, unit.updates_applied, unit.buffer_len, unit.initial_right_q,
    );

    let windowed = run_replay_driven(6, 6, 2, 10)?;
    println!(
        "phase=replay capacity={} sample={} rounds={} updates={} buffer_len={} \
         initial_right_q={:.4}（随机采样，数值会变）",
        6, 2, 10, windowed.updates_applied, windowed.buffer_len, windowed.initial_right_q,
    );

    Ok(())
}
