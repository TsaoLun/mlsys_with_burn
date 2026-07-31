use ch08_rl_rollout::run_rollout;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_rollout(6, 4, 2)?;

    println!(
        "transitions={} buffer_len={} terminal_transitions={} \
         state_shape={:?} action_shape={:?} reward_shape={:?} done_shape={:?} \
         initial_right_q={:.4}",
        report.transitions_collected,
        report.buffer_len,
        report.terminal_transitions,
        report.sampled_state_shape,
        report.sampled_action_shape,
        report.sampled_reward_shape,
        report.sampled_done_shape,
        report.initial_right_q,
    );

    Ok(())
}
