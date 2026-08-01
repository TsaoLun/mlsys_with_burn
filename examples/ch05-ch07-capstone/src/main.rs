use ch05_ch07_capstone::run_capstone;

// ANCHOR: capstone_main
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_capstone()?;

    println!(
        "train_samples={} validation_samples={} train_batches={} validation_batches={} \
initial_loss={:.6} final_loss={:.6} validation_loss={:.6} parameter_delta={:.6} \
record_tensors={} output_shape={:?} max_abs_error={:.6e} wrong_topology_rejected={}",
        report.train_samples,
        report.validation_samples,
        report.train_batches,
        report.validation_batches,
        report.initial_loss,
        report.final_loss,
        report.validation_loss,
        report.parameter_delta,
        report.record_tensors,
        report.output_shape,
        report.max_abs_error,
        report.wrong_topology_rejected,
    );

    Ok(())
}
// ANCHOR_END: capstone_main
