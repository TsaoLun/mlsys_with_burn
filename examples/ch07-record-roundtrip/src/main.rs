use ch07_record_roundtrip::run_round_trip;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_round_trip()?;

    println!(
        "record_tensors={} output_shape={:?} max_abs_error={:.6e}",
        report.record_tensors, report.output_shape, report.max_abs_error
    );

    Ok(())
}
