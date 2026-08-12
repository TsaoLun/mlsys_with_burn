use ch07_record_roundtrip::{inspect_burnpack_layout, run_round_trip, sample_record_bytes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_round_trip()?;

    println!(
        "record_tensors={} output_shape={:?} max_abs_error={:.6e}",
        report.record_tensors, report.output_shape, report.max_abs_error
    );

    let layout = inspect_burnpack_layout(&sample_record_bytes()?)?;
    println!(
        "burnpack magic={} version={} metadata_bytes={} data_section_start={} total_bytes={}",
        String::from_utf8_lossy(&layout.magic),
        layout.version,
        layout.metadata_size,
        layout.data_section_start,
        layout.total_len,
    );

    Ok(())
}
