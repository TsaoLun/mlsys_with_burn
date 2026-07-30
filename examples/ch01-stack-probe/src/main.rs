use std::process::ExitCode;

use ch01_stack_probe::probe_execution_stack;

fn main() -> ExitCode {
    match probe_execution_stack() {
        Ok(report) => {
            println!("snapshot: {}", report.snapshot);
            println!("device: {}", report.device);
            println!("default float dtype: {}", report.float_dtype);
            println!("default int dtype: {}", report.int_dtype);
            println!("autodiff enabled: {}", report.autodiff_enabled);
            println!("observed value after sync: {}", report.observed_value);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("stack probe failed: {error}");
            ExitCode::FAILURE
        }
    }
}
