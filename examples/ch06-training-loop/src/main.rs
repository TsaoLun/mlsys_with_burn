use ch06_training_loop::run_training;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_training(40, 0.1)?;

    for (step, loss) in report.losses.iter().enumerate() {
        println!("step={} loss={loss:.6}", step + 1);
    }
    println!(
        "initial_loss={:.6} final_loss={:.6} parameter_delta={:.6}",
        report.initial_loss, report.final_loss, report.parameter_delta
    );

    Ok(())
}
