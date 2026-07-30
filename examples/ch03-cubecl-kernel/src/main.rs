use ch03_cubecl_kernel::run_scale;
use cubecl::cpu::CpuRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_scale::<CpuRuntime>(&Default::default(), &[1.0, 2.0, 3.0, 4.0], 2)?;

    println!("runtime: {}", report.runtime);
    println!("input:   {:?}", report.input);
    println!("output:  {:?}", report.output);
    Ok(())
}
