use ch03_cubecl_kernel::run_scale;
use cubecl::cpu::CpuRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = [1.0, 2.0, 3.0, 4.0];
    let report = run_scale::<CpuRuntime>(&Default::default(), &input, 2)?;

    println!("runtime: {}", report.runtime);
    println!("input:   {:?}", report.input);
    println!("output:  {:?}", report.output);

    // Optional GPU path: `cargo run -p ch03-cubecl-kernel --features wgpu`.
    // Runs the same kernel through the CubeCL WGPU runtime and checks it
    // against the same host reference. Requires a system GPU driver
    // (Metal/Vulkan/DX12); the default CPU path is unchanged.
    #[cfg(feature = "wgpu")]
    {
        use ch03_cubecl_kernel::scale_reference;
        use cubecl::wgpu::WgpuRuntime;

        let gpu_report = run_scale::<WgpuRuntime>(&Default::default(), &input, 2)?;
        println!("runtime: {}", gpu_report.runtime);
        println!("output:  {:?}", gpu_report.output);
        assert_eq!(gpu_report.output, scale_reference(&input, 2));
        println!("wgpu output matches host reference");
    }

    Ok(())
}
