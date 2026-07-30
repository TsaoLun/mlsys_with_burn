use cubecl::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct KernelReport {
    pub runtime: String,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Debug)]
pub struct KernelError(String);

impl Display for KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for KernelError {}

// ANCHOR: kernel
#[cube(launch_unchecked)]
fn scale_kernel<F: Float>(input: &[F], output: &mut [F], #[comptime] scale: u32) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * F::new(scale as f32);
    }
}
// ANCHOR_END: kernel

pub fn scale_reference(input: &[f32], scale: u32) -> Vec<f32> {
    input.iter().map(|value| value * scale as f32).collect()
}

// ANCHOR: host
pub fn run_scale<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    scale: u32,
) -> Result<KernelReport, KernelError> {
    let client = R::client(device);
    if input.is_empty() {
        return Ok(KernelReport {
            runtime: R::name(&client).to_owned(),
            input: Vec::new(),
            output: Vec::new(),
        });
    }

    let cube_dim = CubeDim::new(&client, input.len());
    let units_per_cube = cube_dim.num_elems() as usize;
    let cube_count = u32::try_from(input.len().div_ceil(units_per_cube))
        .map_err(|_| KernelError("输入过大，CubeCount.x 无法用 u32 表示".to_owned()))?;
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    // SAFETY: both BufferArg values describe allocations for exactly
    // `input.len()` f32 elements. The kernel guards ABSOLUTE_POS before access.
    unsafe {
        scale_kernel::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(cube_count, 1, 1),
            cube_dim,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
            scale,
        );
    }

    let bytes = client
        .read_one(output_handle)
        .map_err(|error| KernelError(format!("读取 CubeCL 输出失败：{error:?}")))?;

    Ok(KernelReport {
        runtime: R::name(&client).to_owned(),
        input: input.to_vec(),
        output: f32::from_bytes(&bytes).to_vec(),
    })
}
// ANCHOR_END: host

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cpu::CpuRuntime;

    #[test]
    fn cpu_kernel_matches_reference() {
        let input = [-2.0, -0.5, 0.0, 1.5, 4.0];
        let report =
            run_scale::<CpuRuntime>(&Default::default(), &input, 3).expect("CPU Kernel 应可执行");

        assert_eq!(report.runtime, "cpu");
        assert_eq!(report.output, scale_reference(&input, 3));
    }

    #[test]
    fn empty_input_does_not_launch_an_invalid_cube() {
        let report =
            run_scale::<CpuRuntime>(&Default::default(), &[], 3).expect("空输入应直接返回");

        assert!(report.output.is_empty());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_kernel_matches_reference_when_requested() {
        use cubecl::wgpu::WgpuRuntime;

        let input = [-2.0, -0.5, 0.0, 1.5, 4.0];
        let report =
            run_scale::<WgpuRuntime>(&Default::default(), &input, 3).expect("WGPU Kernel 应可执行");

        assert_eq!(report.output, scale_reference(&input, 3));
    }
}
