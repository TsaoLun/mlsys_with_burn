//! GEMM 优化阶梯：从朴素三重循环到共享内存 tile。
//!
//! 默认特性只包含纯 Rust 部分：朴素参考实现、cache 分块实现和
//! 确定性矩阵生成，任何机器都能跑。`--features wgpu` 启用 CubeCL
//! Kernel（朴素与 tiled 两级）与计时协议，在本机 GPU 上复现第 3 章
//! 讲的「同一优化动作 → 可测差异」。

/// 行主序 `M×K` 与 `K×N` 的朴素矩阵乘参考实现。它是全书 GEMM 讨论
/// 的语义权威：所有优化版本都必须和它给出相同的数字。
// ANCHOR: reference
pub fn gemm_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A 应为 M×K 行主序");
    assert_eq!(b.len(), k * n, "B 应为 K×N 行主序");
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * b[i * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}
// ANCHOR_END: reference

/// 纯 Rust 的分块（tiled）GEMM：与共享内存 Kernel 相同的循环重排，
/// 用来在 CPU 上验证「按 tile 重排不改变结果」这一语义前提。
// ANCHOR: blocked
pub fn gemm_blocked(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, tile: usize) -> Vec<f32> {
    assert!(tile > 0, "tile 必须为正");
    let mut out = vec![0.0; m * n];
    for row_0 in (0..m).step_by(tile) {
        for col_0 in (0..n).step_by(tile) {
            for i_0 in (0..k).step_by(tile) {
                // 一个 (row_0, col_0) 输出 tile 沿 K 方向消费一对
                // A/B tile——与共享内存 Kernel 的 step 循环一一对应。
                for row in row_0..(row_0 + tile).min(m) {
                    for i in i_0..(i_0 + tile).min(k) {
                        let a_value = a[row * k + i];
                        for col in col_0..(col_0 + tile).min(n) {
                            out[row * n + col] += a_value * b[i * n + col];
                        }
                    }
                }
            }
        }
    }
    out
}
// ANCHOR_END: blocked

/// 确定性伪随机矩阵（线性同余生成器），避免外部依赖与不可复现输入。
pub fn deterministic_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // 取高位映射到 [-1, 1)，数值温和，便于误差比较。
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// 两个矩阵的最大绝对误差，用于把「结果一致」写成可断言的数字。
pub fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f32::max)
}

#[cfg(feature = "wgpu")]
pub mod gpu {
    //! CubeCL Kernel 与计时协议（`--features wgpu`）。

    use cubecl::prelude::*;

    /// 共享内存 tile 的边长；也是 cube 的 x/y 尺寸。16×16 = 256 个
    /// unit，处于常见后端 max_units_per_cube 的安全范围内。
    pub const TILE: u32 = 16;

    // ANCHOR: naive_kernel
    /// 阶梯第 1 级：一个 unit 负责一个输出元素，沿 K 做点积。
    /// 每个输入元素都直接从全局内存读取。
    #[cube(launch_unchecked)]
    fn gemm_naive_kernel<F: Float>(a: &[F], b: &[F], out: &mut [F], m: u32, n: u32, k: u32) {
        let row = CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y;
        let col = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
        if row < m && col < n {
            let mut acc = F::new(0.0);
            for i in 0..k {
                acc += a[(row * k + i) as usize] * b[(i * n + col) as usize];
            }
            out[(row * n + col) as usize] = acc;
        }
    }
    // ANCHOR_END: naive_kernel

    // ANCHOR: tiled_kernel
    /// 阶梯第 2 级：一个 cube 协作装载 A/B 的 tile 到共享内存，
    /// 每个全局元素装载一次、被 tile 内 16 个 unit 复用。
    #[cube(launch_unchecked)]
    fn gemm_tiled_kernel<F: Float>(
        a: &[F],
        b: &[F],
        out: &mut [F],
        m: u32,
        n: u32,
        k: u32,
        #[comptime] tile: u32,
    ) {
        let mut tile_a = Shared::<[F]>::new_slice(comptime!((tile * tile) as usize));
        let mut tile_b = Shared::<[F]>::new_slice(comptime!((tile * tile) as usize));

        let row = CUBE_POS_Y * tile + UNIT_POS_Y;
        let col = CUBE_POS_X * tile + UNIT_POS_X;
        let local = (UNIT_POS_Y * tile + UNIT_POS_X) as usize;
        let mut acc = F::new(0.0);

        // Kernel DSL 里保持显式算式（clippy 的 div_ceil 建议针对
        // host Rust；cube 展开只支持基本运算）。
        #[allow(clippy::manual_div_ceil)]
        let steps = (k + tile - 1) / tile;
        for step in 0..steps {
            // 协作装载：越界位置填零，正是正文说的「边界 tile 必须
            // 填零或被谓词屏蔽」。
            let a_col = step * tile + UNIT_POS_X;
            let b_row = step * tile + UNIT_POS_Y;
            let mut a_value = F::new(0.0);
            if row < m && a_col < k {
                a_value = a[(row * k + a_col) as usize];
            }
            let mut b_value = F::new(0.0);
            if b_row < k && col < n {
                b_value = b[(b_row * n + col) as usize];
            }
            tile_a[local] = a_value;
            tile_b[local] = b_value;
            // 第一道屏障：tile 对整个 cube 可见后才能开始计算。
            sync_cube();

            for i in 0..tile {
                acc += tile_a[(UNIT_POS_Y * tile + i) as usize]
                    * tile_b[(i * tile + UNIT_POS_X) as usize];
            }
            // 第二道屏障：算完才能让下一轮装载覆盖共享内存。
            sync_cube();
        }

        if row < m && col < n {
            out[(row * n + col) as usize] = acc;
        }
    }
    // ANCHOR_END: tiled_kernel

    /// 选择阶梯的某一级。
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Ladder {
        Naive,
        Tiled,
    }

    /// Runtime 的显示名称（在泛型上下文中取，绕开编译器类型参数）。
    pub fn runtime_name<R: Runtime>(device: &R::Device) -> String {
        R::name(&R::client(device)).to_owned()
    }

    /// 在 Runtime `R` 上执行一次 GEMM 并读回结果。
    pub fn run_gemm<R: Runtime>(
        device: &R::Device,
        ladder: Ladder,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let client = R::client(device);
        let handles = upload::<R>(&client, a, b, m, n, k)?;
        launch::<R>(&client, ladder, &handles, m, n, k)?;
        let bytes = client
            .read_one(handles.out.clone())
            .map_err(|error| format!("读取输出失败：{error:?}"))?;
        Ok(f32::from_bytes(&bytes).to_vec())
    }

    /// 计时协议：预热一次（含 JIT 编译），随后提交 `iters` 次 launch，
    /// 用一次输出读回作为完成边界，返回平均每次 launch 的微秒数。
    /// 结果只用于同一设备上两级阶梯的相对比较。
    // ANCHOR: timing
    pub fn time_gemm<R: Runtime>(
        device: &R::Device,
        ladder: Ladder,
        size: usize,
        iters: usize,
    ) -> Result<f64, String> {
        let client = R::client(device);
        let a = super::deterministic_matrix(size, size, 11);
        let b = super::deterministic_matrix(size, size, 23);
        let handles = upload::<R>(&client, &a, &b, size, size, size)?;

        // 预热：触发 JIT 编译与首次分配，并等待完成。
        launch::<R>(&client, ladder, &handles, size, size, size)?;
        client
            .read_one(handles.out.clone())
            .map_err(|error| format!("预热读回失败：{error:?}"))?;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            launch::<R>(&client, ladder, &handles, size, size, size)?;
        }
        // 读回既取数据也充当同步点：所有已提交 launch 必须先完成。
        client
            .read_one(handles.out.clone())
            .map_err(|error| format!("计时读回失败：{error:?}"))?;
        Ok(start.elapsed().as_secs_f64() * 1e6 / iters as f64)
    }
    // ANCHOR_END: timing

    struct GemmHandles {
        a: cubecl::server::Handle,
        b: cubecl::server::Handle,
        out: cubecl::server::Handle,
    }

    fn upload<R: Runtime>(
        client: &ComputeClient<R>,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<GemmHandles, String> {
        if a.len() != m * k || b.len() != k * n {
            return Err("输入长度与 M/N/K 不一致".to_owned());
        }
        if m == 0 || n == 0 || k == 0 {
            return Err("空矩阵不进入 launch 路径".to_owned());
        }
        Ok(GemmHandles {
            a: client.create_from_slice(f32::as_bytes(a)),
            b: client.create_from_slice(f32::as_bytes(b)),
            out: client.empty(m * n * std::mem::size_of::<f32>()),
        })
    }

    fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        ladder: Ladder,
        handles: &GemmHandles,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let cube_dim = CubeDim::new_2d(TILE, TILE);
        let cubes_x = u32::try_from(n.div_ceil(TILE as usize)).map_err(|_| "N 过大".to_owned())?;
        let cubes_y = u32::try_from(m.div_ceil(TILE as usize)).map_err(|_| "M 过大".to_owned())?;
        let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);
        let (m, n, k) = (m as u32, n as u32, k as u32);

        // SAFETY: 三个 BufferArg 描述的分配长度恰为 M×K、K×N、M×N 个
        // f32；两个 Kernel 都在写/读前用 row/col 与 M/N/K 做了越界
        // 判断，tile 装载对越界位置填零。
        unsafe {
            match ladder {
                Ladder::Naive => gemm_naive_kernel::launch_unchecked::<f32, R>(
                    client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(handles.a.clone(), (m * k) as usize),
                    BufferArg::from_raw_parts(handles.b.clone(), (k * n) as usize),
                    BufferArg::from_raw_parts(handles.out.clone(), (m * n) as usize),
                    m,
                    n,
                    k,
                ),
                Ladder::Tiled => gemm_tiled_kernel::launch_unchecked::<f32, R>(
                    client,
                    cube_count,
                    cube_dim,
                    BufferArg::from_raw_parts(handles.a.clone(), (m * k) as usize),
                    BufferArg::from_raw_parts(handles.b.clone(), (k * n) as usize),
                    BufferArg::from_raw_parts(handles.out.clone(), (m * n) as usize),
                    m,
                    n,
                    k,
                    TILE,
                ),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 分块实现与朴素参考在非整除形状上逐元素一致：循环重排不改变
    /// 语义（浮点加法顺序此处按行遍历保持一致，因此可以精确比较）。
    #[test]
    fn blocked_matches_reference_on_ragged_shapes() {
        for (m, n, k) in [(1, 1, 1), (5, 7, 3), (17, 9, 33), (32, 32, 32)] {
            let a = deterministic_matrix(m, k, 1);
            let b = deterministic_matrix(k, n, 2);
            let reference = gemm_reference(&a, &b, m, n, k);
            let blocked = gemm_blocked(&a, &b, m, n, k, 4);
            assert!(
                max_abs_diff(&reference, &blocked) < 1e-5,
                "tile 重排后结果偏离参考：M={m} N={n} K={k}"
            );
        }
    }

    /// 单位矩阵乘法给出可手算的精确结果。
    #[test]
    fn identity_multiplication_is_exact() {
        let m = 4;
        let mut identity = vec![0.0; m * m];
        for i in 0..m {
            identity[i * m + i] = 1.0;
        }
        let a = deterministic_matrix(m, m, 7);
        let product = gemm_reference(&a, &identity, m, m, m);
        assert_eq!(product, a);
    }

    /// 确定性矩阵生成必须可复现，且不同 seed 产生不同矩阵。
    #[test]
    fn deterministic_matrix_is_reproducible() {
        let first = deterministic_matrix(8, 8, 42);
        let second = deterministic_matrix(8, 8, 42);
        let other = deterministic_matrix(8, 8, 43);
        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[cfg(feature = "wgpu")]
    mod gpu_tests {
        use super::super::gpu::{Ladder, run_gemm};
        use super::super::*;
        use cubecl::wgpu::WgpuRuntime;

        /// 两级 Kernel 都必须和 host 参考一致，含非整除形状
        /// （17、33 会触发 tile 的填零与谓词分支）。
        #[test]
        fn kernels_match_host_reference() {
            for (m, n, k) in [(16, 16, 16), (17, 9, 33), (64, 48, 80)] {
                let a = deterministic_matrix(m, k, 5);
                let b = deterministic_matrix(k, n, 6);
                let reference = gemm_reference(&a, &b, m, n, k);
                for ladder in [Ladder::Naive, Ladder::Tiled] {
                    let device = Default::default();
                    let output = run_gemm::<WgpuRuntime>(&device, ladder, &a, &b, m, n, k)
                        .expect("WGPU GEMM 应可执行");
                    let diff = max_abs_diff(&reference, &output);
                    assert!(
                        diff < 1e-3,
                        "{ladder:?} 与 host 参考最大误差 {diff}（M={m} N={n} K={k}）"
                    );
                }
            }
        }

        /// 两级 Kernel 彼此一致：优化不允许改变结果。
        #[test]
        fn naive_and_tiled_agree() {
            let (m, n, k) = (33, 65, 47);
            let a = deterministic_matrix(m, k, 9);
            let b = deterministic_matrix(k, n, 10);
            let device = Default::default();
            let naive = run_gemm::<WgpuRuntime>(&device, Ladder::Naive, &a, &b, m, n, k)
                .expect("naive 应可执行");
            let tiled = run_gemm::<WgpuRuntime>(&device, Ladder::Tiled, &a, &b, m, n, k)
                .expect("tiled 应可执行");
            assert!(max_abs_diff(&naive, &tiled) < 1e-3);
        }
    }
}
