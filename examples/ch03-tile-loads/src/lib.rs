//! Host-side tiling load model for chapter 3 (not a CubeCL shared-memory Kernel).

use std::error::Error;
use std::fmt::{Display, Formatter};

/// 朴素与 tiled GEMM 的全局加载次数模型。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLoadReport {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub naive_loads: usize,
    pub tiled_loads: usize,
}

#[derive(Debug)]
pub struct TileModelError(String);

impl Display for TileModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TileModelError {}

// ANCHOR: tile_loads
/// 统计理想化全局加载次数：朴素路径每个输出读满 K；tiled 路径按 tile 复用。
///
/// 这不是 CubeCL 共享内存实验。它只比较“加载次数”数量级，忽略 bank conflict、
/// 边界 tile、同步和缓存命中。
pub fn tile_load_counts(
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) -> Result<TileLoadReport, TileModelError> {
    if [m, n, k, tile_m, tile_n, tile_k].contains(&0) {
        return Err(TileModelError("矩阵与 tile 尺寸必须为正".to_owned()));
    }
    if !m.is_multiple_of(tile_m) || !n.is_multiple_of(tile_n) || !k.is_multiple_of(tile_k) {
        return Err(TileModelError(
            "本教学模型要求尺寸可被 tile 整除".to_owned(),
        ));
    }

    let naive_loads = m * n * (k + k);
    let tiles_m = m / tile_m;
    let tiles_n = n / tile_n;
    let stages = k / tile_k;
    let loads_per_stage = tile_m * tile_k + tile_k * tile_n;
    let tiled_loads = tiles_m * tiles_n * stages * loads_per_stage;

    Ok(TileLoadReport {
        m,
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
        naive_loads,
        tiled_loads,
    })
}
// ANCHOR_END: tile_loads

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_loads_are_fewer_than_naive_for_square_tiles() {
        let report = tile_load_counts(16, 16, 16, 8, 8, 8).expect("整除尺寸应成功");

        // naive: 16*16*(16+16) = 8192
        // tiled: 2*2*2*(8*8 + 8*8) = 1024
        assert_eq!(report.naive_loads, 8192);
        assert_eq!(report.tiled_loads, 1024);
        assert!(report.tiled_loads < report.naive_loads);
    }

    #[test]
    fn rejects_non_divisible_tiles() {
        assert!(tile_load_counts(16, 16, 16, 8, 8, 7).is_err());
    }
}
