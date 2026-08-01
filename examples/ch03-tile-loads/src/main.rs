use ch03_tile_loads::tile_load_counts;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loads = tile_load_counts(16, 16, 16, 8, 8, 8)?;
    println!(
        "tile model 16x16x16 tile8: naive_loads={}, tiled_loads={}, \
         naive_intensity={:.1}, tiled_intensity={:.1}",
        loads.naive_loads,
        loads.tiled_loads,
        loads.naive_arithmetic_intensity,
        loads.tiled_arithmetic_intensity,
    );
    Ok(())
}
