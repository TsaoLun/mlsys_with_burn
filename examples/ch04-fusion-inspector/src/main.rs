use ch04_fusion_inspector::{inspect_add_exp, inspect_add_mul_exp_twice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fused = inspect_add_exp(false)?;
    let split = inspect_add_exp(true)?;

    println!("连续表达式：{} 个报告，{:?}", fused.reports, fused.blocks);
    println!("同步切分后：{} 个报告，{:?}", split.reports, split.blocks);
    println!("输出前四项：{:?}", &fused.output[..4]);
    let repeated = inspect_add_mul_exp_twice()?;
    println!(
        "重复计划一致={} 输出一致={} cache_log_enabled={} blocks={:?}",
        repeated.same_plan,
        repeated.same_output,
        repeated.cache_log_enabled,
        repeated.second.blocks,
    );
    Ok(())
}
