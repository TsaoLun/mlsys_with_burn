use ch04_fusion_inspector::inspect_add_exp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fused = inspect_add_exp(false)?;
    let split = inspect_add_exp(true)?;

    println!("连续表达式：{} 个报告，{:?}", fused.reports, fused.blocks);
    println!("同步切分后：{} 个报告，{:?}", split.reports, split.blocks);
    println!("输出前四项：{:?}", &fused.output[..4]);
    Ok(())
}
