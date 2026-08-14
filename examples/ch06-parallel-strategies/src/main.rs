use ch06_parallel_strategies::{
    ZeroStage, gpipe_flush, one_f_one_b, ring_allreduce, tensor_parallel_allgather, zero_per_device,
};

// ANCHOR: walkthrough
fn main() {
    let payload = 1024u64;
    println!("环形 AllReduce：payload = {payload} 字节（必须能被 p 整除）");
    println!(
        "{:>4}  {:>12}  {:>10}  {:>8}",
        "p", "每设备发送", "α 步数", "相对 2S"
    );
    for world in [2u32, 4, 8, 16, 32] {
        let cost = ring_allreduce(payload, world).expect("payload 可被 p 整除");
        let (num, den) = cost.two_s_ratio();
        println!(
            "{:>4}  {:>12}  {:>10}  {:>5}/{:<2}",
            world, cost.bytes_sent, cost.alpha_steps, num, den
        );
    }

    println!("\n流水线空泡：p = 4 个 stage（空闲比例用分数，避免浮点）");
    println!(
        "{:>4}  {:>14}  {:>14}  {:>12}  {:>12}",
        "m", "GPipe 跨度", "GPipe 空闲", "1F1B 跨度", "1F1B 空闲"
    );
    for microbatches in [1u32, 4, 16, 64] {
        let gpipe = gpipe_flush(4, microbatches).expect("正数");
        let interleaved = one_f_one_b(4, microbatches).expect("正数");
        println!(
            "{:>4}  {:>14}  {:>8}/{:<3}  {:>12}  {:>8}/{:<3}",
            microbatches,
            gpipe.span,
            gpipe.idle_numerator,
            gpipe.idle_denominator,
            interleaved.span,
            interleaved.idle_numerator,
            interleaved.idle_denominator
        );
    }

    let params = 16u64;
    let grads = 16u64;
    let optimizer = 32u64;
    let world = 8u32;
    println!("\nZeRO 每卡显存：P={params} G={grads} O={optimizer} n={world}");
    println!(
        "{:>10}  {:>6}  {:>6}  {:>6}  {:>6}",
        "stage", "P", "G", "O", "合计"
    );
    for (name, stage) in [
        ("复制 / 0", ZeroStage::Replicated),
        ("ZeRO-1", ZeroStage::Optimizer),
        ("ZeRO-2", ZeroStage::Gradients),
        ("ZeRO-3", ZeroStage::Parameters),
    ] {
        let mem = zero_per_device(params, grads, optimizer, world, stage).expect("可整除");
        println!(
            "{:>10}  {:>6}  {:>6}  {:>6}  {:>6}",
            name,
            mem.params,
            mem.grads,
            mem.optimizer,
            mem.total()
        );
    }

    let activation = 1024u64;
    let tp = tensor_parallel_allgather(activation, 8).expect("可整除");
    println!("\n张量并行一层 AllGather：激活 {activation} 字节、p=8，每设备发送 {tp} 字节");
    println!("虚拟时间 / 字节模型：解释切分对象与流量，不是 NCCL 或 Megatron 的性能。");
}
// ANCHOR_END: walkthrough
