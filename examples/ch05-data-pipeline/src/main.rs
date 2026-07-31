use ch05_data_pipeline::{measure_throughput, run_epoch, run_two_shuffled_epochs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ordered = run_epoch(3, 0, None)?;
    let shuffled = run_epoch(3, 0, Some(42))?;
    let parallel = run_epoch(3, 2, None)?;
    let (first_epoch, second_epoch) = run_two_shuffled_epochs(3, 0, 42)?;

    println!("单线程批大小：{:?}", ordered.batch_sizes);
    println!("单线程输入顺序：{:?}", ordered.ids);
    println!("固定 seed 的顺序：{:?}", shuffled.ids);
    println!(
        "同一 loader 两轮是否相同：{}",
        first_epoch.ids == second_epoch.ids
    );
    println!("多 worker 到达顺序：{:?}", parallel.ids);
    println!("多 worker 排序后样本数：{}", {
        let mut ids = parallel.ids.clone();
        ids.sort_unstable();
        ids.len()
    });

    for (batch_size, num_workers) in [(2, 0), (2, 2), (4, 0), (4, 2)] {
        let report = measure_throughput(batch_size, num_workers, 20)?;
        println!(
            "吞吐观察：batch_size={} workers={} items={} elapsed={}us items/s={:.1}",
            report.batch_size,
            report.num_workers,
            report.items,
            report.elapsed_micros,
            report.items_per_second
        );
    }

    Ok(())
}
