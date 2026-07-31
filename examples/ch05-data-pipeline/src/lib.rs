//! 第 5 章的数据集、变换、批处理与多线程 DataLoader 示例。

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::time::Instant;

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, Progress, batcher::Batcher},
        dataset::{
            DatasetError, InMemDataset,
            transform::{Mapper, MapperDataset},
        },
    },
    tensor::Device,
};

/// 教学数据集中的样本数量。
pub const SAMPLE_COUNT: usize = 12;

/// 从数据源读出的原始样本。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawSample {
    /// 用于观察顺序的稳定标识。
    pub id: usize,
    /// 待变换的整数值。
    pub value: usize,
}

/// 经过惰性 map 变换后的样本。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedSample {
    /// 原样本标识。
    pub id: usize,
    /// `PrepareSample` 产生的值。
    pub value: usize,
}

/// 一个最小的逐样本预处理算子。
#[derive(Clone, Debug, Default)]
pub struct PrepareSample;

impl Mapper<RawSample, PreparedSample> for PrepareSample {
    fn map(&self, item: &RawSample) -> PreparedSample {
        PreparedSample {
            id: item.id,
            value: item.value * 2 + 1,
        }
    }
}

// ANCHOR: dataset
/// 创建一个内存数据集，并用 `MapperDataset` 叠加惰性预处理。
pub fn prepared_dataset() -> MapperDataset<InMemDataset<RawSample>, PrepareSample, RawSample> {
    let samples = (0..SAMPLE_COUNT)
        .map(|id| RawSample { id, value: id })
        .collect();
    let source = InMemDataset::new(samples);

    MapperDataset::new(source, PrepareSample)
}
// ANCHOR_END: dataset

/// 一个只保留 host 值的批次。
///
/// 真实训练中的 Batcher 通常会在这里构造 Tensor；本实验保留 `Device` 字段，
/// 让读者能观察 DataLoader 将批次交给哪个设备，同时把顺序和 batching 结果
/// 与张量计算区分开。
#[derive(Clone, Debug)]
pub struct SampleBatch {
    /// 批次中的样本标识。
    pub ids: Vec<usize>,
    /// 批次中的预处理值。
    pub values: Vec<usize>,
    /// 构造批次时收到的目标设备。
    pub device: Device,
}

/// 将样本收集成 host 批次的 Batcher。
#[derive(Clone, Debug, Default)]
pub struct SampleBatcher;

// ANCHOR: batcher
impl Batcher<PreparedSample, SampleBatch> for SampleBatcher {
    fn batch(&self, items: Vec<PreparedSample>, device: &Device) -> SampleBatch {
        let ids = items.iter().map(|item| item.id).collect();
        let values = items.iter().map(|item| item.value).collect();

        SampleBatch {
            ids,
            values,
            device: device.clone(),
        }
    }
}
// ANCHOR_END: batcher

/// 数据加载失败或实验参数无效。
#[derive(Debug)]
pub enum PipelineError {
    /// 固定批大小必须大于零。
    InvalidBatchSize,
    /// Dataset 在读取过程中返回了错误。
    Dataset(DatasetError),
    /// 至少需要测量一个 epoch。
    InvalidEpochs,
}

impl Display for PipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBatchSize => formatter.write_str("batch size 必须大于零"),
            Self::Dataset(error) => write!(formatter, "读取数据集失败：{error}"),
            Self::InvalidEpochs => formatter.write_str("测量 epoch 数必须大于零"),
        }
    }
}

impl Error for PipelineError {}

impl From<DatasetError> for PipelineError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

/// 迭代器结束时的进度快照。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgressSnapshot {
    /// 已处理的样本数。
    pub items_processed: usize,
    /// 本轮应处理的样本总数。
    pub items_total: usize,
}

/// 一轮数据管道的可观察结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochReport {
    /// 批次大小序列；多 worker 时只表示到达顺序，不代表全局输入序号。
    pub batch_sizes: Vec<usize>,
    /// 批次到达时的样本 ID 展平结果。
    pub ids: Vec<usize>,
    /// 与 `ids` 同位置的预处理值。
    pub values: Vec<usize>,
    /// Batcher 收到的设备 Debug 标签。
    pub device: String,
    /// 迭代器结束后的进度。
    pub progress: ProgressSnapshot,
}

fn build_loader(
    batch_size: usize,
    num_workers: usize,
    shuffle_seed: Option<u64>,
) -> Result<Arc<dyn DataLoader<SampleBatch>>, PipelineError> {
    if batch_size == 0 {
        return Err(PipelineError::InvalidBatchSize);
    }

    let builder = DataLoaderBuilder::new(SampleBatcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .set_device(Device::flex());
    let builder = match shuffle_seed {
        Some(seed) => builder.shuffle(seed),
        None => builder,
    };

    Ok(builder.build(prepared_dataset()))
}

fn progress_snapshot(progress: Progress) -> ProgressSnapshot {
    ProgressSnapshot {
        items_processed: progress.items_processed,
        items_total: progress.items_total,
    }
}

// ANCHOR: pipeline
fn collect_loader(loader: &Arc<dyn DataLoader<SampleBatch>>) -> Result<EpochReport, PipelineError> {
    let mut batch_sizes = Vec::new();
    let mut ids = Vec::new();
    let mut values = Vec::new();
    let mut device = None;
    let mut iterator = loader.iter();

    for batch in iterator.by_ref() {
        let batch = batch?;
        device.get_or_insert_with(|| format!("{:?}", batch.device));
        batch_sizes.push(batch.ids.len());
        ids.extend(batch.ids);
        values.extend(batch.values);
    }

    Ok(EpochReport {
        batch_sizes,
        ids,
        values,
        device: device.unwrap_or_else(|| format!("{:?}", Device::flex())),
        progress: progress_snapshot(iterator.progress()),
    })
}

/// 运行一轮数据管道。
///
/// `num_workers = 0` 走同步的 `BatchDataLoader`；大于零时走固定快照中的
/// `MultiThreadDataLoader`。`shuffle_seed` 为 `Some` 时启用 Burn 的按迭代
/// shuffle。
pub fn run_epoch(
    batch_size: usize,
    num_workers: usize,
    shuffle_seed: Option<u64>,
) -> Result<EpochReport, PipelineError> {
    let loader = build_loader(batch_size, num_workers, shuffle_seed)?;
    collect_loader(&loader)
}
// ANCHOR_END: pipeline

/// 在同一个带 shuffle 的 DataLoader 上连续运行两轮。
pub fn run_two_shuffled_epochs(
    batch_size: usize,
    num_workers: usize,
    seed: u64,
) -> Result<(EpochReport, EpochReport), PipelineError> {
    let loader = build_loader(batch_size, num_workers, Some(seed))?;
    let first = collect_loader(&loader)?;
    let second = collect_loader(&loader)?;
    Ok((first, second))
}

/// 简单的吞吐观察结果。
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughputReport {
    /// 使用的批大小。
    pub batch_size: usize,
    /// 使用的 worker 数。
    pub num_workers: usize,
    /// 测量的 epoch 数。
    pub epochs: usize,
    /// 测量期间处理的样本数。
    pub items: usize,
    /// 以微秒表示的墙钟时间。
    pub elapsed_micros: u128,
    /// `items / elapsed_seconds`，只用于本机相对观察。
    pub items_per_second: f64,
}

/// 对一个已经 warm-up 的数据管道做粗粒度吞吐测量。
///
/// 该函数不是基准测试框架：结果受 CPU、线程调度、编译和进程状态影响。
/// 它的用途是让读者把 batch size/worker 配置与可观察的墙钟指标联系起来。
pub fn measure_throughput(
    batch_size: usize,
    num_workers: usize,
    epochs: usize,
) -> Result<ThroughputReport, PipelineError> {
    if epochs == 0 {
        return Err(PipelineError::InvalidEpochs);
    }

    let loader = build_loader(batch_size, num_workers, None)?;
    let _warmup = collect_loader(&loader)?;

    let started = Instant::now();
    let mut items = 0;
    for _ in 0..epochs {
        items += collect_loader(&loader)?.ids.len();
    }
    let elapsed = started.elapsed();
    let elapsed_seconds = elapsed.as_secs_f64().max(f64::EPSILON);

    Ok(ThroughputReport {
        batch_size,
        num_workers,
        epochs,
        items,
        elapsed_micros: elapsed.as_micros(),
        items_per_second: items as f64 / elapsed_seconds,
    })
}

/// 返回未混洗时应覆盖的全部输入 ID。
pub fn expected_ids() -> Vec<usize> {
    (0..SAMPLE_COUNT).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_ids(report: &EpochReport) -> Vec<usize> {
        let mut ids = report.ids.clone();
        ids.sort_unstable();
        ids
    }

    fn sorted_pairs(report: &EpochReport) -> Vec<(usize, usize)> {
        let mut pairs: Vec<_> = report
            .ids
            .iter()
            .copied()
            .zip(report.values.iter().copied())
            .collect();
        pairs.sort_unstable_by_key(|(id, _)| *id);
        pairs
    }

    #[test]
    fn lazy_map_and_batching_preserve_expected_values() {
        let report = run_epoch(3, 0, None).expect("CPU 单线程数据管道应成功");

        assert_eq!(report.batch_sizes, vec![3, 3, 3, 3]);
        assert_eq!(report.ids, expected_ids());
        assert_eq!(
            report.values,
            expected_ids()
                .into_iter()
                .map(|id| id * 2 + 1)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            report.progress,
            ProgressSnapshot {
                items_processed: SAMPLE_COUNT,
                items_total: SAMPLE_COUNT,
            }
        );
    }

    #[test]
    fn fixed_seed_reproduces_single_worker_epoch() {
        let first = run_epoch(3, 0, Some(42)).expect("第一次 shuffle 应成功");
        let second = run_epoch(3, 0, Some(42)).expect("第二次 shuffle 应成功");

        assert_eq!(first.ids, second.ids);
        assert_ne!(first.ids, expected_ids());
        assert_eq!(sorted_pairs(&first), sorted_pairs(&second));
    }

    #[test]
    fn one_loader_advances_to_a_new_shuffle_each_epoch() {
        let (first, second) = run_two_shuffled_epochs(3, 0, 42).expect("连续两轮 shuffle 应成功");

        assert_ne!(first.ids, second.ids);
        assert_eq!(sorted_ids(&first), expected_ids());
        assert_eq!(sorted_ids(&second), expected_ids());
    }

    #[test]
    fn multi_worker_loader_conserves_items_and_receives_device() {
        let report = run_epoch(3, 2, None).expect("CPU 多 worker 数据管道应成功");

        assert_eq!(report.batch_sizes.iter().sum::<usize>(), SAMPLE_COUNT);
        assert_eq!(sorted_ids(&report), expected_ids());
        assert_eq!(sorted_pairs(&report), {
            expected_ids()
                .into_iter()
                .map(|id| (id, id * 2 + 1))
                .collect::<Vec<_>>()
        });
        assert_eq!(
            report.progress,
            ProgressSnapshot {
                items_processed: SAMPLE_COUNT,
                items_total: SAMPLE_COUNT,
            }
        );
        assert!(report.device.contains("Flex"));
    }

    #[test]
    fn rejects_zero_batch_size_and_zero_epochs() {
        assert!(matches!(
            run_epoch(0, 0, None),
            Err(PipelineError::InvalidBatchSize)
        ));
        assert!(matches!(
            measure_throughput(2, 0, 0),
            Err(PipelineError::InvalidEpochs)
        ));
    }
}
