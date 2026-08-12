//! 第 5–7 章 CPU-first 贯穿实验。
//!
//! 这个 crate 把数据管道、训练循环和 ModuleRecord artifact 放在同一个
//! 可测试的最小闭环中。它不引入 burn-onnx、Remote、DDP、CUDA、HTTP 或
//! 本地 path dependency。

use std::{
    error::Error,
    fmt::{self, Display, Formatter},
    sync::Arc,
};

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::{
            InMemDataset,
            transform::{Mapper, MapperDataset, PartialDataset},
        },
    },
    module::{AutodiffModule, Module},
    nn::LinearConfig,
    optim::{GradientsParams, SgdConfig},
    store::{ModuleRecord, RecordError},
    tensor::{Device, Tensor, TensorData},
};

pub const TOTAL_SAMPLES: usize = 20;
pub const TRAIN_SAMPLES: usize = 16;
pub const VALIDATION_SAMPLES: usize = TOTAL_SAMPLES - TRAIN_SAMPLES;
pub const BATCH_SIZE: usize = 4;
pub const TRAIN_EPOCHS: usize = 32;
pub const SHUFFLE_SEED: u64 = 41;

#[derive(Debug, Clone, PartialEq)]
pub enum CapstoneError {
    InvalidConfiguration(&'static str),
    NonFiniteObservation(&'static str),
    RecordValidation,
}

impl Display for CapstoneError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid configuration: {message}")
            }
            Self::NonFiniteObservation(name) => write!(formatter, "non-finite observation: {name}"),
            Self::RecordValidation => {
                formatter.write_str("wrong-topology record validation did not fail")
            }
        }
    }
}

impl Error for CapstoneError {}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionSample {
    pub id: usize,
    pub features: [f32; 2],
    pub target: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSample {
    pub id: usize,
    pub features: [f32; 2],
    pub target: f32,
}

#[derive(Clone, Debug, Default)]
pub struct PrepareRegressionSample;

impl Mapper<RegressionSample, PreparedSample> for PrepareRegressionSample {
    fn map(&self, item: &RegressionSample) -> PreparedSample {
        PreparedSample {
            id: item.id,
            features: item.features,
            target: item.target,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegressionBatch {
    pub ids: Vec<usize>,
    pub inputs: Tensor<2>,
    pub targets: Tensor<2>,
}

#[derive(Clone, Debug, Default)]
pub struct RegressionBatcher;

impl Batcher<PreparedSample, RegressionBatch> for RegressionBatcher {
    fn batch(&self, items: Vec<PreparedSample>, device: &Device) -> RegressionBatch {
        let mut input_values = Vec::with_capacity(items.len() * 2);
        let mut target_values = Vec::with_capacity(items.len());
        let mut ids = Vec::with_capacity(items.len());

        for item in items {
            ids.push(item.id);
            input_values.extend(item.features);
            target_values.push(item.target);
        }

        let batch_size = ids.len();
        RegressionBatch {
            ids,
            inputs: Tensor::from_data(TensorData::new(input_values, [batch_size, 2]), device),
            targets: Tensor::from_data(TensorData::new(target_values, [batch_size, 1]), device),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoaderAudit {
    pub items_processed: usize,
    pub batches: usize,
    pub ids: Vec<usize>,
    pub input_shapes: Vec<[usize; 2]>,
    pub target_shapes: Vec<[usize; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapstoneReport {
    pub train_samples: usize,
    pub validation_samples: usize,
    pub train_batches: usize,
    pub validation_batches: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub validation_loss: f32,
    pub parameter_delta: f32,
    pub record_tensors: usize,
    pub output_shape: [usize; 2],
    pub max_abs_error: f32,
    pub wrong_topology_rejected: bool,
}

type RegressionLoader = Arc<dyn DataLoader<RegressionBatch>>;
type MappedDataset =
    MapperDataset<InMemDataset<RegressionSample>, PrepareRegressionSample, RegressionSample>;

fn samples() -> Vec<RegressionSample> {
    (0..TOTAL_SAMPLES)
        .map(|id| {
            let first = id as f32 / 10.0 - 1.0;
            let second = ((id * 3) % 7) as f32 / 3.0 - 1.0;
            let target = 1.5 * first - 0.75 * second + 0.25;
            RegressionSample {
                id,
                features: [first, second],
                target,
            }
        })
        .collect()
}

fn mapped_dataset() -> MappedDataset {
    MapperDataset::new(InMemDataset::new(samples()), PrepareRegressionSample)
}

fn build_loaders(
    train_device: Device,
    validation_device: Device,
) -> (RegressionLoader, RegressionLoader, RegressionLoader) {
    let train_dataset = PartialDataset::new(mapped_dataset(), 0, TRAIN_SAMPLES);
    let train_eval_dataset = PartialDataset::new(mapped_dataset(), 0, TRAIN_SAMPLES);
    let validation_dataset = PartialDataset::new(mapped_dataset(), TRAIN_SAMPLES, TOTAL_SAMPLES);

    let train_loader = DataLoaderBuilder::new(RegressionBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(0)
        .shuffle(SHUFFLE_SEED)
        .set_device(train_device)
        .build(train_dataset);
    let train_eval_loader = DataLoaderBuilder::new(RegressionBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(0)
        .set_device(validation_device.clone())
        .build(train_eval_dataset);
    let validation_loader = DataLoaderBuilder::new(RegressionBatcher)
        .batch_size(BATCH_SIZE)
        .num_workers(0)
        .set_device(validation_device)
        .build(validation_dataset);

    (train_loader, train_eval_loader, validation_loader)
}

fn audit_loader(loader: &RegressionLoader) -> Result<LoaderAudit, Box<dyn Error>> {
    let mut iterator = loader.iter();
    let mut ids = Vec::new();
    let mut input_shapes = Vec::new();
    let mut target_shapes = Vec::new();
    let mut batches = 0;

    for result in iterator.by_ref() {
        let batch = result?;
        batches += 1;
        input_shapes.push(batch.inputs.dims());
        target_shapes.push(batch.targets.dims());
        ids.extend(batch.ids);
    }

    Ok(LoaderAudit {
        items_processed: iterator.progress().items_processed,
        batches,
        ids,
        input_shapes,
        target_shapes,
    })
}

fn io_error(message: &'static str) -> Box<dyn Error> {
    Box::new(std::io::Error::other(message))
}

fn audit_matches_expected(
    audit: &LoaderAudit,
    expected_ids: &[usize],
    expected_batches: usize,
) -> bool {
    let mut actual_ids = audit.ids.clone();
    actual_ids.sort_unstable();

    audit.items_processed == expected_ids.len()
        && actual_ids == expected_ids
        && audit.batches == expected_batches
        && audit
            .input_shapes
            .iter()
            .all(|shape| *shape == [BATCH_SIZE, 2])
        && audit
            .target_shapes
            .iter()
            .all(|shape| *shape == [BATCH_SIZE, 1])
}

// ANCHOR: capstone_pipeline
/// Run the complete deterministic data → train → artifact → inference path.
pub fn run_capstone() -> Result<CapstoneReport, Box<dyn Error>> {
    // Stage 1: build and audit the data boundary before any training begins.
    let train_device = Device::flex().autodiff();
    let validation_device = Device::flex();
    train_device.seed(73);

    let (train_loader, train_eval_loader, validation_loader) =
        build_loaders(train_device.clone(), validation_device.clone());
    let train_audit = audit_loader(&train_loader)?;
    let validation_audit = audit_loader(&validation_loader)?;

    let expected_train_ids = (0..TRAIN_SAMPLES).collect::<Vec<_>>();
    let expected_validation_ids = (TRAIN_SAMPLES..TOTAL_SAMPLES).collect::<Vec<_>>();
    if !audit_matches_expected(
        &train_audit,
        &expected_train_ids,
        TRAIN_SAMPLES / BATCH_SIZE,
    ) {
        return Err(io_error(
            "train loader did not preserve the exact split or batch contract",
        ));
    }
    if !audit_matches_expected(
        &validation_audit,
        &expected_validation_ids,
        VALIDATION_SAMPLES / BATCH_SIZE,
    ) {
        return Err(io_error(
            "validation loader did not preserve the exact split or batch contract",
        ));
    }

    // Stage 2: train on the already audited loader.
    let config = LinearConfig::new(2, 1);
    let mut model = config.init(&train_device);
    let initial_weights: Vec<f32> = model.weight.val().into_data().iter::<f32>().collect();
    let initial_loss = evaluate_loss(&model.clone().valid(), &train_eval_loader)?;
    let mut optimizer = SgdConfig::new().init();

    for _ in 0..TRAIN_EPOCHS {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for result in train_loader.iter() {
            let batch = result?;
            let prediction = model.forward(batch.inputs);
            let loss = (prediction - batch.targets).powf_scalar(2.0).mean();
            let loss_value = loss.clone().into_scalar::<f32>();
            if !loss_value.is_finite() {
                return Err(Box::new(CapstoneError::NonFiniteObservation(
                    "training loss",
                )));
            }
            epoch_loss += loss_value;
            batches += 1;

            let gradients = loss.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(0.05.into(), model, gradients);
        }
        if batches == 0 {
            return Err(Box::new(CapstoneError::InvalidConfiguration(
                "training loader produced no batches",
            )));
        }
        let _epoch_loss = epoch_loss / batches as f32;
    }

    let final_weights: Vec<f32> = model.weight.val().into_data().iter::<f32>().collect();
    let parameter_delta = initial_weights
        .iter()
        .zip(final_weights.iter())
        .map(|(before, after)| (after - before).abs())
        .sum::<f32>();

    // Stage 3: move the trained module to the validation device and serialize it.
    let valid_model = model.valid();
    let validation_loss = evaluate_loss(&valid_model, &validation_loader)?;
    let final_train_loss = evaluate_loss(&valid_model, &train_eval_loader)?;
    let record = valid_model.clone().into_record();
    let record_tensors = record.len();
    let bytes = record.into_bytes()?;
    let restored_record = ModuleRecord::from_bytes(bytes.clone())?;
    let restored = config
        .clone()
        .init(&validation_device)
        .try_load_record(restored_record)?;
    // Stage 4: verify restored inference and reject a wrong topology.
    let inference_input =
        Tensor::<2>::from_data([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], &validation_device);
    let reference: Vec<f32> = valid_model
        .forward(inference_input.clone())
        .into_data()
        .iter::<f32>()
        .collect();
    let restored_output = restored.forward(inference_input);
    let output_shape = restored_output.dims();
    let restored_values: Vec<f32> = restored_output.into_data().iter::<f32>().collect();
    let max_abs_error = reference
        .iter()
        .zip(restored_values.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f32::max);

    let wrong_topology = LinearConfig::new(3, 1)
        .init(&validation_device)
        .try_load_record(ModuleRecord::from_bytes(bytes)?);
    if !matches!(wrong_topology, Err(RecordError::Validation(_))) {
        return Err(Box::new(CapstoneError::RecordValidation));
    }

    let final_loss = final_train_loss;
    for (name, value) in [
        ("initial_loss", initial_loss),
        ("final_loss", final_loss),
        ("validation_loss", validation_loss),
        ("final_train_loss", final_train_loss),
        ("parameter_delta", parameter_delta),
        ("max_abs_error", max_abs_error),
    ] {
        if !value.is_finite() {
            return Err(Box::new(CapstoneError::NonFiniteObservation(name)));
        }
    }

    Ok(CapstoneReport {
        train_samples: train_audit.items_processed,
        validation_samples: validation_audit.items_processed,
        train_batches: train_audit.batches,
        validation_batches: validation_audit.batches,
        initial_loss,
        final_loss: final_train_loss,
        validation_loss,
        parameter_delta,
        record_tensors,
        output_shape,
        max_abs_error,
        wrong_topology_rejected: true,
    })
}
// ANCHOR_END: capstone_pipeline

fn evaluate_loss<M>(model: &M, loader: &RegressionLoader) -> Result<f32, Box<dyn Error>>
where
    M: ForwardRegression,
{
    let mut total = 0.0;
    let mut batches = 0;
    for result in loader.iter() {
        let batch = result?;
        let prediction = model.forward_regression(batch.inputs);
        let loss = (prediction - batch.targets).powf_scalar(2.0).mean();
        total += loss.into_scalar::<f32>();
        batches += 1;
    }
    if batches == 0 {
        return Err(io_error("evaluation loader produced no batches"));
    }
    Ok(total / batches as f32)
}

trait ForwardRegression {
    fn forward_regression(&self, input: Tensor<2>) -> Tensor<2>;
}

impl ForwardRegression for burn::nn::Linear {
    fn forward_regression(&self, input: Tensor<2>) -> Tensor<2> {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capstone_covers_data_training_artifact_and_inference() {
        let report = run_capstone().expect("deterministic capstone should pass");

        assert_eq!(report.train_samples, TRAIN_SAMPLES);
        assert_eq!(report.validation_samples, VALIDATION_SAMPLES);
        assert_eq!(report.train_batches, 4);
        assert_eq!(report.validation_batches, 1);
        assert!(report.final_loss < report.initial_loss);
        assert!(report.parameter_delta > 0.0);
        assert_eq!(report.record_tensors, 2);
        assert_eq!(report.output_shape, [3, 1]);
        assert!(report.max_abs_error < 1e-6);
        assert!(report.wrong_topology_rejected);
    }

    #[test]
    fn generated_samples_have_disjoint_fixed_split() {
        let all = samples();
        assert_eq!(all.len(), TOTAL_SAMPLES);
        assert_eq!(
            all.iter().map(|sample| sample.id).collect::<Vec<_>>(),
            (0..TOTAL_SAMPLES).collect::<Vec<_>>()
        );
        assert!(all.iter().all(|sample| sample.target.is_finite()));
    }
}
