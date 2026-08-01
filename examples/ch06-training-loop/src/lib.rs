use burn::{
    nn::LinearConfig,
    optim::{GradientsParams, SgdConfig},
    tensor::{Device, Tensor},
};
use std::fmt::{Display, Formatter};

/// Errors that can be detected before starting the training loop.
#[derive(Debug, PartialEq)]
pub enum TrainingError {
    /// The loop needs at least one optimization step.
    NoSteps,
    /// The learning rate must be finite and positive.
    InvalidLearningRate,
}

impl Display for TrainingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSteps => write!(formatter, "training requires at least one step"),
            Self::InvalidLearningRate => {
                write!(formatter, "learning rate must be finite and positive")
            }
        }
    }
}

impl std::error::Error for TrainingError {}

/// Observable values produced by the small training experiment.
#[derive(Debug, PartialEq)]
pub struct TrainingReport {
    /// Mean-squared loss after each forward pass.
    pub losses: Vec<f32>,
    /// Loss before the first optimizer step.
    pub initial_loss: f32,
    /// Loss from the final forward pass.
    pub final_loss: f32,
    /// Sum of absolute changes in the learned weight parameters.
    pub parameter_delta: f32,
}

/// Run a deterministic CPU linear-regression loop.
///
/// The loop keeps the model and optimizer explicit so each system boundary is
/// visible: forward, loss, backward, gradient extraction, and optimizer step.
pub fn run_training(steps: usize, learning_rate: f64) -> Result<TrainingReport, TrainingError> {
    if steps == 0 {
        return Err(TrainingError::NoSteps);
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(TrainingError::InvalidLearningRate);
    }

    // ANCHOR: setup
    let device = Device::flex().autodiff();
    device.seed(7);

    let mut model = LinearConfig::new(1, 1).init(&device);
    let initial_weights: Vec<f32> = model.weight.val().into_data().iter::<f32>().collect();
    let mut optimizer = SgdConfig::new().init();

    let inputs = Tensor::<2>::from_data([[-2.0], [-1.0], [0.0], [1.0], [2.0]], &device);
    let targets = Tensor::<2>::from_data([[-3.0], [-1.0], [1.0], [3.0], [5.0]], &device);
    let mut losses = Vec::with_capacity(steps);
    // ANCHOR_END: setup

    // ANCHOR: train_step
    for _ in 0..steps {
        let prediction = model.forward(inputs.clone());
        let difference = prediction - targets.clone();
        let loss = difference.powf_scalar(2.0).mean();
        losses.push(loss.clone().into_scalar::<f32>());

        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(learning_rate.into(), model, gradients);
    }
    // ANCHOR_END: train_step

    let final_weights: Vec<f32> = model.weight.val().into_data().iter::<f32>().collect();
    let parameter_delta = initial_weights
        .iter()
        .zip(final_weights.iter())
        .map(|(before, after)| (after - before).abs())
        .sum();
    let initial_loss = losses.first().copied().ok_or(TrainingError::NoSteps)?;
    let final_loss = losses.last().copied().ok_or(TrainingError::NoSteps)?;

    Ok(TrainingReport {
        losses,
        initial_loss,
        final_loss,
        parameter_delta,
    })
}

/// Weighted average used by a pure protocol-level AllReduce comparison.
pub fn weighted_all_reduce(values: &[(f32, usize)]) -> Option<f32> {
    let total_samples = values.iter().map(|(_, samples)| *samples).sum::<usize>();
    if total_samples == 0 {
        return None;
    }
    let weighted_sum = values
        .iter()
        .map(|(gradient, samples)| *gradient * *samples as f32)
        .sum::<f32>();
    Some(weighted_sum / total_samples as f32)
}

/// Apply a gradient only when its version is within the allowed staleness.
pub fn apply_stale_gradient(
    current_version: u64,
    gradient_version: u64,
    gradient: f32,
    learning_rate: f32,
    max_staleness: u64,
) -> Option<(f32, u64)> {
    let staleness = current_version.checked_sub(gradient_version)?;
    if staleness > max_staleness {
        return None;
    }
    Some((learning_rate * gradient, current_version + 1))
}

pub fn quorum_reached(acknowledged: usize, replicas: usize, quorum: usize) -> bool {
    replicas > 0 && quorum > 0 && quorum <= replicas && acknowledged >= quorum
}

/// Return pipeline slots and the idealized bubble slots for a 1F1B card.
pub fn pipeline_slots(stages: usize, micro_batches: usize) -> Option<(usize, usize)> {
    if stages == 0 || micro_batches == 0 {
        return None;
    }
    Some((
        stages.saturating_add(micro_batches).saturating_sub(1),
        stages.saturating_sub(1),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointVersion {
    pub step: u64,
    pub committed: bool,
}

pub fn commit_checkpoint(
    current: CheckpointVersion,
    candidate: CheckpointVersion,
) -> Option<CheckpointVersion> {
    if candidate.committed && candidate.step > current.step {
        Some(candidate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reduces_loss_and_updates_parameters() {
        let report = run_training(40, 0.1).expect("valid training configuration");

        assert_eq!(report.losses.len(), 40);
        assert!(
            report.final_loss < report.initial_loss,
            "expected loss to decrease ({} -> {})",
            report.initial_loss,
            report.final_loss
        );
        assert!(report.parameter_delta > 0.0);
    }

    #[test]
    fn invalid_training_configuration_is_reported() {
        assert_eq!(run_training(0, 0.1), Err(TrainingError::NoSteps));
        assert_eq!(
            run_training(1, 0.0),
            Err(TrainingError::InvalidLearningRate)
        );
    }

    #[test]
    fn protocol_card_checks_weighted_collective_and_stale_gradient() {
        assert_eq!(weighted_all_reduce(&[(2.0, 1), (4.0, 3)]), Some(3.5));
        assert_eq!(weighted_all_reduce(&[(1.0, 0)]), None);
        assert_eq!(apply_stale_gradient(5, 4, 2.0, 0.1, 1), Some((0.2, 6)));
        assert_eq!(apply_stale_gradient(5, 2, 2.0, 0.1, 1), None);
    }

    #[test]
    fn protocol_card_checks_quorum_pipeline_and_checkpoint_commit() {
        assert!(quorum_reached(2, 3, 2));
        assert!(!quorum_reached(1, 3, 2));
        assert_eq!(pipeline_slots(4, 8), Some((11, 3)));
        assert_eq!(pipeline_slots(0, 8), None);

        let current = CheckpointVersion {
            step: 10,
            committed: true,
        };
        let next = CheckpointVersion {
            step: 12,
            committed: true,
        };
        assert_eq!(commit_checkpoint(current, next), Some(next));
        assert_eq!(
            commit_checkpoint(
                current,
                CheckpointVersion {
                    step: 9,
                    committed: true
                }
            ),
            None
        );
    }
}
