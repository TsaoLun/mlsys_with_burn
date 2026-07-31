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
}
