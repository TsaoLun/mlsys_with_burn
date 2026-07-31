use burn::{
    module::{Initializer, Module},
    nn::LinearConfig,
    store::ModuleRecord,
    tensor::{Device, Tensor},
};

/// Observable results from the model-state round-trip.
#[derive(Debug, PartialEq)]
pub struct RoundTripReport {
    /// Number of parameter tensors stored in the record.
    pub record_tensors: usize,
    /// Output shape after loading the record.
    pub output_shape: [usize; 2],
    /// Maximum absolute difference between reference and restored outputs.
    pub max_abs_error: f32,
}

/// Save a small model to in-memory Burnpack bytes and load it again on CPU.
///
/// This tests the model artifact boundary independently from an ONNX importer,
/// network transport, or a particular service framework.
// ANCHOR: run_round_trip
pub fn run_round_trip() -> Result<RoundTripReport, burn::store::RecordError> {
    let device = Device::flex();
    let config = LinearConfig::new(2, 1).with_initializer(Initializer::Constant { value: 0.5 });
    let model = config.init(&device);
    let input = Tensor::<2>::from_data([[1.0, 2.0], [-1.0, 3.0], [4.0, -2.0]], &device);

    let reference: Vec<f32> = model
        .forward(input.clone())
        .into_data()
        .iter::<f32>()
        .collect();

    let record = model.into_record();
    let record_tensors = record.len();
    let bytes = record.into_bytes()?;
    let restored_record = ModuleRecord::from_bytes(bytes)?;
    let restored = config.init(&device).try_load_record(restored_record)?;

    let restored_output = restored.forward(input);
    let output_shape = restored_output.dims();
    let restored_values: Vec<f32> = restored_output.into_data().iter::<f32>().collect();
    let max_abs_error = reference
        .iter()
        .zip(restored_values.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f32::max);

    Ok(RoundTripReport {
        record_tensors,
        output_shape,
        max_abs_error,
    })
}
// ANCHOR_END: run_round_trip

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trip_preserves_inference() {
        let report = run_round_trip().expect("valid in-memory record round-trip");

        assert_eq!(report.record_tensors, 2);
        assert_eq!(report.output_shape, [3, 1]);
        assert!(
            report.max_abs_error < 1e-6,
            "restored output changed by {}",
            report.max_abs_error
        );
    }
}
