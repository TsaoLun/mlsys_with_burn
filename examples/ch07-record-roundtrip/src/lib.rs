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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub version: u64,
    pub payload_len: usize,
    pub checksum: u64,
}

pub fn stable_checksum(payload: &[u8]) -> u64 {
    payload.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

impl ArtifactManifest {
    pub fn for_payload(version: u64, payload: &[u8]) -> Self {
        Self {
            version,
            payload_len: payload.len(),
            checksum: stable_checksum(payload),
        }
    }

    pub fn verifies(&self, payload: &[u8]) -> bool {
        self.payload_len == payload.len() && self.checksum == stable_checksum(payload)
    }
}

pub fn rollback_allowed(current: &ArtifactManifest, candidate: &ArtifactManifest) -> bool {
    candidate.version < current.version && candidate.payload_len > 0
}

/// Group requests by shape key while respecting a maximum dynamic batch size.
pub fn dynamic_batch_groups(
    requests: &[(u8, usize)],
    max_batch: usize,
) -> Option<Vec<Vec<(u8, usize)>>> {
    if max_batch == 0 {
        return None;
    }
    let mut groups: Vec<Vec<(u8, usize)>> = Vec::new();
    for request in requests {
        let can_append = groups
            .last()
            .is_some_and(|group| group[0].0 == request.0 && group.len() < max_batch);
        if can_append {
            groups
                .last_mut()
                .expect("checked by is_some_and")
                .push(*request);
        } else {
            groups.push(vec![*request]);
        }
    }
    Some(groups)
}

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

    #[test]
    fn artifact_contract_checks_checksum_version_and_rollback() {
        let payload = b"model-record";
        let manifest = ArtifactManifest::for_payload(3, payload);
        assert!(manifest.verifies(payload));
        assert!(!manifest.verifies(b"tampered"));

        let older = ArtifactManifest::for_payload(2, b"previous");
        assert!(rollback_allowed(&manifest, &older));
        assert!(!rollback_allowed(&older, &manifest));
    }

    #[test]
    fn dynamic_batching_respects_shape_and_capacity() {
        let groups = dynamic_batch_groups(&[(1, 0), (1, 1), (2, 2), (1, 3)], 2)
            .expect("positive batch capacity");
        assert_eq!(
            groups,
            vec![vec![(1, 0), (1, 1)], vec![(2, 2)], vec![(1, 3)]]
        );
        assert_eq!(dynamic_batch_groups(&[], 2), Some(Vec::new()));
        assert_eq!(dynamic_batch_groups(&[(1, 0)], 0), None);
    }
}
