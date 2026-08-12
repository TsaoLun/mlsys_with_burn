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

/// Serialize the example model into in-memory Burnpack bytes.
pub fn sample_record_bytes() -> Result<burn::tensor::Bytes, burn::store::RecordError> {
    let device = Device::flex();
    let config = LinearConfig::new(2, 1).with_initializer(Initializer::Constant { value: 0.5 });
    let model = config.init(&device);
    model.into_record().into_bytes()
}

/// Errors reported by the minimal Burnpack header reader.
#[derive(Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Fewer bytes than the fixed header.
    TruncatedHeader,
    /// The magic number does not identify a Burnpack container.
    BadMagic([u8; 4]),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => formatter.write_str("burnpack header is truncated"),
            Self::BadMagic(magic) => write!(formatter, "bad burnpack magic: {magic:02x?}"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Byte-level facts read from the fixed Burnpack header alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnpackLayout {
    /// Magic bytes as stored on disk.
    pub magic: [u8; 4],
    /// Container format version.
    pub version: u16,
    /// Size of the CBOR metadata section in bytes.
    pub metadata_size: u32,
    /// Absolute offset where the 256-byte-aligned tensor data section starts.
    pub data_section_start: usize,
    /// Total container length in bytes.
    pub total_len: usize,
}

/// The fixed header is magic (4) + version (2) + metadata size (4).
pub const BURNPACK_HEADER_SIZE: usize = 10;

/// Tensor data starts on a 256-byte boundary so absolute file offsets stay
/// aligned for mmap-style zero-copy loading.
pub const BURNPACK_TENSOR_ALIGNMENT: usize = 256;

// ANCHOR: burnpack_layout
/// Parse only the fixed Burnpack header, without any serde machinery.
///
/// The format's magic constant is `0x4255524E` ("BURN" in ASCII). Because the
/// header is written little-endian, the bytes on disk read `NRUB` — seeing the
/// letters reversed is expected, not corruption.
pub fn inspect_burnpack_layout(bytes: &[u8]) -> Result<BurnpackLayout, LayoutError> {
    if bytes.len() < BURNPACK_HEADER_SIZE {
        return Err(LayoutError::TruncatedHeader);
    }
    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    let metadata_size = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    if u32::from_le_bytes(magic) != 0x4255_524E {
        return Err(LayoutError::BadMagic(magic));
    }
    let data_section_start = (BURNPACK_HEADER_SIZE + metadata_size as usize)
        .div_ceil(BURNPACK_TENSOR_ALIGNMENT)
        * BURNPACK_TENSOR_ALIGNMENT;

    Ok(BurnpackLayout {
        magic,
        version,
        metadata_size,
        data_section_start,
        total_len: bytes.len(),
    })
}
// ANCHOR_END: burnpack_layout

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
    fn burnpack_layout_exposes_endianness_version_and_alignment() {
        let bytes = sample_record_bytes().expect("record serialization should work");
        let layout = inspect_burnpack_layout(&bytes).expect("valid burnpack header");

        // The magic constant is "BURN" (0x4255524E); stored little-endian, the
        // bytes on disk read "NRUB".
        assert_eq!(&layout.magic, b"NRUB");
        assert_eq!(layout.version, 1);
        assert!(layout.metadata_size > 0);
        assert_eq!(
            layout.data_section_start % BURNPACK_TENSOR_ALIGNMENT,
            0,
            "tensor data must start on a 256-byte boundary"
        );
        assert!(
            layout.data_section_start >= BURNPACK_HEADER_SIZE + layout.metadata_size as usize,
            "data section must not overlap the metadata"
        );
        assert!(layout.total_len >= layout.data_section_start);
    }

    #[test]
    fn burnpack_layout_rejects_truncated_or_foreign_bytes() {
        assert_eq!(
            inspect_burnpack_layout(b"BUR"),
            Err(LayoutError::TruncatedHeader)
        );

        let mut bytes = sample_record_bytes()
            .expect("record serialization should work")
            .to_vec();
        bytes[0] = b'X';
        assert!(matches!(
            inspect_burnpack_layout(&bytes),
            Err(LayoutError::BadMagic(_))
        ));
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
