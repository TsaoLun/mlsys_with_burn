use burn::tensor::{Device, StreamId, Tensor};
use burn_fusion::inspect::{FusionInspector, matchers};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSummary {
    pub fuser: Option<String>,
    pub operations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionSummary {
    pub split_by_sync: bool,
    pub reports: usize,
    pub blocks: Vec<BlockSummary>,
    pub observed_add: bool,
    pub observed_exp: bool,
    pub combined_add_exp: bool,
    pub output: Vec<f32>,
}

impl FusionSummary {
    pub fn has_add_exp_elementwise_block(&self) -> bool {
        self.combined_add_exp
    }
}

#[derive(Debug)]
pub struct InspectError(String);

impl Display for InspectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for InspectError {}

// ANCHOR: inspect
pub fn inspect_add_exp(split_by_sync: bool) -> Result<FusionSummary, InspectError> {
    let stream = StreamId::allocate();

    stream.executes(|| {
        let device = Device::cpu();
        let left = Tensor::<2>::ones([4, 4], &device);
        let right = Tensor::<2>::ones([4, 4], &device);
        let dtype = left.dtype();
        device
            .sync()
            .map_err(|error| InspectError(format!("物化输入失败：{error}")))?;

        let inspector = FusionInspector::install(stream);
        let intermediate = left + right;

        if split_by_sync {
            device
                .sync()
                .map_err(|error| InspectError(format!("物化中间结果失败：{error}")))?;
        }

        let output = intermediate.exp();
        let values = output
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| InspectError(format!("读取输出失败：{error}")))?;
        device
            .sync()
            .map_err(|error| InspectError(format!("排空 Fusion stream 失败：{error}")))?;

        let reports = inspector.drain();
        let add = matchers::is_add_float(dtype);
        let exp = matchers::is_exp(dtype);
        let observed_add = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .flat_map(|block| block.operations.iter())
            .any(add);
        let observed_exp = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .flat_map(|block| block.operations.iter())
            .any(exp);
        let combined_add_exp =
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| {
                    block.fuser_name() == Some("ElementWise")
                        && block
                            .ops_match(&[matchers::is_add_float(dtype), matchers::is_exp(dtype)])
                });
        let blocks = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .map(|block| BlockSummary {
                fuser: block.fuser_name().map(str::to_owned),
                operations: block.operations.len(),
            })
            .collect();

        Ok(FusionSummary {
            split_by_sync,
            reports: reports.len(),
            blocks,
            observed_add,
            observed_exp,
            combined_add_exp,
            output: values,
        })
    })
}
// ANCHOR_END: inspect

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observes_fusion_and_sync_boundary() {
        let fused = inspect_add_exp(false).expect("连续表达式应可执行");
        let split = inspect_add_exp(true).expect("带同步边界的表达式应可执行");

        assert!(fused.has_add_exp_elementwise_block());
        assert!(!split.has_add_exp_elementwise_block());
        assert!(fused.observed_add && fused.observed_exp);
        assert!(split.observed_add && split.observed_exp);
        assert!(fused.reports > 0 && split.reports > 0);
        assert_eq!(fused.output, split.output);
        assert!(
            fused
                .output
                .iter()
                .all(|value| (value - std::f32::consts::E.powi(2)).abs() < 1.0e-5)
        );
    }
}
