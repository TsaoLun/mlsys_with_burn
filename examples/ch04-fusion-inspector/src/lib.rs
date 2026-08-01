use burn::tensor::{Device, StreamId, Tensor};
use burn_fusion::inspect::{FusionInspector, matchers};
use std::env;
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

#[derive(Debug, Clone, PartialEq)]
pub struct TripleFusionSummary {
    pub reports: usize,
    pub blocks: Vec<BlockSummary>,
    pub combined_add_mul_exp: bool,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatedFusionSummary {
    pub first: TripleFusionSummary,
    pub second: TripleFusionSummary,
    pub same_plan: bool,
    pub same_output: bool,
    pub cache_log_enabled: bool,
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

// ANCHOR: inspect_triple
pub fn inspect_add_mul_exp() -> Result<TripleFusionSummary, InspectError> {
    let stream = StreamId::allocate();

    inspect_add_mul_exp_once(stream)
}

fn inspect_add_mul_exp_once(stream: StreamId) -> Result<TripleFusionSummary, InspectError> {
    stream.executes(|| {
        let device = Device::cpu();
        let left = Tensor::<2>::ones([4, 4], &device);
        let right = Tensor::<2>::ones([4, 4], &device);
        let scale = Tensor::<2>::ones([4, 4], &device);
        let dtype = left.dtype();
        device
            .sync()
            .map_err(|error| InspectError(format!("物化输入失败：{error}")))?;

        let inspector = FusionInspector::install(stream);
        let output = ((left + right) * scale).exp();
        let values = output
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| InspectError(format!("读取输出失败：{error}")))?;
        device
            .sync()
            .map_err(|error| InspectError(format!("排空 Fusion stream 失败：{error}")))?;

        let reports = inspector.drain();
        let combined_add_mul_exp =
            reports
                .iter()
                .flat_map(|report| report.blocks.iter())
                .any(|block| {
                    block.fuser_name() == Some("ElementWise")
                        && block.ops_match(&[
                            matchers::is_add_float(dtype),
                            matchers::is_mul_float(dtype),
                            matchers::is_exp(dtype),
                        ])
                });
        let blocks = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .map(|block| BlockSummary {
                fuser: block.fuser_name().map(str::to_owned),
                operations: block.operations.len(),
            })
            .collect();

        Ok(TripleFusionSummary {
            reports: reports.len(),
            blocks,
            combined_add_mul_exp,
            output: values,
        })
    })
}
// ANCHOR_END: inspect_triple

/// Repeat the same shape/dtype/device plan and compare observable structure.
///
/// The report intentionally compares plan structure and output values, not
/// wall-clock time or a private cache key. Set `BURN_FUSION_LOG=full` to ask
/// the fixed runtime for optional compile/cache logs.
pub fn inspect_add_mul_exp_twice() -> Result<RepeatedFusionSummary, InspectError> {
    let stream = StreamId::allocate();
    let first = inspect_add_mul_exp_once(stream)?;
    let second = inspect_add_mul_exp_once(stream)?;
    let same_plan = first.reports == second.reports && first.blocks == second.blocks;
    let same_output = first.output == second.output;
    let cache_log_enabled = env::var("BURN_FUSION_LOG")
        .map(|value| value == "full")
        .unwrap_or(false);

    Ok(RepeatedFusionSummary {
        first,
        second,
        same_plan,
        same_output,
        cache_log_enabled,
    })
}

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

    #[test]
    fn observes_three_op_elementwise_block() {
        let report = inspect_add_mul_exp().expect("三操作表达式应可执行");

        assert!(report.combined_add_mul_exp);
        assert!(report.reports > 0);
        assert!(
            report
                .output
                .iter()
                .all(|value| (value - std::f32::consts::E.powi(2)).abs() < 1.0e-5)
        );
    }

    #[test]
    fn repeated_plan_and_output_are_stable() {
        let report = inspect_add_mul_exp_twice().expect("重复 Fusion 观察应成功");

        assert!(report.same_plan);
        assert!(report.same_output);
        assert_eq!(report.first.output, report.second.output);
        assert!(report.first.output.iter().all(|value| value.is_finite()));
    }
}
