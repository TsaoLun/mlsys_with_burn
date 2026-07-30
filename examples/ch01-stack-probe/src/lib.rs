//! 第一章的执行栈探测实验。

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use burn::tensor::{Device, Tensor};

const PINS: &str = include_str!("../../../pins.toml");

/// 从固定源码快照和真实执行中收集的最小报告。
#[derive(Debug, Clone, PartialEq)]
pub struct StackReport {
    /// `pins.toml` 中的写作快照名称。
    pub snapshot: String,
    /// `Device` 暴露的具体分派变体。
    pub device: String,
    /// 设备默认浮点类型。
    pub float_dtype: String,
    /// 设备默认整数类型。
    pub int_dtype: String,
    /// 设备是否记录自动微分。
    pub autodiff_enabled: bool,
    /// 经后端执行并读回主机的值。
    pub observed_value: f32,
}

/// 探测执行或读取结果失败。
#[derive(Debug)]
pub enum ProbeError {
    /// `pins.toml` 缺少预期的 snapshot 名称。
    MissingSnapshot,
    /// 后端同步失败。
    Sync(String),
    /// 张量数据无法按预期类型读回。
    Data(String),
}

impl Display for ProbeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSnapshot => write!(formatter, "pins.toml 缺少 snapshot.name"),
            Self::Sync(error) => write!(formatter, "设备同步失败：{error}"),
            Self::Data(error) => write!(formatter, "张量数据读回失败：{error}"),
        }
    }
}

impl Error for ProbeError {}

fn snapshot_name() -> Option<String> {
    let mut in_snapshot = false;

    for line in PINS.lines().map(str::trim) {
        if line.starts_with('[') {
            in_snapshot = line == "[snapshot]";
            continue;
        }
        if in_snapshot && line.starts_with("name") {
            return line
                .split_once('=')
                .map(|(_, value)| value.trim().trim_matches('"').to_owned());
        }
    }

    None
}

// ANCHOR: example
/// 探测“固定快照 → Device → Flex 后端 → Tensor 执行”的最短路径。
pub fn probe_execution_stack() -> Result<StackReport, ProbeError> {
    let device = Device::flex();
    let settings = device.settings();

    device.seed(42);
    let tensor = Tensor::<1>::from_floats([7.0], &device);
    device
        .sync()
        .map_err(|error| ProbeError::Sync(error.to_string()))?;
    let values = tensor
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| ProbeError::Data(error.to_string()))?;
    let observed_value = values
        .first()
        .copied()
        .ok_or_else(|| ProbeError::Data("后端返回了空张量数据".to_owned()))?;

    Ok(StackReport {
        snapshot: snapshot_name().ok_or(ProbeError::MissingSnapshot)?,
        device: format!("{device:?}"),
        float_dtype: format!("{:?}", settings.float_dtype),
        int_dtype: format!("{:?}", settings.int_dtype),
        autodiff_enabled: device.is_autodiff(),
        observed_value,
    })
}
// ANCHOR_END: example

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_pinned_flex_execution() {
        let report = probe_execution_stack().expect("Flex CPU 探测应当成功");

        assert_eq!(report.snapshot, "burn-0.22.0-pre.1");
        assert!(report.device.contains("Flex"));
        assert_eq!(report.float_dtype, "F32");
        assert_eq!(report.int_dtype, "I32");
        assert!(!report.autodiff_enabled);
        assert_eq!(report.observed_value, 7.0);
    }

    #[test]
    fn embedded_pins_contain_the_burn_revision() {
        assert!(PINS.contains("976aa9c5ec1d2dd3412710f99759e3c44bdff03d"));
    }
}
