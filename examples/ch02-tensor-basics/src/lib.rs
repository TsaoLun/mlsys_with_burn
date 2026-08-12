//! 第 2 章的张量、Module 与自动微分示例。

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{DType, Device, Tensor},
};

/// 教材示例读回数据或梯度失败。
#[derive(Debug)]
pub enum ExampleError {
    /// 张量无法按 F32 数据读回。
    Data(String),
    /// 被标记的叶子张量没有梯度。
    GradientMissing(&'static str),
}

impl Display for ExampleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(error) => write!(formatter, "读取 F32 张量失败：{error}"),
            Self::GradientMissing(name) => write!(formatter, "叶子张量 {name} 没有梯度"),
        }
    }
}

impl Error for ExampleError {}

fn values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>, ExampleError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| ExampleError::Data(error.to_string()))
}

// ANCHOR: example
/// 在 CPU 上执行逐元素运算，并把结果带回主机。
pub fn average_vectors() -> Result<Vec<f32>, ExampleError> {
    let device = Device::flex();
    let left = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
    let right = Tensor::<1>::from_floats([3.0, 4.0, 5.0], &device);
    let average = (left + right) / 2.0;

    values(average)
}
// ANCHOR_END: example

/// 广播实验的可观察结果。
#[derive(Debug, Clone, PartialEq)]
pub struct BroadcastReport {
    /// 广播后的二维形状。
    pub dims: [usize; 2],
    /// 按行展开的结果。
    pub values: Vec<f32>,
}

// ANCHOR: broadcasting
/// 展示编译期秩与运行时形状，以及二维广播。
pub fn broadcast_rows_and_columns() -> Result<BroadcastReport, ExampleError> {
    let device = Device::flex();
    let column = Tensor::<2>::from_floats([[1.0], [2.0], [3.0]], &device);
    let row = Tensor::<2>::from_floats([[10.0, 20.0]], &device);
    let result = column + row;
    let dims = result.dims();

    Ok(BroadcastReport {
        dims,
        values: values(result)?,
    })
}
// ANCHOR_END: broadcasting

/// 乘法前向结果及两个叶子张量的梯度。
#[derive(Debug, Clone, PartialEq)]
pub struct GradientReport {
    /// `left * right` 的前向值。
    pub product: Vec<f32>,
    /// 对 `left` 的梯度，等于 `right`。
    pub left_gradient: Vec<f32>,
    /// 对 `right` 的梯度，等于 `left`。
    pub right_gradient: Vec<f32>,
}

/// Byte-level facts about a tensor's host-side representation.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorBytesReport {
    /// Runtime shape of the tensor.
    pub dims: [usize; 2],
    /// Runtime element type.
    pub dtype: DType,
    /// Raw byte length of the f32 payload.
    pub byte_len: usize,
    /// Little-endian bytes of the first element (1.0).
    pub first_element_bytes: [u8; 4],
    /// Little-endian bytes of the last element (-0.0).
    pub negative_zero_bytes: [u8; 4],
    /// Byte length of the same values after converting to f64.
    pub f64_byte_len: usize,
}

// ANCHOR: tensor_bytes
/// Read a tensor back and inspect its raw bytes, shape, and dtype.
///
/// `TensorData` is just `bytes + shape + dtype`: the bytes are the content,
/// the shape says how to group them, and the dtype says how wide each element
/// is. Converting dtype changes the byte width without changing the shape.
pub fn inspect_tensor_bytes() -> TensorBytesReport {
    let device = Device::flex();
    let tensor = Tensor::<2>::from_floats([[1.0, -2.0], [3.5, 0.25], [0.0, -0.0]], &device);
    let data = tensor.into_data();
    let bytes = data.as_bytes();
    let first_element_bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let tail = bytes.len() - 4;
    let negative_zero_bytes = [
        bytes[tail],
        bytes[tail + 1],
        bytes[tail + 2],
        bytes[tail + 3],
    ];

    let f64_byte_len = data.clone().convert_dtype(DType::F64).as_bytes().len();

    TensorBytesReport {
        dims: data.shape.dims(),
        dtype: data.dtype,
        byte_len: bytes.len(),
        first_element_bytes,
        negative_zero_bytes,
        f64_byte_len,
    }
}
// ANCHOR_END: tensor_bytes

// ANCHOR: autodiff
/// 在 Flex 上构建动态反向图并读取两个叶子张量的梯度。
pub fn multiply_with_gradients() -> Result<GradientReport, ExampleError> {
    let device = Device::flex().autodiff();
    let left = Tensor::<1>::from_floats([1.0, 7.0], &device).require_grad();
    let right = Tensor::<1>::from_floats([4.0, 7.0], &device).require_grad();
    let product = left.clone() * right.clone();
    let gradients = product.backward();

    let left_gradient = left
        .grad(&gradients)
        .ok_or(ExampleError::GradientMissing("left"))?;
    let right_gradient = right
        .grad(&gradients)
        .ok_or(ExampleError::GradientMissing("right"))?;

    Ok(GradientReport {
        product: values(product)?,
        left_gradient: values(left_gradient)?,
        right_gradient: values(right_gradient)?,
    })
}
// ANCHOR_END: autodiff

/// 控制流分支实验：只走一侧前向路径时的梯度。
#[derive(Debug, Clone, PartialEq)]
pub struct BranchGradientReport {
    /// 实际执行的分支名。
    pub branch: &'static str,
    /// 前向输出。
    pub output: Vec<f32>,
    /// 对输入叶子的梯度。
    pub input_gradient: Vec<f32>,
}

// ANCHOR: branch_autodiff
/// Eager 控制流只把被执行分支记入 autodiff tape。
pub fn branch_gradient(use_double: bool) -> Result<BranchGradientReport, ExampleError> {
    let device = Device::flex().autodiff();
    let input = Tensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let output = if use_double {
        input.clone() * 2.0
    } else {
        input.clone() + 1.0
    };
    let gradients = output.clone().backward();
    let input_gradient = input
        .grad(&gradients)
        .ok_or(ExampleError::GradientMissing("input"))?;

    Ok(BranchGradientReport {
        branch: if use_double { "double" } else { "plus_one" },
        output: values(output)?,
        input_gradient: values(input_gradient)?,
    })
}
// ANCHOR_END: branch_autodiff

/// 观察 `detach` 产生的新 autodiff leaf 与原始 leaf 的梯度状态。
#[derive(Debug, Clone, PartialEq)]
pub struct DetachGradientReport {
    /// detached leaf 的前向输出。
    pub output: Vec<f32>,
    /// 原始 leaf 的梯度；只走 detached 分支时应为 `None`。
    pub original_gradient: Option<Vec<f32>>,
    /// detached leaf 的梯度。
    pub detached_gradient: Option<Vec<f32>>,
}

// ANCHOR: detach_autodiff
/// `detach` 切断原路径，再由 `require_grad` 建立一个新的 leaf。
pub fn detached_leaf_gradient() -> Result<DetachGradientReport, ExampleError> {
    let device = Device::flex().autodiff();
    let original = Tensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let detached = original.clone().detach().require_grad();
    let output = detached.clone() * 3.0;
    let gradients = output.clone().backward();

    let original_gradient = original.grad(&gradients).map(values).transpose()?;
    let detached_gradient = detached.grad(&gradients).map(values).transpose()?;

    Ok(DetachGradientReport {
        output: values(output)?,
        original_gradient,
        detached_gradient,
    })
}
// ANCHOR_END: detach_autodiff

// ANCHOR: module
/// 由一个线性层和激活函数组成的最小 Module。
#[derive(Module, Debug)]
pub struct TinyModel {
    projection: Linear,
    activation: Relu,
}

impl TinyModel {
    /// 在指定设备上初始化参数。
    pub fn new(device: &Device) -> Self {
        Self {
            projection: LinearConfig::new(3, 2).init(device),
            activation: Relu::new(),
        }
    }

    /// 输入形状为 `[batch, 3]`，输出形状为 `[batch, 2]`。
    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.activation.forward(self.projection.forward(input))
    }
}
// ANCHOR_END: module

/// 初始化最小 Module，返回参数量与前向输出形状。
pub fn inspect_tiny_model() -> (usize, [usize; 2]) {
    let device = Device::flex();
    device.seed(42);
    let model = TinyModel::new(&device);
    let input = Tensor::<2>::zeros([4, 3], &device);
    let output = model.forward(input);

    (model.num_params(), output.dims())
}

/// 比较普通 Flex Device 与 autodiff Device 的运行时能力标志。
pub fn inspect_device_modes() -> (bool, bool) {
    let plain = Device::flex();
    let autodiff = plain.clone().autodiff();

    (plain.is_autodiff(), autodiff.is_autodiff())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_elementwise_average() {
        assert_eq!(
            average_vectors().expect("F32 读回应成功"),
            vec![2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn broadcasts_runtime_shapes() {
        let report = broadcast_rows_and_columns().expect("广播实验应成功");

        assert_eq!(report.dims, [3, 2]);
        assert_eq!(report.values, vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
    }

    #[test]
    fn tensor_bytes_expose_ieee754_little_endian_layout() {
        let report = inspect_tensor_bytes();

        assert_eq!(report.dims, [3, 2]);
        assert_eq!(report.dtype, DType::F32);
        assert_eq!(report.byte_len, 6 * 4);
        assert_eq!(report.first_element_bytes, 1.0f32.to_le_bytes());
        // -0.0 compares equal to 0.0, but its sign bit is visible in the bytes.
        assert_eq!(report.negative_zero_bytes, (-0.0f32).to_le_bytes());
        assert_eq!(report.negative_zero_bytes, [0x00, 0x00, 0x00, 0x80]);
        assert_eq!(report.f64_byte_len, 6 * 8);
    }

    #[test]
    fn computes_gradients_on_flex() {
        let report = multiply_with_gradients().expect("Flex 自动微分应成功");

        assert_eq!(report.product, vec![4.0, 49.0]);
        assert_eq!(report.left_gradient, vec![4.0, 7.0]);
        assert_eq!(report.right_gradient, vec![1.0, 7.0]);
    }

    #[test]
    fn module_registers_parameters_and_preserves_batch_shape() {
        let (num_params, output_dims) = inspect_tiny_model();

        assert_eq!(num_params, 8);
        assert_eq!(output_dims, [4, 2]);
    }

    #[test]
    fn device_reports_autodiff_mode_separately() {
        assert_eq!(inspect_device_modes(), (false, true));
    }

    #[test]
    fn autodiff_tape_follows_executed_branch_only() {
        let doubled = branch_gradient(true).expect("double 分支应成功");
        let plus_one = branch_gradient(false).expect("plus_one 分支应成功");

        assert_eq!(doubled.branch, "double");
        assert_eq!(doubled.output, vec![4.0, 6.0]);
        assert_eq!(doubled.input_gradient, vec![2.0, 2.0]);

        assert_eq!(plus_one.branch, "plus_one");
        assert_eq!(plus_one.output, vec![3.0, 4.0]);
        assert_eq!(plus_one.input_gradient, vec![1.0, 1.0]);
    }

    #[test]
    fn detach_creates_a_new_leaf_and_cuts_the_original_path() {
        let report = detached_leaf_gradient().expect("detach 语义实验应成功");

        assert_eq!(report.output, vec![6.0, 9.0]);
        assert_eq!(report.original_gradient, None);
        assert_eq!(report.detached_gradient, Some(vec![3.0, 3.0]));
        assert_eq!(report.detached_gradient.as_ref().map(Vec::len), Some(2));
    }
}
