//! 第 2 章的张量、Module 与自动微分示例。

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Device, Tensor},
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
}
