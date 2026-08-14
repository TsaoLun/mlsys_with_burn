//! 「算子解剖」贯穿页的可运行验证：书里逐层引用的 `tanh` 源码事实，
//! 在这里全部变成断言。
//!
//! 三个观察点对应解剖页的三段结论：API 层的 `tanh` 最终落到 Flex 的
//! 元素级 `f32::tanh`；autodiff 层注册的反向规则确实计算
//! `grad * (1 - tanh(x)^2)`；tape 会把多个算子的反向规则按链式法则
//! 组合起来。

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use burn::tensor::{Device, Tensor};

/// 读回数据或梯度失败。
#[derive(Debug)]
pub enum AnatomyError {
    /// 张量无法按 F32 数据读回。
    Data(String),
    /// 被标记的叶子张量没有梯度。
    GradientMissing,
}

impl Display for AnatomyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(error) => write!(formatter, "读取 F32 张量失败：{error}"),
            Self::GradientMissing => formatter.write_str("require_grad 的叶子没有梯度"),
        }
    }
}

impl Error for AnatomyError {}

fn values(tensor: Tensor<1>) -> Result<Vec<f32>, AnatomyError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| AnatomyError::Data(format!("{error:?}")))
}

/// 元素级参考：`Tensor::tanh` 的语义权威就是标量 `f32::tanh`。
pub fn tanh_reference(inputs: &[f32]) -> Vec<f32> {
    inputs.iter().map(|value| value.tanh()).collect()
}

/// 解析梯度参考：d tanh(x)/dx = 1 - tanh(x)^2。
pub fn tanh_grad_reference(inputs: &[f32]) -> Vec<f32> {
    inputs
        .iter()
        .map(|value| 1.0 - value.tanh() * value.tanh())
        .collect()
}

// ANCHOR: forward
/// 前向：走 Burn 的完整调用链（API → dispatch → Flex 元素级实现），
/// 把结果读回主机。
pub fn tanh_forward(inputs: &[f32]) -> Result<Vec<f32>, AnatomyError> {
    let device = Device::flex();
    let tensor = Tensor::<1>::from_floats(inputs, &device);
    values(tensor.tanh())
}
// ANCHOR_END: forward

// ANCHOR: backward
/// 反向：autodiff 装饰器记录 tanh，backward 后取回输入梯度。书中
/// 引用的 `Tanh` backward 实现应精确给出 `1 - tanh(x)^2`
/// （根梯度为全 1）。
pub fn tanh_backward(inputs: &[f32]) -> Result<Vec<f32>, AnatomyError> {
    let device = Device::flex().autodiff();
    let input = Tensor::<1>::from_floats(inputs, &device).require_grad();
    let gradients = input.clone().tanh().backward();
    let grad = input
        .grad(&gradients)
        .ok_or(AnatomyError::GradientMissing)?;
    values(grad)
}
// ANCHOR_END: backward

/// 组合：y = x · tanh(x) 的梯度应为 tanh(x) + x·(1 - tanh(x)^2)，
/// 验证 tape 按乘积法则组合两个算子的反向规则。
pub fn mul_tanh_backward(inputs: &[f32]) -> Result<Vec<f32>, AnatomyError> {
    let device = Device::flex().autodiff();
    let input = Tensor::<1>::from_floats(inputs, &device).require_grad();
    let gradients = input.clone().mul(input.clone().tanh()).backward();
    let grad = input
        .grad(&gradients)
        .ok_or(AnatomyError::GradientMissing)?;
    values(grad)
}

pub fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f32::max)
}

/// 对照解剖 `add`：广播加法中，被广播一侧的梯度是按广播维度的归约。
/// 返回 (b 的梯度形状, b 的梯度值)。
pub fn broadcast_add_backward() -> Result<([usize; 2], Vec<f32>), AnatomyError> {
    let device = Device::flex().autodiff();
    let a = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).require_grad();
    let b = Tensor::<2>::from_floats([[10.0, 20.0, 30.0]], &device).require_grad();
    let gradients = (a.clone() + b.clone()).backward();
    let grad_b = b.grad(&gradients).ok_or(AnatomyError::GradientMissing)?;
    let dims = grad_b.dims();
    let grad_values = grad_b
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| AnatomyError::Data(format!("{error:?}")))?;
    Ok((dims, grad_values))
}

/// 对照解剖 `sum`：归约的反向是广播——每个输入元素的梯度都是 1。
pub fn sum_backward(inputs: &[f32]) -> Result<Vec<f32>, AnatomyError> {
    let device = Device::flex().autodiff();
    let input = Tensor::<1>::from_floats(inputs, &device).require_grad();
    let gradients = input.clone().sum().backward();
    let grad = input
        .grad(&gradients)
        .ok_or(AnatomyError::GradientMissing)?;
    values(grad)
}

// ANCHOR: mean_backward
/// `mean` 是带缩放的归约：`mean(x) = sum(x)/n`，反向把上游梯度乘 `1/n`
/// 再广播回每个输入。根梯度为 1 时，每个元素的梯度恰好是 `1/n`。
pub fn mean_backward(inputs: &[f32]) -> Result<Vec<f32>, AnatomyError> {
    let device = Device::flex().autodiff();
    let input = Tensor::<1>::from_floats(inputs, &device).require_grad();
    let gradients = input.clone().mean().backward();
    let grad = input
        .grad(&gradients)
        .ok_or(AnatomyError::GradientMissing)?;
    values(grad)
}
// ANCHOR_END: mean_backward

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [f32; 7] = [-3.0, -1.5, -0.25, 0.0, 0.5, 1.0, 2.5];

    /// API → dispatch → Flex 的完整前向与标量 `f32::tanh` 一致。
    #[test]
    fn forward_matches_scalar_reference() {
        let forward = tanh_forward(&SAMPLES).expect("前向应可执行");
        assert!(max_abs_diff(&forward, &tanh_reference(&SAMPLES)) < 1e-6);
    }

    /// autodiff 注册的反向规则精确实现 1 - tanh(x)^2。
    #[test]
    fn backward_matches_analytic_derivative() {
        let grad = tanh_backward(&SAMPLES).expect("反向应可执行");
        assert!(max_abs_diff(&grad, &tanh_grad_reference(&SAMPLES)) < 1e-6);
    }

    /// 反向规则与中心差分数值梯度一致——不依赖解析式的独立核对。
    #[test]
    fn backward_matches_numeric_gradient() {
        let grad = tanh_backward(&SAMPLES).expect("反向应可执行");
        let eps = 1e-3f32;
        for (index, &value) in SAMPLES.iter().enumerate() {
            let numeric = ((value + eps).tanh() - (value - eps).tanh()) / (2.0 * eps);
            assert!(
                (grad[index] - numeric).abs() < 1e-4,
                "x={value}: 解析 {} vs 数值 {numeric}",
                grad[index]
            );
        }
    }

    /// add 的反向对被广播一侧做归约：b 形状 [1,3]，梯度按列求和
    /// （根梯度全 1 时即为每列的行数 2）。
    #[test]
    fn add_gradient_reduces_over_broadcast() {
        let (dims, grad) = broadcast_add_backward().expect("反向应可执行");
        assert_eq!(dims, [1, 3], "梯度形状必须回到被广播输入的形状");
        assert_eq!(grad, vec![2.0, 2.0, 2.0]);
    }

    /// sum 的反向是广播：每个输入元素的梯度恰为 1。
    #[test]
    fn sum_gradient_broadcasts_ones() {
        let grad = sum_backward(&SAMPLES).expect("反向应可执行");
        assert_eq!(grad, vec![1.0; SAMPLES.len()]);
    }

    /// mean 的反向是缩放广播：每个输入元素的梯度恰为 1/n。
    #[test]
    fn mean_gradient_is_scaled_broadcast() {
        let grad = mean_backward(&SAMPLES).expect("反向应可执行");
        let scale = 1.0 / SAMPLES.len() as f32;
        assert!(max_abs_diff(&grad, &vec![scale; SAMPLES.len()]) < 1e-6);
        let sum_grad = sum_backward(&SAMPLES).expect("sum 反向应可执行");
        for (mean_g, sum_g) in grad.iter().zip(&sum_grad) {
            assert!((mean_g * SAMPLES.len() as f32 - sum_g).abs() < 1e-5);
        }
    }

    /// tape 按乘积法则组合两个算子的反向规则。
    #[test]
    fn composite_gradient_follows_product_rule() {
        let grad = mul_tanh_backward(&SAMPLES).expect("反向应可执行");
        for (index, &value) in SAMPLES.iter().enumerate() {
            let expected = value.tanh() + value * (1.0 - value.tanh() * value.tanh());
            assert!(
                (grad[index] - expected).abs() < 1e-5,
                "x={value}: 实际 {} vs 期望 {expected}",
                grad[index]
            );
        }
    }
}
