//! 第 2 章的最小张量示例。

use burn::tensor::{Device, Tensor};

// ANCHOR: example
/// 在 CPU 上执行逐元素运算，并把结果带回主机。
pub fn average_vectors() -> Vec<f32> {
    let device = Device::flex();
    let left = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
    let right = Tensor::<1>::from_floats([3.0, 4.0, 5.0], &device);
    let average = (left + right) / 2.0;

    average
        .to_data()
        .to_vec::<f32>()
        .expect("示例使用 F32 张量，转换为 Vec<f32> 应当成功")
}
// ANCHOR_END: example

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_elementwise_average() {
        assert_eq!(average_vectors(), vec![2.0, 3.0, 4.0]);
    }
}
