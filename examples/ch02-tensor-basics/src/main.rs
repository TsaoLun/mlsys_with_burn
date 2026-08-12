use std::process::ExitCode;

use ch02_tensor_basics::{
    average_vectors, branch_gradient, broadcast_rows_and_columns, detached_leaf_gradient,
    inspect_device_modes, inspect_tensor_bytes, inspect_tiny_model, multiply_with_gradients,
};

fn main() -> ExitCode {
    let result = (|| {
        println!("逐元素平均值：{:?}", average_vectors()?);

        let broadcast = broadcast_rows_and_columns()?;
        println!("广播：{:?} -> {:?}", broadcast.dims, broadcast.values);

        let bytes = inspect_tensor_bytes();
        println!(
            "张量字节：dims={:?} dtype={:?} bytes={} 首元素(1.0)={:02x?} 末元素(-0.0)={:02x?} f64_bytes={}",
            bytes.dims,
            bytes.dtype,
            bytes.byte_len,
            bytes.first_element_bytes,
            bytes.negative_zero_bytes,
            bytes.f64_byte_len
        );

        let gradients = multiply_with_gradients()?;
        println!("乘法前向值：{:?}", gradients.product);
        println!("left 梯度：{:?}", gradients.left_gradient);
        println!("right 梯度：{:?}", gradients.right_gradient);

        let (params, output_dims) = inspect_tiny_model();
        println!("TinyModel：{params} 个参数，输出形状 {output_dims:?}");

        let (plain_autodiff, wrapped_autodiff) = inspect_device_modes();
        println!("Device autodiff：普通={plain_autodiff}，autodiff 包装={wrapped_autodiff}");

        let branch = branch_gradient(true)?;
        println!(
            "控制流分支 {}：输出 {:?}，梯度 {:?}",
            branch.branch, branch.output, branch.input_gradient
        );

        let detach = detached_leaf_gradient()?;
        println!(
            "detach：原始梯度={:?}，新 leaf 梯度={:?}",
            detach.original_gradient, detach.detached_gradient
        );

        Ok::<(), ch02_tensor_basics::ExampleError>(())
    })();

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("第二章示例失败：{error}");
            ExitCode::FAILURE
        }
    }
}
