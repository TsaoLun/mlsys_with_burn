use std::process::ExitCode;

use ch02_tensor_basics::{
    average_vectors, branch_gradient, broadcast_rows_and_columns, inspect_tiny_model,
    multiply_with_gradients,
};

fn main() -> ExitCode {
    let result = (|| {
        println!("逐元素平均值：{:?}", average_vectors()?);

        let broadcast = broadcast_rows_and_columns()?;
        println!("广播：{:?} -> {:?}", broadcast.dims, broadcast.values);

        let gradients = multiply_with_gradients()?;
        println!("乘法前向值：{:?}", gradients.product);
        println!("left 梯度：{:?}", gradients.left_gradient);
        println!("right 梯度：{:?}", gradients.right_gradient);

        let (params, output_dims) = inspect_tiny_model();
        println!("TinyModel：{params} 个参数，输出形状 {output_dims:?}");

        let branch = branch_gradient(true)?;
        println!(
            "控制流分支 {}：输出 {:?}，梯度 {:?}",
            branch.branch, branch.output, branch.input_gradient
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
