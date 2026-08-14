use ch02_ch04_op_anatomy::{
    max_abs_diff, mean_backward, mul_tanh_backward, sum_backward, tanh_backward, tanh_forward,
    tanh_grad_reference, tanh_reference,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = [-3.0, -1.5, -0.25, 0.0, 0.5, 1.0, 2.5];

    let forward = tanh_forward(&samples)?;
    println!(
        "前向（API → dispatch → Flex）与标量 f32::tanh 的最大误差：{:.2e}",
        max_abs_diff(&forward, &tanh_reference(&samples))
    );

    let grad = tanh_backward(&samples)?;
    println!(
        "反向（autodiff Tanh backward）与 1 - tanh(x)^2 的最大误差：{:.2e}",
        max_abs_diff(&grad, &tanh_grad_reference(&samples))
    );

    let composite = mul_tanh_backward(&samples)?;
    let expected: Vec<f32> = samples
        .iter()
        .map(|&x| x.tanh() + x * (1.0 - x.tanh() * x.tanh()))
        .collect();
    println!(
        "组合 y = x·tanh(x) 的梯度与乘积法则的最大误差：{:.2e}",
        max_abs_diff(&composite, &expected)
    );

    let sum_grad = sum_backward(&samples)?;
    let mean_grad = mean_backward(&samples)?;
    let n = samples.len() as f32;
    println!(
        "sum 反向全 1；mean 反向为 1/n={:.2e}，相对 sum/n 的最大误差：{:.2e}",
        1.0 / n,
        max_abs_diff(
            &mean_grad,
            &sum_grad.iter().map(|value| value / n).collect::<Vec<_>>()
        )
    );
    println!("以上断言逐层核对了解剖页引用的源码事实；GPU/Fusion 层见正文的源码路径。");
    Ok(())
}
