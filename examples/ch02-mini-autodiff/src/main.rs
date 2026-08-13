use ch02_mini_autodiff::Tape;

// ANCHOR: walkthrough
fn main() {
    // w = relu(x*y + x)，x = 2, y = 3。
    let mut tape = Tape::new();
    let x = tape.leaf(2.0);
    let y = tape.leaf(3.0);
    let xy = tape.mul(x, y);
    let s = tape.add(xy, x);
    let w = tape.relu(s);
    tape.backward(w);

    println!("按追加顺序打开 tape（编号即拓扑序）：");
    println!("{:>4}  {:<5} {:>8}  {:>8}", "id", "op", "value", "grad");
    for (index, name, value, grad) in tape.rows() {
        println!("{index:>4}  {name:<5} {value:>8.3}  {grad:>8.3}");
    }
    println!(
        "dw/dx = {}（= y + 1），dw/dy = {}（= x）",
        tape.grad(x),
        tape.grad(y)
    );
}
// ANCHOR_END: walkthrough
