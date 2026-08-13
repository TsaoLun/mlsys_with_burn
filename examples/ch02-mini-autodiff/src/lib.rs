//! 一个约百行的反向模式自动微分 tape。
//!
//! 目标不是替代 Burn，而是把第 2 章讲的三件事变成可运行、可断言的
//! 最小实现：tape 按执行顺序追加、扇出的梯度沿反向累加、只有真正
//! 执行过的分支才会被记录。

/// tape 上一个值的编号。追加顺序天然是一个拓扑序：任何操作节点都在
/// 它的输入之后入表。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VarId(usize);

// ANCHOR: op
/// tape 记录的原语。每个节点只保存「我是哪种运算、输入是谁」，
/// 反向规则由 [`Tape::backward`] 按类型分派。
#[derive(Clone, Copy, Debug)]
enum Op {
    /// 叶子：输入或常量，没有上游。
    Leaf,
    /// 加法：两条输入边都把梯度原样传回。
    Add(VarId, VarId),
    /// 乘法：梯度乘以「另一个输入的前向值」——这就是 tape 必须
    /// 保存前向中间值的原因。
    Mul(VarId, VarId),
    /// ReLU：前向值大于零时透传梯度，否则截断。
    Relu(VarId),
}
// ANCHOR_END: op

/// 记录前向执行并支持一次反向传播的 tape。
#[derive(Default)]
pub struct Tape {
    values: Vec<f64>,
    grads: Vec<f64>,
    ops: Vec<Op>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, value: f64, op: Op) -> VarId {
        self.values.push(value);
        self.grads.push(0.0);
        self.ops.push(op);
        VarId(self.ops.len() - 1)
    }

    /// 建立一个叶子（输入或常量）。
    pub fn leaf(&mut self, value: f64) -> VarId {
        self.push(value, Op::Leaf)
    }

    pub fn add(&mut self, a: VarId, b: VarId) -> VarId {
        self.push(self.values[a.0] + self.values[b.0], Op::Add(a, b))
    }

    pub fn mul(&mut self, a: VarId, b: VarId) -> VarId {
        self.push(self.values[a.0] * self.values[b.0], Op::Mul(a, b))
    }

    pub fn relu(&mut self, a: VarId) -> VarId {
        self.push(self.values[a.0].max(0.0), Op::Relu(a))
    }

    /// 与 Burn 的 `detach` 同义：值照抄，但作为新叶子入表，切断
    /// 反向路径。
    pub fn detach(&mut self, a: VarId) -> VarId {
        self.leaf(self.values[a.0])
    }

    pub fn value(&self, id: VarId) -> f64 {
        self.values[id.0]
    }

    /// 上一次 [`Tape::backward`] 之后，`id` 的梯度。
    pub fn grad(&self, id: VarId) -> f64 {
        self.grads[id.0]
    }

    /// tape 里已经记录的节点数——用来观察「哪些操作真的执行了」。
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    // ANCHOR: backward
    /// 从 `root` 反向传播。梯度清零后种下 d root/d root = 1，再按
    /// 编号从大到小扫一遍：因为追加顺序就是拓扑序，反向遍历保证
    /// 读取每个节点梯度时，它的所有消费者都已经把贡献累加进来。
    pub fn backward(&mut self, root: VarId) {
        for grad in &mut self.grads {
            *grad = 0.0;
        }
        self.grads[root.0] = 1.0;

        for index in (0..=root.0).rev() {
            let grad = self.grads[index];
            if grad == 0.0 {
                continue;
            }
            match self.ops[index] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    self.grads[a.0] += grad;
                    self.grads[b.0] += grad;
                }
                Op::Mul(a, b) => {
                    // 先取值再写梯度，避免 a == b（例如 x*x）时读到
                    // 写了一半的状态；扇入同一节点时两次 += 自然叠加。
                    let (va, vb) = (self.values[a.0], self.values[b.0]);
                    self.grads[a.0] += grad * vb;
                    self.grads[b.0] += grad * va;
                }
                Op::Relu(a) => {
                    if self.values[a.0] > 0.0 {
                        self.grads[a.0] += grad;
                    }
                }
            }
        }
    }
    // ANCHOR_END: backward

    /// 按 tape 顺序导出（编号、算子名、前向值、当前梯度），供打印
    /// 或测试检查。
    pub fn rows(&self) -> Vec<(usize, &'static str, f64, f64)> {
        self.ops
            .iter()
            .enumerate()
            .map(|(index, op)| {
                let name = match op {
                    Op::Leaf => "leaf",
                    Op::Add(_, _) => "add",
                    Op::Mul(_, _) => "mul",
                    Op::Relu(_) => "relu",
                };
                (index, name, self.values[index], self.grads[index])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// w = relu(x*y + x)，在 x=2, y=3 处：w = 8，dw/dx = y+1 = 4，
    /// dw/dy = x = 2。
    #[test]
    fn chain_rule_matches_hand_computation() {
        let mut tape = Tape::new();
        let x = tape.leaf(2.0);
        let y = tape.leaf(3.0);
        let xy = tape.mul(x, y);
        let s = tape.add(xy, x);
        let w = tape.relu(s);
        tape.backward(w);

        assert_eq!(tape.value(w), 8.0);
        assert_eq!(tape.grad(x), 4.0);
        assert_eq!(tape.grad(y), 2.0);
    }

    /// u = x * x：同一 VarId 作为两个输入，梯度必须累加成 2x。
    #[test]
    fn fan_out_accumulates_gradients() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.0);
        let u = tape.mul(x, x);
        tape.backward(u);

        assert_eq!(tape.grad(x), 6.0);
    }

    /// relu 关死的一侧不透传梯度。
    #[test]
    fn relu_blocks_gradient_when_negative() {
        let mut tape = Tape::new();
        let x = tape.leaf(-1.5);
        let r = tape.relu(x);
        tape.backward(r);

        assert_eq!(tape.value(r), 0.0);
        assert_eq!(tape.grad(x), 0.0);
    }

    /// 用中心差分核对解析梯度：f(x, y) = relu(x*y + x) * y。
    #[test]
    fn numeric_gradient_check() {
        fn f(x: f64, y: f64) -> f64 {
            let mut tape = Tape::new();
            let vx = tape.leaf(x);
            let vy = tape.leaf(y);
            let xy = tape.mul(vx, vy);
            let s = tape.add(xy, vx);
            let r = tape.relu(s);
            let out = tape.mul(r, vy);
            tape.value(out)
        }

        let (x, y) = (1.25, -0.5);
        let mut tape = Tape::new();
        let vx = tape.leaf(x);
        let vy = tape.leaf(y);
        let xy = tape.mul(vx, vy);
        let s = tape.add(xy, vx);
        let r = tape.relu(s);
        let out = tape.mul(r, vy);
        tape.backward(out);

        let eps = 1e-6;
        let dx_numeric = (f(x + eps, y) - f(x - eps, y)) / (2.0 * eps);
        let dy_numeric = (f(x, y + eps) - f(x, y - eps)) / (2.0 * eps);

        assert!((tape.grad(vx) - dx_numeric).abs() < 1e-6);
        assert!((tape.grad(vy) - dy_numeric).abs() < 1e-6);
    }

    /// 只有执行过的分支进入 tape：这就是第 2 章「tape 只记录实际
    /// 路径」的最小复现。
    #[test]
    fn only_executed_branch_is_recorded() {
        fn run(take_mul: bool) -> (usize, f64) {
            let mut tape = Tape::new();
            let x = tape.leaf(2.0);
            let y = tape.leaf(5.0);
            let out = if take_mul {
                tape.mul(x, y)
            } else {
                tape.add(x, y)
            };
            tape.backward(out);
            (tape.len(), tape.grad(x))
        }

        let (len_mul, grad_mul) = run(true);
        let (len_add, grad_add) = run(false);

        // 两条路径的 tape 一样长：都只多了一个节点，但梯度规则不同。
        assert_eq!(len_mul, 3);
        assert_eq!(len_add, 3);
        assert_eq!(grad_mul, 5.0);
        assert_eq!(grad_add, 1.0);
    }

    /// detach 把值抄成新叶子：下游照常用值，梯度不再流回上游。
    #[test]
    fn detach_stops_gradient() {
        let mut tape = Tape::new();
        let x = tape.leaf(3.0);
        let doubled = tape.mul(x, x);
        let frozen = tape.detach(doubled);
        let out = tape.mul(frozen, x);
        tape.backward(out);

        // dout/dx 只剩下「frozen 当作常数 9」的那一条路径。
        assert_eq!(tape.grad(x), 9.0);
        assert_eq!(tape.grad(doubled), 0.0);
        assert_eq!(tape.grad(frozen), 3.0);
    }

    /// 再次 backward 会重置梯度，而不是叠加两次结果。
    #[test]
    fn backward_resets_previous_gradients() {
        let mut tape = Tape::new();
        let x = tape.leaf(4.0);
        let u = tape.mul(x, x);
        tape.backward(u);
        tape.backward(u);

        assert_eq!(tape.grad(x), 8.0);
    }
}
