//! 迷你表达式 IR 与四个经典 Pass：常量折叠、死代码消除、公共子表达式
//! 消除，以及一个**故意非法**的 fast-math 消去改写。
//!
//! 目标是把第 4 章的 Pass 契约变成可运行断言：每个合法 Pass 的测试
//! 都验证「优化前后语义等价」，非法改写的测试则精确复现正文的
//! `(10^16 + 1) - 10^16` 浮点反例。数值语义与解释器一致（`f32`），
//! 因此合法 Pass 可以按位比较结果。

use std::collections::HashMap;

pub type NodeId = usize;

// ANCHOR: ir
/// 迷你 IR 的全部算子。所有算子逐元素（这里退化为标量），节点按
/// 创建顺序编号，任何节点的输入编号都小于自身——追加顺序即拓扑序。
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Op {
    /// 第 `usize` 个外部输入。
    Input(usize),
    Const(f32),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Exp(NodeId),
}

/// 一张纯数据的表达式图：节点表 + 输出列表。
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub ops: Vec<Op>,
    pub outputs: Vec<NodeId>,
}
// ANCHOR_END: ir

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, op: Op) -> NodeId {
        self.ops.push(op);
        self.ops.len() - 1
    }

    pub fn input(&mut self, index: usize) -> NodeId {
        self.push(Op::Input(index))
    }

    pub fn constant(&mut self, value: f32) -> NodeId {
        self.push(Op::Const(value))
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Op::Add(a, b))
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Op::Mul(a, b))
    }

    pub fn exp(&mut self, a: NodeId) -> NodeId {
        self.push(Op::Exp(a))
    }

    pub fn mark_output(&mut self, id: NodeId) {
        self.outputs.push(id);
    }

    /// 解释执行：按编号顺序求值一遍（拓扑序保证输入已就绪）。
    pub fn evaluate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let value = match *op {
                Op::Input(index) => inputs[index],
                Op::Const(value) => value,
                Op::Add(a, b) => values[a] + values[b],
                Op::Mul(a, b) => values[a] * values[b],
                Op::Exp(a) => f32::exp(values[a]),
            };
            values.push(value);
        }
        self.outputs.iter().map(|&id| values[id]).collect()
    }

    /// 每个节点被多少个下游节点消费（不含 output 标记）。
    fn consumer_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.ops.len()];
        for op in &self.ops {
            match *op {
                Op::Add(a, b) | Op::Mul(a, b) => {
                    counts[a] += 1;
                    counts[b] += 1;
                }
                Op::Exp(a) => counts[a] += 1,
                Op::Input(_) | Op::Const(_) => {}
            }
        }
        counts
    }

    /// 文本 dump，一行一个节点，供打印与书内展示。
    pub fn dump(&self) -> String {
        let mut text = String::new();
        for (id, op) in self.ops.iter().enumerate() {
            let line = match *op {
                Op::Input(index) => format!("%{id}: input#{index}"),
                Op::Const(value) => format!("%{id}: const {value}"),
                Op::Add(a, b) => format!("%{id}: add %{a} %{b}"),
                Op::Mul(a, b) => format!("%{id}: mul %{a} %{b}"),
                Op::Exp(a) => format!("%{id}: exp %{a}"),
            };
            text.push_str(&line);
            text.push('\n');
        }
        let outputs: Vec<String> = self.outputs.iter().map(|id| format!("%{id}")).collect();
        text.push_str(&format!("outputs: {}\n", outputs.join(", ")));
        text
    }
}

/// 把旧图的节点经 `remap` 重写进新图；所有 Pass 都用这一套「重建 +
/// 编号映射」骨架，天然保持拓扑序。
fn rebuild(old: &Graph, mut rewrite: impl FnMut(&mut Graph, NodeId, Op) -> NodeId) -> Graph {
    let mut new = Graph::new();
    let mut remap: Vec<NodeId> = Vec::with_capacity(old.ops.len());
    for (id, op) in old.ops.iter().enumerate() {
        let mapped = match *op {
            Op::Add(a, b) => Op::Add(remap[a], remap[b]),
            Op::Mul(a, b) => Op::Mul(remap[a], remap[b]),
            Op::Exp(a) => Op::Exp(remap[a]),
            other => other,
        };
        remap.push(rewrite(&mut new, id, mapped));
    }
    new.outputs = old.outputs.iter().map(|&id| remap[id]).collect();
    new
}

// ANCHOR: constant_fold
/// 常量折叠：输入都是 `Const` 的节点在编译期求值。求值用与解释器
/// 完全相同的 `f32` 运算，因此结果按位一致——这是它合法的原因。
pub fn constant_fold(graph: &Graph) -> Graph {
    rebuild(graph, |new, _, op| {
        let folded = match op {
            Op::Add(a, b) => match (new.ops[a], new.ops[b]) {
                (Op::Const(x), Op::Const(y)) => Some(x + y),
                _ => None,
            },
            Op::Mul(a, b) => match (new.ops[a], new.ops[b]) {
                (Op::Const(x), Op::Const(y)) => Some(x * y),
                _ => None,
            },
            Op::Exp(a) => match new.ops[a] {
                Op::Const(x) => Some(f32::exp(x)),
                _ => None,
            },
            _ => None,
        };
        match folded {
            Some(value) => new.push(Op::Const(value)),
            None => new.push(op),
        }
    })
}
// ANCHOR_END: constant_fold

/// 死代码消除：从 outputs 反向标记可达节点，重建时丢弃其余节点。
pub fn dead_code_elimination(graph: &Graph) -> Graph {
    let mut live = vec![false; graph.ops.len()];
    let mut stack: Vec<NodeId> = graph.outputs.clone();
    while let Some(id) = stack.pop() {
        if live[id] {
            continue;
        }
        live[id] = true;
        match graph.ops[id] {
            Op::Add(a, b) | Op::Mul(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            Op::Exp(a) => stack.push(a),
            Op::Input(_) | Op::Const(_) => {}
        }
    }

    // 不变量：live 节点的输入必为 live，且编号更小、先被重映射；
    // 用哨兵值代替 Option，可让违例在测试里立即以越界暴露。
    const UNMAPPED: NodeId = NodeId::MAX;
    let mut new = Graph::new();
    let mut remap: Vec<NodeId> = vec![UNMAPPED; graph.ops.len()];
    for (id, op) in graph.ops.iter().enumerate() {
        if !live[id] {
            continue;
        }
        let mapped = match *op {
            Op::Add(a, b) => Op::Add(remap[a], remap[b]),
            Op::Mul(a, b) => Op::Mul(remap[a], remap[b]),
            Op::Exp(a) => Op::Exp(remap[a]),
            other => other,
        };
        remap[id] = new.push(mapped);
    }
    new.outputs = graph.outputs.iter().map(|&id| remap[id]).collect();
    new
}

/// 结构化键：`Const` 用位模式比较（NaN 也稳定），其余用算子加输入
/// 编号。这就是 CSE 契约里「操作与输入都等价」的最小体现。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Key {
    Input(usize),
    Const(u32),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Exp(NodeId),
}

/// 公共子表达式消除：结构相同的节点合并为第一次出现的那个。
/// 注意它**不做**交换律归一化：`add %0 %1` 与 `add %1 %0` 不合并，
/// 那是练习。
pub fn common_subexpression_elimination(graph: &Graph) -> Graph {
    let mut seen: HashMap<Key, NodeId> = HashMap::new();
    rebuild(graph, |new, _, op| {
        let key = match op {
            Op::Input(index) => Key::Input(index),
            Op::Const(value) => Key::Const(value.to_bits()),
            Op::Add(a, b) => Key::Add(a, b),
            Op::Mul(a, b) => Key::Mul(a, b),
            Op::Exp(a) => Key::Exp(a),
        };
        if let Some(&existing) = seen.get(&key) {
            existing
        } else {
            let id = new.push(op);
            seen.insert(key, id);
            id
        }
    })
}

// ANCHOR: fast_math
/// **故意非法**的 fast-math 改写：把 `(x + c) + (-c)` 消去为 `x`。
/// 数学上恒等，但 `f32` 里大 `c` 会先吃掉 `x` 的尾数——见测试
/// `fast_math_cancellation_changes_float_results` 对正文反例的复现。
pub fn fast_math_cancel(graph: &Graph) -> Graph {
    rebuild(graph, |new, _, op| {
        if let Op::Add(outer, c2) = op
            && let (Op::Add(x, c1), Op::Const(v2)) = (new.ops[outer], new.ops[c2])
            && let Op::Const(v1) = new.ops[c1]
            && v1 == -v2
        {
            return x;
        }
        new.push(op)
    })
}
// ANCHOR_END: fast_math

// ANCHOR: fusion
/// 贪心融合分组（只分析、不改写）：一个节点若只有一个消费者、且自身
/// 不是对外输出，就并进消费者所在的组。返回每个计算节点所属组号；
/// 组数即「一个融合块一次 launch」模型下的 Kernel 数。
pub fn fusion_groups(graph: &Graph) -> Vec<usize> {
    let consumers = graph.consumer_counts();
    let mut is_output = vec![false; graph.ops.len()];
    for &id in &graph.outputs {
        is_output[id] = true;
    }

    // 组号沿唯一消费者传播；多消费者或对外可见的节点开新组。
    let mut group: Vec<usize> = vec![usize::MAX; graph.ops.len()];
    let mut next_group = 0;
    for id in (0..graph.ops.len()).rev() {
        if matches!(graph.ops[id], Op::Input(_) | Op::Const(_)) {
            continue;
        }
        if group[id] == usize::MAX {
            group[id] = next_group;
            next_group += 1;
        }
        let inherit = group[id];
        let mut assign = |input: NodeId| {
            if matches!(graph.ops[input], Op::Input(_) | Op::Const(_)) {
                return;
            }
            // 中间值一旦被多个下游读取或本身是输出，就必须物化，
            // 不能再融进同一组——与第 4 章的同步切分观察同构。
            if consumers[input] == 1 && !is_output[input] {
                group[input] = inherit;
            }
        };
        match graph.ops[id] {
            Op::Add(a, b) | Op::Mul(a, b) => {
                assign(a);
                assign(b);
            }
            Op::Exp(a) => assign(a),
            Op::Input(_) | Op::Const(_) => {}
        }
    }
    group
}

/// 融合组的数量（忽略 Input/Const 这类不计算的节点）。
pub fn fusion_group_count(graph: &Graph) -> usize {
    let groups = fusion_groups(graph);
    let mut distinct: Vec<usize> = groups
        .into_iter()
        .filter(|&group| group != usize::MAX)
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len()
}
// ANCHOR_END: fusion

/// fold → DCE → CSE 依次执行一轮。
pub fn standard_pipeline(graph: &Graph) -> Graph {
    common_subexpression_elimination(&dead_code_elimination(&constant_fold(graph)))
}

/// 确定性伪随机图，用于「任意图上语义不变」的批量断言。
pub fn random_graph(seed: u64, node_budget: usize) -> Graph {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    let mut graph = Graph::new();
    graph.input(0);
    graph.input(1);
    graph.constant(0.5);
    while graph.ops.len() < node_budget {
        let len = graph.ops.len();
        let a = next() as usize % len;
        let b = next() as usize % len;
        match next() % 4 {
            0 => graph.add(a, b),
            1 => graph.mul(a, b),
            2 => graph.exp(a),
            _ => graph.constant((next() % 7) as f32 - 3.0),
        };
    }
    // 随机挑两个节点作为输出，让 DCE 有事可做。
    let first = graph.ops.len() - 1;
    let second = next() as usize % graph.ops.len();
    graph.mark_output(first);
    graph.mark_output(second);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// 2*3 + 4 折叠成单个常量 10，且输出不变。
    #[test]
    fn constant_folding_shrinks_and_preserves_semantics() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let two = graph.constant(2.0);
        let three = graph.constant(3.0);
        let six = graph.mul(two, three);
        let four = graph.constant(4.0);
        let ten = graph.add(six, four);
        let out = graph.add(x, ten);
        graph.mark_output(out);

        let folded = constant_fold(&graph);
        let cleaned = dead_code_elimination(&folded);

        assert_eq!(graph.evaluate(&[1.5]), folded.evaluate(&[1.5]));
        assert!(cleaned.ops.len() < graph.ops.len());
        assert!(
            cleaned
                .ops
                .iter()
                .any(|op| matches!(op, Op::Const(v) if *v == 10.0))
        );
    }

    /// 死子树被整棵删除，输出不变。
    #[test]
    fn dce_removes_unreachable_subtree() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let y = graph.input(1);
        let live = graph.add(x, y);
        let dead_c = graph.constant(3.0);
        let dead = graph.mul(y, dead_c);
        let _dead_tail = graph.exp(dead);
        graph.mark_output(live);

        let cleaned = dead_code_elimination(&graph);

        assert_eq!(cleaned.ops.len(), 3);
        assert_eq!(
            bits(&graph.evaluate(&[0.25, -2.0])),
            bits(&cleaned.evaluate(&[0.25, -2.0]))
        );
    }

    /// 两棵结构相同的 exp(x*y) 合并为一棵。
    #[test]
    fn cse_merges_duplicate_subexpressions() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let y = graph.input(1);
        let first = graph.mul(x, y);
        let second = graph.mul(x, y);
        let e1 = graph.exp(first);
        let e2 = graph.exp(second);
        let out = graph.add(e1, e2);
        graph.mark_output(out);

        let merged = common_subexpression_elimination(&graph);

        assert_eq!(merged.ops.len(), graph.ops.len() - 2);
        assert_eq!(
            bits(&graph.evaluate(&[1.25, 0.5])),
            bits(&merged.evaluate(&[1.25, 0.5]))
        );
    }

    /// 标准流水线在一批随机图上保持输出按位一致——合法 Pass 的
    /// 总契约。
    #[test]
    fn standard_pipeline_preserves_semantics_on_random_graphs() {
        for seed in 0..16 {
            let graph = random_graph(seed, 24);
            let optimized = standard_pipeline(&graph);
            let inputs = [0.75, -1.25];
            assert_eq!(
                bits(&graph.evaluate(&inputs)),
                bits(&optimized.evaluate(&inputs)),
                "seed {seed} 语义漂移"
            );
            assert!(optimized.ops.len() <= graph.ops.len());
        }
    }

    /// 融合分组在「中间值被观察」时必须切开：与第 4 章 FusionInspector
    /// 的同步切分观察同构。
    #[test]
    fn fusion_groups_split_when_intermediate_is_observed() {
        let mut chain = Graph::new();
        let x = chain.input(0);
        let e = chain.exp(x);
        let c = chain.constant(2.0);
        let out = chain.mul(e, c);
        chain.mark_output(out);
        assert_eq!(fusion_group_count(&chain), 1);

        let mut observed = chain.clone();
        observed.mark_output(e);
        assert_eq!(fusion_group_count(&observed), 2);
    }

    /// 多消费者的中间值同样阻断融合。
    #[test]
    fn fusion_groups_split_at_fan_out() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let e = graph.exp(x);
        let a = graph.exp(e);
        let b = graph.mul(e, e);
        let out = graph.add(a, b);
        graph.mark_output(out);

        // e 被 a 和 b 消费两次，必须物化：a 链、b 链与顶端 add 无法
        // 全部融为一组。
        assert!(fusion_group_count(&graph) >= 2);
    }

    /// 正文反例的可运行版：(x + 1e16) + (-1e16) 被 fast-math 消去后，
    /// x=1 的结果从 0 变成 1——按位等价被破坏，改写非法。
    #[test]
    fn fast_math_cancellation_changes_float_results() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let big = graph.constant(1.0e16);
        let neg_big = graph.constant(-1.0e16);
        let shifted = graph.add(x, big);
        let out = graph.add(shifted, neg_big);
        graph.mark_output(out);

        let rewritten = fast_math_cancel(&graph);

        assert_eq!(graph.evaluate(&[1.0]), vec![0.0]);
        assert_eq!(rewritten.evaluate(&[1.0]), vec![1.0]);
        assert_ne!(graph.evaluate(&[1.0]), rewritten.evaluate(&[1.0]));
    }

    /// 同一改写在小常量上碰巧无害——「样例通过」不能证明 Pass 合法，
    /// 合法性来自数值语义论证。
    #[test]
    fn fast_math_cancellation_can_look_harmless_on_small_values() {
        let mut graph = Graph::new();
        let x = graph.input(0);
        let c = graph.constant(0.5);
        let neg_c = graph.constant(-0.5);
        let shifted = graph.add(x, c);
        let out = graph.add(shifted, neg_c);
        graph.mark_output(out);

        let rewritten = fast_math_cancel(&graph);
        assert_eq!(graph.evaluate(&[1.0]), rewritten.evaluate(&[1.0]));
    }
}
