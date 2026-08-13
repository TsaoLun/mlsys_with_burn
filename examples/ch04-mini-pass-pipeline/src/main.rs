use ch04_mini_pass_pipeline::{
    Graph, common_subexpression_elimination, constant_fold, dead_code_elimination,
    fusion_group_count,
};

// ANCHOR: walkthrough
fn main() {
    // out = exp(x*y) + exp(x*y) + (2*3 + 4)，外加一棵死子树 y*7。
    let mut graph = Graph::new();
    let x = graph.input(0);
    let y = graph.input(1);
    let first = graph.mul(x, y);
    let second = graph.mul(x, y);
    let e1 = graph.exp(first);
    let e2 = graph.exp(second);
    let pair = graph.add(e1, e2);
    let two = graph.constant(2.0);
    let three = graph.constant(3.0);
    let six = graph.mul(two, three);
    let four = graph.constant(4.0);
    let ten = graph.add(six, four);
    let out = graph.add(pair, ten);
    let seven = graph.constant(7.0);
    let _dead = graph.mul(y, seven);
    graph.mark_output(out);

    let inputs = [0.5, 1.5];
    println!(
        "原图：{} 个节点，输出 = {:?}",
        graph.ops.len(),
        graph.evaluate(&inputs)
    );

    let folded = constant_fold(&graph);
    println!(
        "常量折叠后：{} 个节点（2*3+4 变为 const 10）",
        folded.ops.len()
    );

    let cleaned = dead_code_elimination(&folded);
    println!(
        "DCE 后：{} 个节点（死子树与折叠残留被删除）",
        cleaned.ops.len()
    );

    let merged = common_subexpression_elimination(&cleaned);
    println!("CSE 后：{} 个节点（两棵 exp(x*y) 合并）", merged.ops.len());
    println!(
        "优化后输出 = {:?}（与原图按位一致）\n",
        merged.evaluate(&inputs)
    );
    println!("优化后的 IR：\n{}", merged.dump());

    // 融合分组：同一条链在中间值被观察时切开。
    let mut chain = Graph::new();
    let input = chain.input(0);
    let exped = chain.exp(input);
    let scale = chain.constant(2.0);
    let scaled = chain.mul(exped, scale);
    chain.mark_output(scaled);
    println!("链 exp→mul 的融合组数：{}", fusion_group_count(&chain));
    chain.mark_output(exped);
    println!(
        "把中间值 exp(x) 标记为输出后：{}（必须物化，切成两组）",
        fusion_group_count(&chain)
    );
}
// ANCHOR_END: walkthrough
