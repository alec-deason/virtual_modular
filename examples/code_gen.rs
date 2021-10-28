use instruments::{
    dynamic_graph::DynamicGraph,
    code_generator::to_rust
};

fn main() {
    let lines = DynamicGraph::parse_inner(
        "
lfo = Square(3, 1)
osc = Svfl(1000,0.3,0,Square(220,1))

(output,0,osc)
 ").unwrap();
    dbg!(&lines);
    let rust_code = to_rust(&lines);
    println!("{}", rust_code);
}
