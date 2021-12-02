use virtual_modular_core_nodes::std_nodes;
use virtual_modular_definition_language::{code_generation::to_rust, parse};

fn main() {
    let parsed = parse(
        r##"
osc=Sine(220)
(output, 0, osc)
(output, 1, osc)
"##,
    )
    .unwrap();

    let (rust, _input_count) = to_rust(&std_nodes(), &parsed);
    println!("{}", rust);
}
