 use virtual_modular_core_nodes::std_nodes;
use virtual_modular_definition_language::{code_generation::to_rust, parse};

fn main() {
    let input_path = std::env::args().nth(1).expect("Must supply input path");
    let output_path = std::env::args().nth(2).expect("Must supply output path");

    let parsed = parse(&std::fs::read_to_string(&input_path).expect("Couldn't read inputs")).unwrap();
    let (rust, _input_count) = to_rust(&std_nodes(), &parsed);
    std::fs::write(
        &output_path,
        rust
    )
    .unwrap();
}
