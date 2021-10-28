extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, LitStr, ExprTuple, Ident, token::Comma};

use instruments::{
    dynamic_graph::DynamicGraphBuilder,
    code_generator::to_rust,
};


#[proc_macro]
pub fn make_instrument(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as LitStr);
    let builder = DynamicGraphBuilder::default();
    let (rust_source, input_count) = to_rust(&builder, &builder.parse_inner(&input.value()).unwrap());
    println!("{}", rust_source);
    format!("{{ {} }}", rust_source).parse().unwrap()
}
