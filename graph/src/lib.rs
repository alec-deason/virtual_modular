use dyn_clone::DynClone;
use generic_array::{ArrayLength, GenericArray};

pub const BLOCK_SIZE: usize = 32;

pub type Ports<N> = GenericArray<[f32; BLOCK_SIZE], N>;

pub trait Node: DynClone {
    type Input: ArrayLength<[f32; BLOCK_SIZE]>;
    type Output: ArrayLength<[f32; BLOCK_SIZE]>;
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output>;
    fn post_process(&mut self) {}
    fn set_static_parameters(&mut self, _parameters: &str) -> Result<(), String> { Ok(()) }
    fn set_sample_rate(&mut self, _rate: f32) {}
}
dyn_clone::clone_trait_object!(<A,B> Node<Input=A, Output=B>);
