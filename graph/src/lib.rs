use dyn_clone::DynClone;
use generic_array::{typenum::ToInt, ArrayLength, GenericArray};

pub const BLOCK_SIZE: usize = 32;

pub trait PortCount: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> {}
impl<T: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize>> PortCount for T {}
pub type Ports<N> = GenericArray<[f32; BLOCK_SIZE], N>;

pub trait Node: DynClone {
    type Input: PortCount;
    type Output: PortCount;
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output>;
    fn set_static_parameters(&mut self, _parameters: &str) -> Result<(), String> {
        Ok(())
    }
    fn set_sample_rate(&mut self, _rate: f32) {}
    fn input_len(&self) -> usize {
        Self::Input::to_int()
    }
    fn output_len(&self) -> usize {
        Self::Output::to_int()
    }
}
dyn_clone::clone_trait_object!(<A,B> Node<Input=A, Output=B>);

pub trait WrappedNode: DynClone {
    fn process(&mut self, input: &[[f32; BLOCK_SIZE]], output: &mut [[f32; BLOCK_SIZE]]);
    fn set_static_parameters(&mut self, _parameters: &str) -> Result<(), String>;
    fn set_sample_rate(&mut self, _rate: f32);
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
}
dyn_clone::clone_trait_object!(WrappedNode);

#[derive(Clone)]
pub struct NodeWrapper<N>(N);

impl<A: PortCount, B: PortCount, N: Node<Input = A, Output = B> + Clone> WrappedNode
    for NodeWrapper<N>
{
    fn process(&mut self, input: &[[f32; BLOCK_SIZE]], output: &mut [[f32; BLOCK_SIZE]]) {
        assert_eq!(input.len(), A::to_int());
        assert_eq!(output.len(), B::to_int());
        let inputs = <Ports<A>>::clone_from_slice(input);
        let outputs = self.0.process(inputs);
        output.copy_from_slice(&outputs);
    }
    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.0.set_static_parameters(parameters)
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
    fn input_len(&self) -> usize {
        self.0.input_len()
    }
    fn output_len(&self) -> usize {
        self.0.output_len()
    }
}

#[derive(Clone)]
pub struct NodeTemplate {
    pub node: Box<dyn WrappedNode>,
    pub code: String,
}

impl<A: PortCount, B: PortCount, N: Node<Input = A, Output = B> + Clone + 'static> From<N>
    for Box<dyn WrappedNode>
{
    fn from(src: N) -> Box<dyn WrappedNode> {
        Box::new(NodeWrapper(src))
    }
}
