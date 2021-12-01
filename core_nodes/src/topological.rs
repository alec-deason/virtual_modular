use std::{cell::RefCell, marker::PhantomData, sync::Arc};

use generic_array::{
    arr,
    sequence::{Concat, Split},
    typenum::*,
    ArrayLength,
};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone)]
pub struct Stereo;
impl Node for Stereo {
    type Input = U1;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![[f32; BLOCK_SIZE]; input[0], input[0]]
    }
}
#[derive(Clone)]
pub struct Pipe<A: Clone, B: Clone>(pub A, pub B);
impl<A: Clone, B: Clone> Node for Pipe<A, B>
where
    A: Node<Output = B::Input> + Clone,
    B: Node + Clone,
{
    type Input = A::Input;
    type Output = B::Output;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        self.1.process(self.0.process(input))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct Branch<A: Clone, B: Clone>(pub A, pub B);
impl<A, B, C, D> Node for Branch<A, B>
where
    A: Node<Input = B::Input, Output = C> + Clone,
    B: Node + Clone,
    C: ArrayLength<[f32; BLOCK_SIZE]> + std::ops::Add<B::Output, Output = D>,
    D: ArrayLength<[f32; BLOCK_SIZE]>,
{
    type Input = A::Input;
    type Output = <A::Output as std::ops::Add<B::Output>>::Output;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let a = self.0.process(input.clone());
        let b = self.1.process(input);
        a.concat(b)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct Pass<A: ArrayLength<[f32; BLOCK_SIZE]>>(A);
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Default for Pass<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Node for Pass<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        input
    }
}
#[derive(Clone)]
pub struct Sink<A: ArrayLength<[f32; BLOCK_SIZE]>>(PhantomData<A>);
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Default for Sink<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Node for Sink<A> {
    type Input = A;
    type Output = U0;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![[f32; BLOCK_SIZE]; ]
    }
}

#[derive(Clone)]
pub struct Stack<A: Clone, B: Clone, C: Clone>(A, B, PhantomData<C>);
impl<A, B, C> Stack<A, B, C>
where
    A: Node + Clone,
    B: Node + Clone,
    C: Clone,
{
    pub fn new(a: A, b: B) -> Self {
        Self(a, b, Default::default())
    }
}
impl<A, B, C, D, E> Node for Stack<A, B, D>
where
    A: Node<Input = C, Output = E> + Clone,
    B: Node + Clone,
    C: ArrayLength<[f32; BLOCK_SIZE]> + std::ops::Add<B::Input, Output = D>,
    D: ArrayLength<[f32; BLOCK_SIZE]> + std::ops::Sub<C, Output = B::Input>,
    E: ArrayLength<[f32; BLOCK_SIZE]> + std::ops::Add<B::Output, Output = D>,
{
    type Input = <A::Input as std::ops::Add<B::Input>>::Output;
    type Output = <A::Output as std::ops::Add<B::Output>>::Output;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a, b): (Ports<A::Input>, Ports<B::Input>) = Split::split(input);
        self.0.process(a).concat(self.1.process(b))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct ArcConstant<A: ArrayLength<[f32; BLOCK_SIZE]>>(pub Arc<RefCell<Ports<A>>>);
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> ArcConstant<A> {
    pub fn new(a: Ports<A>) -> (Self, Arc<RefCell<Ports<A>>>) {
        let cell = Arc::new(RefCell::new(a));
        (Self(Arc::clone(&cell)), cell)
    }
}
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Node for ArcConstant<A> {
    type Input = U0;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        self.0.borrow().clone()
    }
}

#[derive(Clone)]
pub struct Bridge<A: ArrayLength<[f32; BLOCK_SIZE]>>(pub Arc<RefCell<Ports<A>>>);
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Bridge<A> {
    pub fn new() -> (Self, ArcConstant<A>) {
        let (constant, cell) = ArcConstant::new(Default::default());
        (Self(cell), constant)
    }
}
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Node for Bridge<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        *self.0.borrow_mut() = input.clone();
        input
    }
}

#[derive(Clone)]
pub struct Identity;
impl Node for Identity {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        input
    }
}

#[derive(Clone)]
pub struct StereoIdentity;
impl Node for StereoIdentity {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        input
    }
}

#[derive(Clone)]
pub struct IndexPort<A: ArrayLength<[f32; BLOCK_SIZE]>>(usize, PhantomData<A>);
impl<A: ArrayLength<[f32; BLOCK_SIZE]>> IndexPort<A> {
    pub fn new(port: usize) -> Self {
        IndexPort(port, Default::default())
    }
}

impl<A: ArrayLength<[f32; BLOCK_SIZE]>> Node for IndexPort<A> {
    type Input = A;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![[f32; BLOCK_SIZE]; input[self.0]]
    }
}
