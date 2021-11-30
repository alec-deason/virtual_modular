use generic_array::{
    arr,
    typenum::*,
};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

macro_rules! binary_op {
    ($ty:ident, $op:expr) => {
        #[derive(Copy, Clone)]
        pub struct $ty;
        impl Node for $ty {
            type Input = U2;
            type Output = U1;
            #[inline]
            fn process(&mut self, mut input: Ports<Self::Input>) -> Ports<Self::Output> {
                let (a, b) = input.split_at_mut(1);
                a[0].iter_mut().zip(b[0].iter_mut()).for_each(|(a, b)| { *a = $op(*a, *b); });
                arr![[f32; BLOCK_SIZE]; a[0]]
            }
        }
    }
}

binary_op!(Mul, |a, b| (a*b));
binary_op!(Div, |a, b| (a/b));
binary_op!(Add, |a, b| (a+b));
binary_op!(Sub, |a, b| (a-b));
binary_op!(Pow, |a:f32, b| (a.powf(b)));
