use dyn_clone::DynClone;
use packed_simd_2::{f32x8, u32x8};
use rand::prelude::*;
use portmidi as pm;
use std::{
    cell::RefCell,
    f32::consts::{PI,TAU},
    marker::PhantomData,
    convert::TryFrom,
    sync::{Arc, Mutex},
};
use generic_array::{ArrayLength, GenericArray, typenum::*, arr, sequence::{Concat, Split}};

pub type Ports<N> = GenericArray<f32x8, N>;

pub trait Node: DynClone {
    type Input: ArrayLength<f32x8>;
    type Output: ArrayLength<f32x8>;
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output>;
    fn post_process(&mut self) {}
    fn set_sample_rate(&mut self, rate: f32) {}
}
dyn_clone::clone_trait_object!(<A,B> Node<Input=A, Output=B>);


#[derive(Copy, Clone)]
pub struct Mul;
impl Node for Mul {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[0] * input[1]]
    }
}
#[derive(Copy, Clone)]
pub struct Div;
impl Node for Div {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[0] / input[1]]
    }
}
#[derive(Copy, Clone)]
pub struct Add;
impl Node for Add {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[0] + input[1]]
    }
}
#[derive(Copy, Clone)]
pub struct Sub;
impl Node for Sub {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[0] - input[1]]
    }
}

#[derive(Clone)]
pub struct Stereo;
impl Node for Stereo {
    type Input = U1;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[0], input[0]]
    }
}

#[derive(Clone)]
pub struct Constant(pub f32);
impl Node for Constant {
    type Input = U0;
    type Output = U1;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; f32x8::splat(self.0)]
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
    C: ArrayLength<f32x8> + std::ops::Add<B::Output, Output= D>,
    D: ArrayLength<f32x8>
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
pub struct Pass<A: ArrayLength<f32x8>>(A);
impl<A: ArrayLength<f32x8>> Default for Pass<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ArrayLength<f32x8>> Node for Pass<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        input
    }
}
#[derive(Clone)]
pub struct Sink<A: ArrayLength<f32x8>>(PhantomData<A>);
impl<A: ArrayLength<f32x8>> Default for Sink<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ArrayLength<f32x8>> Node for Sink<A> {
    type Input = A;
    type Output = U0;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; ]
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
    A: Node<Input= C, Output= E> + Clone,
    B: Node + Clone,
    C: ArrayLength<f32x8> + std::ops::Add<B::Input, Output= D>,
    D: ArrayLength<f32x8> + std::ops::Sub<C, Output=B::Input>,
    E: ArrayLength<f32x8> + std::ops::Add<B::Output, Output= D>,
{
    type Input = <A::Input as std::ops::Add<B::Input>>::Output;
    type Output = <A::Output as std::ops::Add<B::Output>>::Output;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a, b):(Ports<A::Input>, Ports<B::Input>) = Split::split(input);
        self.0.process(a).concat(self.1.process(b))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}

/*
#[derive(Clone)]
pub struct Curry<A, B>(A, B);
impl<A, B> Node for Curry<A, B>
where
    A: Node<Output = U1> + Clone,
    B: Node + Clone,
{
    type Input = <A::Input as std::ops::Add<generic_array::typenum::operator_aliases::Sub1<B::Input>>>::Output;
    type Output = B::Output;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (a, b) = D::split(input);
        let a = self.0.process(a);
        let input = C::combine(a, b);
        self.1.process(input)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}
pub fn curry<A: Node, B: Node>(
    a: A,
    b: B,
) -> Curry<
    A,
    B,
    (A::Output, Value<(<B::Input as ValueT>::Cdr,)>),
    (A::Input, Value<(<B::Input as ValueT>::Cdr,)>),
    Value<(<B::Input as ValueT>::Cdr,)>,
> {
    Curry(
        a,
        b,
        Default::default(),
        Default::default(),
        Default::default(),
    )
}
pub fn curry2<A: Node, B: Node>(
    a: A,
    b: B,
) -> Curry<
    A,
    B,
    (A::Output, Value<<B::Input as ValueT>::Cdr>),
    (A::Input, Value<<B::Input as ValueT>::Cdr>),
    Value<<B::Input as ValueT>::Cdr>,
> {
    Curry(
        a,
        b,
        Default::default(),
        Default::default(),
        Default::default(),
    )
}
pub fn curry3<A: Node, B: Node>(
    a: A,
    b: B,
) -> Curry<
    A,
    B,
    (A::Output, Value<<B::Input as ValueT>::Cdr>),
    (A::Input, Value<<B::Input as ValueT>::Cdr>),
    Value<<B::Input as ValueT>::Cdr>,
> {
    Curry(
        a,
        b,
        Default::default(),
        Default::default(),
        Default::default(),
    )
}

#[derive(Clone)]
pub struct Concat<A, B, C, D>(A, PhantomData<B>, PhantomData<C>, PhantomData<D>);
impl<A, B, C, D> Concat<A, B, C, D> {
    pub fn new(a: A) -> Self {
        Self(
            a,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<A, B, C, D> Node for Concat<A, B, C, D>
where
    A: Node + Clone,
    B: ValueT,
    C: Combine<A::Input, B> + Clone,
    D: Combine<A::Output, B> + Clone,
{
    type Input = C::Output;
    type Output = D::Output;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (a, b) = C::split(input);
        D::combine(self.0.process(a), b)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}


*/
#[derive(Clone,Default)]
pub struct PulseOnLoad(bool);
impl Node for PulseOnLoad {
    type Input = U0;
    type Output = U1;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        if self.0 {
            arr![f32x8; f32x8::splat(0.0)]
        } else {
            self.0 = true;
            arr![f32x8; f32x8::from_slice_unaligned(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        }
    }
}
/*
#[derive(Clone)]
pub struct RawConstant<A:ValueT>(pub A::Inner);
impl<A: ValueT> Node for RawConstant<A> {
    type Input = NoValue;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        A::from_inner(self.0.clone())
    }
}

#[derive(Clone)]
pub struct Splat;
impl Node for Splat {
    type Input = Value<(f32,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((f32x8::splat(*input.car()),))
    }
}
#[derive(Clone)]
pub struct Max;
impl Node for Max {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.car().max_element(),))
    }
}

#[derive(Clone)]
pub struct Mean;
impl Node for Mean {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.car().sum() / f32x8::lanes() as f32,))
    }
}
#[derive(Clone)]
pub struct One;
impl Node for One {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.car().extract(0),))
    }
}

*/
#[derive(Clone)]
pub struct RingBufConstant<A: ArrayLength<f32x8>>(Arc<ringbuf::Consumer<(usize, f32)>>, Ports<A>);
impl<A: ArrayLength<f32x8>> RingBufConstant<A> {
    pub fn new() -> (Self, ringbuf::Producer<(usize, f32)>) {
        let buf = ringbuf::RingBuffer::new(100);
        let (p,c) = buf.split();
        (RingBufConstant(Arc::new(c), Default::default()), p)
    }
}
impl<A: ArrayLength<f32x8>> Node for RingBufConstant<A> {
    type Input = U0;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        while let Some((i,v)) = Arc::get_mut(&mut self.0).and_then(|b| b.pop()) {
            self.1[i] = f32x8::splat(v);
        }
        self.1.clone()
    }
}
#[derive(Clone)]
pub struct ArcConstant<A: ArrayLength<f32x8>>(pub Arc<RefCell<Ports<A>>>);
impl<A: ArrayLength<f32x8>>  ArcConstant<A> {
    pub fn new(a: Ports<A>) -> (Self, Arc<RefCell<Ports<A>>>) {
        let cell = Arc::new(RefCell::new(a));
        (Self(Arc::clone(&cell)), cell)
    }
}
impl<A: ArrayLength<f32x8>> Node for ArcConstant<A> {
    type Input = U0;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        self.0.borrow().clone()
    }
}
#[derive(Clone)]
pub struct MutexConstant<A: ArrayLength<f32x8>>(pub Arc<Mutex<Ports<A>>>);
impl<A: ArrayLength<f32x8>>  MutexConstant<A> {
    pub fn new(a: Ports<A>) -> (Self, Arc<Mutex<Ports<A>>>) {
        let mutex = Arc::new(Mutex::new(a));
        (Self(Arc::clone(&mutex)), mutex)
    }
}
impl<A: ArrayLength<f32x8>> Node for MutexConstant<A> {
    type Input = U0;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        self.0.lock().unwrap().clone()
    }
}
#[derive(Clone)]
pub struct OneArcConstant(pub Arc<RefCell<f32x8>>);
impl OneArcConstant {
    pub fn new(a: f32x8) -> (Self, Arc<RefCell<f32x8>>) {
        let cell = Arc::new(RefCell::new(a));
        (Self(Arc::clone(&cell)), cell)
    }
}
impl Node for OneArcConstant {
    type Input = U0;
    type Output = U1;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; self.0.borrow().clone()]
    }
}
/*
#[derive(Clone)]
pub struct RawArcConstant<A:ValueT>(pub Arc<RefCell<A::Inner>>);
impl<A:ValueT> RawArcConstant<A> {
    pub fn new(a: A::Inner) -> (Self, Arc<RefCell<A::Inner>>) {
        let cell = Arc::new(RefCell::new(a));
        (Self(Arc::clone(&cell)), cell)
    }
}
impl<A: ValueT> Node for RawArcConstant<A> {
    type Input = NoValue;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        A::from_inner(self.0.borrow().clone())
    }
}
*/

#[derive(Clone)]
pub struct Bridge<A: ArrayLength<f32x8>>(pub Arc<RefCell<Ports<A>>>);
impl<A: ArrayLength<f32x8>> Bridge<A> {
    pub fn new() -> (Self, ArcConstant<A>) {
        let (constant, cell) = ArcConstant::new(Default::default());
        (Self(cell), constant)
    }
}
impl<A: ArrayLength<f32x8>> Node for Bridge<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        *self.0.borrow_mut() = input.clone();
        input
    }
}
#[derive(Clone)]
pub struct MutexBridge<A: ArrayLength<f32x8>>(pub Arc<Mutex<Ports<A>>>);
impl<A: ArrayLength<f32x8>> MutexBridge<A> {
    pub fn new() -> (Self, MutexConstant<A>) {
        let (constant, cell) = MutexConstant::new(Default::default());
        (Self(cell), constant)
    }
}
impl<A: ArrayLength<f32x8>> Node for MutexBridge<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        *self.0.lock().unwrap() = input.clone();
        input
    }
}
#[derive(Clone)]
pub struct OneBridge(pub Arc<RefCell<f32x8>>);
impl OneBridge {
    pub fn new() -> (Self, OneArcConstant) {
        let (constant, cell) = OneArcConstant::new(Default::default());
        (Self(cell), constant)
    }
}
impl Node for OneBridge {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        *self.0.borrow_mut() = input[0].clone();
        input
    }
}
/*

#[derive(Clone)]
pub struct RawBridge<A: ValueT>(pub Arc<RefCell<A::Inner>>);
impl<A: ValueT> RawBridge<A> {
    pub fn new(a: A::Inner) -> (Self, RawArcConstant<A>) {
        let (constant, cell) = RawArcConstant::new(a);
        (Self(cell), constant)
    }
}
impl<A: ValueT> Node for RawBridge<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        *self.0.borrow_mut() = input.inner().clone();
        input
    }
}

#[derive(Clone)]
pub struct FnConstant<F>(pub F);
impl<F: Fn() -> A + Clone, A: Clone> Node for FnConstant<F> {
    type Input = NoValue;
    type Output = Value<(A,)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        Value(((self.0)(),))
    }
}

#[derive(Default, Clone)]
pub struct ThreshEnvelope<F> {
    func: F,
    time: f32,
    off_time: Option<f32>,
    per_sample: f32,
}
impl<F> ThreshEnvelope<F> {
    pub fn new(func: F) -> Self {
        let r = Self {
            func,
            time: 0.0,
            off_time: None,
            per_sample: 0.0,
        };
        r
    }
}

impl<F> Node for ThreshEnvelope<F>
where
    F: FnMut(f32, Option<f32>) -> f32 + Clone,
{
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = input.car();
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            if gate > 0.5 {
                if self.off_time.take().is_some() {
                    self.time = 0.0;
                }
            } else {
                if self.off_time.is_none() {
                    self.off_time = Some(self.time);
                }
            }
            self.time += self.per_sample;
        }
        let m = f32x8::splat((self.func)(self.time, self.off_time));
        Value((m,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
*/

#[derive(Copy,Clone,Default)]
pub struct InlineADEnvelope {
    time: f32,
    triggered: bool,
    running_cycle: bool,
    current: f32,
    per_sample: f32
}
impl Node for InlineADEnvelope {
    type Input = U4;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (attack, decay, gate, do_loop) = (input[0], input[1], input[2], input[3]);
        let mut r = f32x8::splat(0.0);
        let mut eoc = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let attack = attack.extract(i);
            let decay = decay.extract(i);
            let gate = gate.extract(i);
            let do_loop = do_loop.extract(i);
            if gate > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.time = 0.0;
                    self.running_cycle = true;
                }
            } else {
                if self.triggered {
                    self.triggered = false
                }
            }
            self.time += self.per_sample;
            let v = if self.time < attack {
                self.time/attack
            } else if self.time < attack+decay {
                let t = (self.time - attack) / decay;
                1.0-t
            } else {
                if self.running_cycle {
                    if do_loop > 0.5 {
                        self.time = 0.0;
                    } else {
                        self.running_cycle = false;
                    }
                    eoc = eoc.replace(i, 1.0);
                }
                0.0
            };
            self.current = self.current * 0.001 + v * 0.999;
            r = r.replace(i, self.current);
        }

        arr![f32x8; r, eoc]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}


#[derive(Copy,Clone,Default)]
pub struct InlineADSREnvelope {
    time: f32,
    triggered: bool,
    current: f32,
    per_sample: f32
}
impl Node for InlineADSREnvelope {
    type Input = U5;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (attack, decay, sustain, release, gate) = (input[0], input[1], input[2], input[3], input[4]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let attack = attack.extract(i);
            let decay = decay.extract(i);
            let release = release.extract(i);
            let sustain = sustain.extract(i);
            let gate = gate.extract(i);
            if gate > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.time = 0.0;
                }
            } else {
                if self.triggered {
                    self.triggered = false;
                    self.time = 0.0;
                }
            }
            self.time += self.per_sample;
            if self.triggered {
                let target = if self.time < attack {
                    self.time/attack
                } else if self.time < attack+decay {
                    let t = (self.time - attack) / decay;
                    (1.0-t) + t*sustain
                } else {
                    sustain
                };
                self.current = self.current * 0.001 + target * 0.999;
                r = r.replace(i, self.current);
            } else {
                if self.time < release {
                    r=r.replace(i, (1.0 - self.time/release)*sustain);
                }
            }
        }

        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}


/*
#[derive(Clone)]
pub struct Unison<A: Clone> {
    voices: Vec<(f32, f32, A)>,
}
impl<A: Clone> Unison<A> {
    pub fn new(voice: A, spread: f32, detune: f32, count: u32) -> Self {
        let mut voices = Vec::with_capacity(count as usize);
        let mut spread = -spread / 2.0;
        let spread_step = spread / count as f32;
        let mut detune = -detune / 2.0;
        let detune_step = detune / count as f32;
        for _ in 0..count {
            let d = if detune > 0.0 {
                detune + 1.0
            } else {
                1.0 / (detune.abs() + 1.0)
            };
            voices.push((spread, 1.05946f32.powf(d), voice.clone()));
            spread += spread_step;
            detune += detune_step;
        }
        Self { voices }
    }
}

impl<A, C> Node for Unison<A>
where
    A: Node<Input = C, Output = Value<(f32x8,)>> + Clone,
    C: ValueT<Car = f32x8> + Copy,
{
    type Input = A::Input;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut l_r = f32x8::splat(0.0);
        let mut r_r = f32x8::splat(0.0);
        for (spread, detune, v) in &mut self.voices {
            let l_spread = f32x8::splat(*spread);
            let r_spread = f32x8::splat(1.0 - *spread);
            let detune = f32x8::splat(*detune);
            let input = input.map_car(|v| v * detune);
            l_r += *v.process(input).car() * l_spread;
            r_r += *v.process(input).car() * r_spread;
        }
        let a = f32x8::splat(1.0 / self.voices.len() as f32);
        Value((l_r * a, r_r * a))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for (_, _, v) in &mut self.voices {
            v.set_sample_rate(rate);
        }
    }
}

*/
#[derive(Clone)]
pub struct WaveTable {
    sample_rate: f32,
    table: Vec<f32>,
    len: f32,
    pub idx: f32,
}
impl WaveTable {
    pub fn square() -> Self {
        let sz = 1024; // the size of the table
        let mut table = vec![0.0; sz];
        let scale = 1.0;
        let omega = 1.0 / sz as f32;
        for i in 0..sz {
            let mut amp = scale;
            let mut x = 0.0; // the sample
            let mut h = 1.0; // harmonic number (starts from 1)
            let mut dd; // fix high frequency "ring"
            let pd = i as f32 / sz as f32; // calc phase
            let mut hpd = pd; // phase of the harmonic
            loop {
                if (omega * h) < 0.5
                // harmonic frequency is in range?
                {
                    dd = ((omega * h * std::f32::consts::PI).sin() * 0.5 * std::f32::consts::PI)
                        .cos();
                    x = x + (amp * dd * (hpd * 2.0 * std::f32::consts::PI).sin());
                    h = h + 2.0;
                    hpd = pd * h;
                    amp = scale / h;
                } else {
                    break;
                }
            }
            table[i] = x;
        }
        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }

    pub fn sine() -> Self {
        let sz = 4096; // the size of the table
        let mut table = vec![0.0; sz];
        for i in 0..sz {
            let x = ((i as f32 / sz as f32) * std::f32::consts::PI * 2.0).sin();
            table[i] = x;
        }
        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }

    pub fn positive_sine() -> Self {
        let sz = 1024; // the size of the table
        let mut table = vec![0.0; sz];
        for i in 0..sz {
            let x = ((i as f32 / sz as f32) * std::f32::consts::PI * 2.0).sin();
            let x = (x + 1.0) / 2.0;
            table[i] = x;
        }
        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }

    pub fn saw() -> Self {
        let sz = 1024; // the size of the table
        let mut table = vec![0.0; sz];
        let scale = 1.0;
        let omega = 1.0 / sz as f32;

        for i in 0..sz {
            let mut amp = scale;
            let mut x = 0.0; // the sample
            let mut h = 1.0; // harmonic number (starts from 1)
            let mut dd; // fix high frequency "ring"
            let pd = i as f32 / sz as f32; // calc phase
            let mut hpd = pd; // phase of the harmonic
            loop {
                if (omega * h) < 0.5
                // harmonic frequency is in range?
                {
                    dd = ((omega * h * std::f32::consts::PI).sin() * 0.5 * std::f32::consts::PI)
                        .cos();
                    x = x + (amp * dd * (hpd * 2.0 * std::f32::consts::PI).sin());
                    h = h + 1.0;
                    hpd = pd * h;
                    amp = scale / h;
                } else {
                    break;
                }
            }
            table[i] = x;
        }

        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }

    pub fn noise() -> Self {
        let mut rng = StdRng::seed_from_u64(2);
        let table: Vec<f32> = (0..1024 * 1000).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }
}

impl Node for WaveTable {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let reset = input[1];
        let input = input[0];
        let d = 1.0 / (self.sample_rate / self.len);

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            if reset.extract(i) > 0.5 {
                self.idx = 0.0;
            }
            if self.idx >= self.len {
                self.idx -= self.len;
            }
            r = unsafe { r.replace_unchecked(i, self.table[self.idx as usize % self.table.len()]) };
            self.idx += input.extract(i) * d;
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }
}

#[derive(Clone)]
pub struct Sine {
    phase: f64,
    per_sample: f64
}

impl Default for Sine {
    fn default() -> Self {
        Self {
            phase: thread_rng().gen(),
            per_sample: 1.0/44100.0,
        }
    }
}

impl Node for Sine {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let input = input[0];

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let v = (TAU as f64 * self.phase).sin();
            self.phase += self.per_sample * input.extract(i) as f64;
            if self.phase > 1.0 {
                self.phase -= 1.0;
            }
            r = r.replace(i, v as f32);
        }

        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
    }
}
/*


#[derive(Clone, Default)]
pub struct LowPass {
    s: f32,
    m: f32,
    per_sample: f32,
}
impl Node for LowPass {
    type Input = Value<((f32x8, f32x8), f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        let Value(((cutoff, resonance), signal)) = input;
        for i in 0..f32x8::lanes() {
            let cutoff = cutoff.extract(i);
            let resonance = resonance.extract(i);
            let signal = signal.extract(i);
            let alpha = (std::f32::consts::TAU * self.per_sample * cutoff)
                / (std::f32::consts::TAU * self.per_sample * cutoff + 1.0);
            self.s = (alpha * signal) + ((1.0 - alpha) * self.s);
            self.m = (self.s - signal) * resonance;
            self.s += self.m;
            r = unsafe { r.replace_unchecked(i, self.s) };
        }
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Clone)]
enum BiquadConfig {
    LowPass,
    HighPass,
    PeakingEq(Arc<RefCell<(f32,)>>),
    Notch,
    BandPass,
}
impl BiquadConfig {
    /// http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    fn coefficients(&self, freq: f32, q: f32, sample_rate: f32) -> BiquadCoefficients {
        match self {
            BiquadConfig::LowPass => {
                let omega = TAU * (freq / sample_rate);
                let alpha = omega.sin() / (2.0 * q);
                BiquadCoefficients {
                    a0: (1.0 + alpha) as f64,
                    a1: (-2.0 * omega.cos()) as f64,
                    a2: (1.0 - alpha) as f64,
                    b0: ((1.0 - omega.cos()) / 2.0) as f64,
                    b1: (1.0 - omega.cos()) as f64,
                    b2: ((1.0 - omega.cos()) / 2.0) as f64,
                }
            }
            BiquadConfig::HighPass => {
                let omega = TAU * (freq / sample_rate);
                let alpha = omega.sin() / (2.0 * q);
                BiquadCoefficients {
                    a0: (1.0 + alpha) as f64,
                    a1: (-2.0 * omega.cos()) as f64,
                    a2: (1.0 - alpha) as f64,
                    b0: ((1.0 + omega.cos()) / 2.0) as f64,
                    b1: (-(1.0 + omega.cos())) as f64,
                    b2: ((1.0 + omega.cos()) / 2.0) as f64,
                }
            }
            BiquadConfig::Notch => {
                let omega = TAU * (freq / sample_rate);
                let alpha = omega.sin() / (2.0 * q);
                BiquadCoefficients {
                    a0: (1.0 + alpha) as f64,
                    a1: (-2.0 * omega.cos()) as f64,
                    a2: (1.0 - alpha) as f64,
                    b0: 1.0,
                    b1: (-2.0 * omega.cos()) as f64,
                    b2: 1.0,
                }
            }
            BiquadConfig::BandPass => {
                let omega = TAU * (freq / sample_rate);
                let alpha = omega.sin() / (2.0 * q);
                BiquadCoefficients {
                    a0: (1.0 + alpha) as f64,
                    a1: (-2.0 * omega.cos()) as f64,
                    a2: (1.0 - alpha) as f64,
                    b0: (q * alpha) as f64,
                    b1: 0.0,
                    b2: (-q * alpha) as f64,
                }
            }
            BiquadConfig::PeakingEq(db_gain) => {
                let omega = TAU * (freq / sample_rate);
                let alpha = omega.sin() / (2.0 * q);
                let a = (10.0f32 * (db_gain.borrow().0 / 20.0)).sqrt();
                BiquadCoefficients {
                    a0: (1.0 + alpha / a) as f64,
                    a1: (-2.0 * omega.cos()) as f64,
                    a2: (1.0 - alpha / a) as f64,
                    b0: (1.0 + alpha * a) as f64,
                    b1: (-2.0 * omega.cos()) as f64,
                    b2: (1.0 - alpha * a) as f64,
                }
            }
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct BiquadCoefficients {
    b0: f64,
    b1: f64,
    b2: f64,
    a0: f64,
    a1: f64,
    a2: f64,
}

#[derive(Clone)]
pub struct Biquad {
    coefficients: BiquadCoefficients,
    config: BiquadConfig,

    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,

    sample_rate: f32,
}
impl Biquad {
    pub fn lowpass() -> Self {
        Self {
            config: BiquadConfig::LowPass,
            coefficients: Default::default(),

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,

            sample_rate: 44100.0,
        }
    }
    pub fn highpass() -> Self {
        Self {
            config: BiquadConfig::HighPass,
            coefficients: Default::default(),

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,

            sample_rate: 44100.0,
        }
    }
    pub fn notch() -> Self {
        Self {
            config: BiquadConfig::HighPass,
            coefficients: Default::default(),

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,

            sample_rate: 44100.0,
        }
    }
    pub fn bandpass() -> Self {
        Self {
            config: BiquadConfig::BandPass,
            coefficients: Default::default(),

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,

            sample_rate: 44100.0,
        }
    }

    pub fn peaking_eq() -> PeakingEq {
        let db_gain =  Arc::new(RefCell::new((0.0,)));
        PeakingEq(
            Self {
                config: BiquadConfig::PeakingEq(db_gain.clone()),
                coefficients: Default::default(),

                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,

                sample_rate: 44100.0,
            },
            db_gain
        )
    }
}
#[derive(Clone)]
pub struct PeakingEq(Biquad, Arc<RefCell<(f32,)>>);


impl Node for Biquad {
    type Input = Value<((f32x8, f32x8), f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((freq, q), x)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let x = x.extract(i) as f64;
            let freq = freq.extract(i);
            let q = q.extract(i);
            self.coefficients = self.config.coefficients(freq, q, self.sample_rate);
            let y = (self.coefficients.b0 / self.coefficients.a0) * x
                + (self.coefficients.b1 / self.coefficients.a0) * self.x1
                + (self.coefficients.b2 / self.coefficients.a0) * self.x2
                - (self.coefficients.a1 / self.coefficients.a0) * self.y1
                - (self.coefficients.a2 / self.coefficients.a0) * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            r = r.replace(i, y as f32);
        }
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }
}

impl Node for PeakingEq {
    type Input = Value<((f32x8, f32x8), (f32x8,f32x8))>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((db_gain, freq), (q, signal))) = input;
        *self.1.borrow_mut() = (db_gain.max_element(),);
        self.0.process(Value(((freq,q), signal)))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[macro_export]
macro_rules! file_constants {
    ($($name:ident : $t:ty = $initial_value:expr),* $(,)*) => {
        {
            use serde::Deserialize;
            use std::{
                rc::Arc,
                cell::RefCell,
            };
            #[derive(Deserialize)]
            struct ParameterContainer {
                $(
                    $name: $t,
                )*
            }
            struct RefCellContainer {
                $(
                    $name: Arc<RefCell<$t>>,
                )*
            }
            $(
                let $name = ArcConstant::new($initial_value);
            )*
            let cell_container = RefCellContainer {
                $(
                    $name: $name.1,
                )*
            };
            let poller = move |path: &str| {
                //let f = std::fs::File::open(&path).expect("Failed opening file");
                //match ron::de::from_reader::<_, ParameterContainer>(f) {
                //    Ok(container) => {
                //        $(
                //            cell_container.$name.replace(container.$name);
                //        )*
                //    },
                //    Err(e) => { println!("{:?}", e); }
                //}
            };
            (poller, $($name.0,)*)
        }
    }
}

#[derive(Clone)]
pub struct Choice(pub Vec<f32>);
impl Node for Choice {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((mut r,)) = input;
        for i in 0..f32x8::lanes() {
            let j = r.extract(i);
            let j = (j.max(0.0).min(1.0) * self.0.len() as f32) as usize;
            r = r.replace(i, self.0[j.min(self.0.len() - 1)]);
        }
        Value((r,))
    }
}

#[derive(Clone)]
pub struct ArcChoice(pub Arc<RefCell<Vec<f32>>>);
impl Node for ArcChoice {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let choices = self.0.borrow();
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let j = ((input.0).0.extract(i).max(0.0).min(1.0) * choices.len() as f32) as usize;
            r = r.replace(i, choices[j.min(choices.len() - 1)]);
        }
        Value((r,))
    }
}
*/

#[derive(Copy, Clone, Default)]
pub struct SampleAndHold(f32, bool, bool);
impl Node for SampleAndHold {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (gate, signal) = (input[0], input[1]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            let signal = signal.extract(i);
            if !self.1 && gate > 0.5 || !self.2 {
                self.0 = signal;
                self.1 = true;
                self.2 = true;
            } else if gate < 0.5 {
                self.1 = false;
            }
            r = unsafe { r.replace_unchecked(i, self.0) };
        }
        arr![f32x8; r]
    }
}


#[derive(Copy, Clone)]
pub struct Impulse(f32, bool);
impl Default for Impulse {
    fn default() -> Self {
        Self::new(0.5)
    }
}
impl Impulse {
    pub fn new(threshold: f32) -> Self {
        Self(threshold, false)
    }
}
impl Node for Impulse {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let input = input[0];
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = input.extract(i);
            let mut switched = false;
            if !self.1 && gate > self.0 {
                self.1 = true;
                switched = true;
            } else if gate < self.0 {
                self.1 = false;
            }
            if switched {
                r = unsafe { r.replace_unchecked(i, 1.0) };
            }
        }
        arr![f32x8; r]
    }
}
/*
#[derive(Copy, Clone)]
pub struct SImpulse(f32, bool);
impl Default for SImpulse {
    fn default() -> Self {
        Self::new(0.5)
    }
}
impl SImpulse {
    pub fn new(threshold: f32) -> Self {
        Self(threshold, false)
    }
}
impl Node for SImpulse {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = (input.0).0;
        let mut switched = false;
        if !self.1 && gate > self.0 {
            self.1 = true;
            switched = true;
        } else if gate < self.0 {
            self.1 = false;
        }
        if switched {
            Value((1.0,))
        } else {
            Value((0.0,))
        }
    }
}

#[derive(Copy, Clone)]
pub struct ToGate {
    threshold: f32,
    triggered: bool,
    v: f32,
    abs: bool,
}
impl Default for ToGate {
    fn default() -> Self {
        Self::new(0.5)
    }
}
impl ToGate {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            triggered: false,
            v: 0.0,
            abs: false,
        }
    }
    pub fn abs(threshold: f32) -> Self {
        Self {
            threshold,
            triggered: false,
            v: 0.0,
            abs: true,
        }
    }
}
impl Node for ToGate {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let mut gate = (input.0).0.extract(i);
            if self.abs {
                gate = gate.abs();
            }
            self.v = self.v.max(gate);
            self.v *= 0.999;
            if !self.triggered && self.v > self.threshold {
                self.v = 1.0;
                self.triggered = true;
            } else if self.v < self.threshold {
                self.triggered = false;
            }
            if self.triggered {
                r = unsafe { r.replace_unchecked(i, 1.0) };
            }
        }
        Value((r,))
    }
}

#[derive(Clone)]
pub struct Accents {
    amount: f32,
    triggered: bool,
    count: u32,
    v: f32,
}
impl Default for Accents {
    fn default() -> Self {
        Self::new(2.0)
    }
}
impl Accents {
    pub fn new(amount: f32) -> Self {
        Self {
            amount,
            triggered: false,
            count: 0,
            v: 0.0,
        }
    }
}
impl Node for Accents {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(1.0);
        for i in 0..f32x8::lanes() {
            let mut gate = (input.0).0.extract(i);
            self.v = self.v.max(gate);
            self.v *= 0.999;
            if !self.triggered && self.v > 0.5 {
                self.v = 1.0;
                self.triggered = true;
                self.count += 1;
            } else if self.v < 0.5 {
                self.triggered = false;
            }
            if self.triggered {
                if self.count == 0 || self.count == 2 {
                    r = unsafe { r.replace_unchecked(i, self.amount) };
                } else if self.count == 4 {
                    self.count = 0;
                }
            }
        }
        Value((r,))
    }
}

#[derive(Clone)]
pub struct GateSequencer(Arc<RefCell<Vec<bool>>>, usize, bool);
impl GateSequencer {
    pub fn new(values: Arc<RefCell<Vec<bool>>>) -> Self {
        Self(values, 0, false)
    }
}
impl Node for GateSequencer {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        let seq = self.0.borrow();
        for i in 0..f32x8::lanes() {
            let gate = (input.0).0.extract(i);
            let mut switched = false;
            if !self.2 && gate > 0.5 {
                self.1 = (self.1 + 1) % seq.len();
                self.2 = true;
                switched = true;
            } else if gate < 0.5 {
                self.2 = false;
            }
            if !switched && seq[self.1] {
                r = unsafe { r.replace_unchecked(i, 1.0) };
            }
        }
        Value((r,))
    }
}

#[derive(Clone)]
pub struct Sequencer<A>(Vec<A>, usize, bool);
impl<A> Sequencer<A> {
    pub fn new(values: Vec<A>) -> Self {
        Self(values, 0, false)
    }
}
impl<A: Clone> Node for Sequencer<A> {
    type Input = Value<(f32x8,)>;
    type Output = Value<(A,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = (input.0).0.max_element();
        if !self.2 && gate > 0.5 {
            self.1 = (self.1 + 1) % self.0.len();
            self.2 = true;
        } else if gate < 0.5 {
            self.2 = false;
        }
        Value((self.0[self.1 % self.0.len()].clone(),))
    }
}

#[derive(Clone)]
pub struct DelayLine<T> {
    idx: usize,
    buffer: Vec<T>,
    filter: Option<Box<dyn Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>>>>,
}

impl DelayLine<f32x8> {
    pub fn new(len: usize) -> Self {
        Self {
            idx: 0,
            buffer: vec![f32x8::splat(0.0); len],
            filter: None,
        }
    }

    pub fn new_nested(
        len: usize,
        filter: impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>> + 'static,
    ) -> Self {
        Self {
            idx: 0,
            buffer: vec![f32x8::splat(0.0); len],
            filter: Some(Box::new(filter)),
        }
    }

    fn set_sample_rate(&mut self, rate: f32) {
        if let Some(filter) = &mut self.filter {
            filter.set_sample_rate(rate)
        }
    }
}

impl DelayLine<f32x8> {
    fn post_swap(&mut self, v: f32x8, scale: f32x8) -> f32x8 {
        let mut inner = self.buffer[self.idx] * scale + v;

        if let Some(filter) = &mut self.filter {
            inner = (filter.process(Value((v,))).0).0;
        }

        self.buffer[self.idx] = inner;
        self.idx = (self.idx + 1) % self.buffer.len();
        inner
    }

    fn pre_swap(&mut self, v: f32x8) -> f32x8 {
        let mut new_v = self.buffer[self.idx];

        if let Some(filter) = &mut self.filter {
            new_v = (filter.process(Value((v,))).0).0;
        }
        self.buffer[self.idx] = v;
        self.idx = (self.idx + 1) % self.buffer.len();
        new_v
    }
}
*/

#[derive(Copy, Clone, Default)]
pub struct AllPass(f32, f32);

impl Node for AllPass {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (scale, signal) = (input[0], input[1]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let scale = scale.extract(i);
            let signal = signal.extract(i);
            let v = scale * signal + self.0 - scale * self.1;
            self.0 = signal;
            self.1 = v;
            r = unsafe { r.replace_unchecked(i, v) };
        }
        arr![f32x8; r]
    }
}
/*


#[derive(Copy, Clone)]
pub struct Flip<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Flip<A, B> {
    fn default() -> Self {
        Flip(Default::default(), Default::default())
    }
}

impl<A: Clone, B: Clone> Node for Flip<A, B> {
    type Input = Value<(A, B)>;
    type Output = Value<(B, A)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((a, b)) = input;
        Value((b, a))
    }
}

#[derive(Clone)]
pub struct Comb<T> {
    delay: DelayLine<T>,
}

impl<T> Comb<T> {
    pub fn new(delay: DelayLine<T>) -> Self {
        Self { delay }
    }
}

impl Node for Comb<f32x8> {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let scale = *input.car();
        let src = *input.cdr();
        let v = self.delay.post_swap(src, scale);
        Value((v,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.delay.set_sample_rate(rate)
    }
}

#[derive(Clone)]
pub struct Mix;

impl Node for Mix {
    type Input = Value<((f32x8, f32x8), (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((la, ra), (lb, rb))) = input;
        Value((la + lb, ra + rb))
    }
}
*/

#[derive(Copy, Clone, Default)]
pub struct PulseDivider(u64, bool);

impl Node for PulseDivider {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (division, gate) = (input[0], input[1]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            let division = division.extract(i).round() as u64;
            if gate > 0.5 {
                if !self.1 {
                    self.0 += 1;
                    self.1 = true;
                }
            } else if self.1 {
                self.1 = false;
            }
            if self.1 && division > 0 && self.0 % division == 0 {
                r = r.replace(i, gate);
            }
        }
        arr![f32x8; r]
    }
}

/*
#[derive(Clone,Default)]
pub struct Compressor {
    time: f32,
    decaying: bool,
    per_sample: f32,
}
impl Node for Compressor {
    type Input = Value<(f32x8, (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((amount, (left, right))) = input;
        let mut l = f32x8::splat(0.0);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let left = left.extract(i);
            let right = right.extract(i);
            let amount = amount.extract(i);
            if left.abs() > amount || right.abs() > amount {
                if self.decaying {
                    self.decaying = false;
                    self.time = 0.0;
                }
                let t = (self.time / 0.1).min(1.0);
                let m = (1.0 - t) + t * 0.25;
                l = l.replace(i, left * m);
                r = r.replace(i, right * m);
                self.time += self.per_sample;
            } else {
                if !self.decaying {
                    self.decaying = true;
                    self.time = 0.0;
                }
                let t = (self.time / 0.1).min(1.0);
                let m = (1.0 - t) * 0.25 + t;
                l = l.replace(i, left * m);
                r = r.replace(i, right * m);
                self.time += self.per_sample;
            }
        }
        Value((l, r))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
*/

#[derive(Clone)]
pub struct SidechainCompressor {
    threshold: f32,
    time: f32,
    v: f32,
    decaying: bool,
    per_sample: f32,
}
impl SidechainCompressor {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            time: 0.0,
            v: 0.0,
            decaying: false,
            per_sample: 0.0,
        }
    }
}
impl Node for SidechainCompressor {
    type Input = U3;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (sidechain, left, right) = (input[0], input[1], input[2]);
        let mut l = f32x8::splat(0.0);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let left = left.extract(i);
            let right = right.extract(i);
            let sidechain = sidechain.extract(i);
            if sidechain.abs() > self.threshold {
                if self.decaying {
                    self.decaying = false;
                    self.time = 0.0;
                }
                let t = (self.time / 0.1).min(1.0);
                let m = (1.0 - t) + t * 0.25;
                self.v = self.v*0.99 + m*0.01;
                l = l.replace(i, left * self.v);
                r = r.replace(i, right * self.v);
                self.time += self.per_sample;
            } else {
                if !self.decaying {
                    self.decaying = true;
                    self.time = 0.0;
                }
                let t = (self.time / 0.1).min(1.0);
                let m = (1.0 - t) * 0.25 + t;
                self.v = self.v*0.99 + m*0.01;
                l = l.replace(i, left * self.v);
                r = r.replace(i, right * self.v);
                self.time += self.per_sample;
            }
        }
        arr![f32x8; l, r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
/*

#[derive(Clone)]
pub struct Limiter {
    threshold: f32,
}
impl Limiter {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}
impl Node for Limiter {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((left, right)) = input;
        let mut l = f32x8::splat(0.0);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let left = left.extract(i);
            let right = right.extract(i);
            let max = left.abs().max(right.abs());
            let m = if max > self.threshold {
                self.threshold / max
            } else {
                1.0
            };
            l = l.replace(i, left * m);
            r = r.replace(i, right * m);
        }
        Value((l, r))
    }
}

#[derive(Clone)]
pub struct HarmonicOscillator {
    x: f32,
    k: f32,
    u: f32,
    v: f32,
    m: f32,
    per_sample: f32,
}

impl HarmonicOscillator {
    pub fn new(mass: f32) -> Self {
        Self {
            x: 0.0,
            k: 1.0,
            u: 40.0,
            v: 0.0,
            m: mass,
            per_sample: 0.0,
        }
    }
}
impl Node for HarmonicOscillator {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        let Value((gate, frequency)) = input;
        for i in 0..f32x8::lanes() {
            let frequency = frequency.extract(i);
            let gate = gate.extract(i);
            self.k = (frequency * std::f32::consts::TAU).powi(2) / self.m;
            let f = -self.k * self.x;
            let f = f - self.u * self.v;
            let a = f / self.m;
            self.v += a * self.per_sample + gate * self.m * 100.0;
            self.x += self.v * self.per_sample;

            r = r.replace(i, self.x);
        }
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
*/

#[derive(Copy, Clone)]
pub struct Log;

impl Node for Log {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        println!("{:?}", input[0]);
        input
    }
}
/*

#[derive(Clone)]
pub struct Nest<A>(PhantomData<A>);
impl<A> Default for Nest<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ValueT> Node for Nest<A> {
    type Input = A;
    type Output = Value<(A::Inner,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.inner().clone(),))
    }
}

#[derive(Clone)]
pub struct Strip<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Strip<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A: ValueT, B: ValueT<Inner = A::Car>> Node for Strip<A, B> {
    type Input = A;
    type Output = B;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        B::from_inner(input.car().clone())
    }
}

#[derive(Clone)]
pub struct Car<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Car<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A: ValueT, B: ValueT<Inner = (A::Car,)>> Node for Car<A, B> {
    type Input = A;
    type Output = B;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        B::from_inner((input.car().clone(),))
    }
}

#[derive(Default)]
pub struct ProcessSequenceBuilder {
    processors: Vec<Box<Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>>,
}
#[derive(Clone)]
pub struct ProcessSequence {
    processors: Vec<(
        f32,
        Box<Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>,
    )>,
    idx: usize,
    per_sample: f32,
}
impl ProcessSequenceBuilder {
    pub fn add(
        &mut self,
        p: impl Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>> + 'static,
    ) -> &mut Self {
        self.processors.push(Box::new(p));
        self
    }

    pub fn build(self) -> ProcessSequence {
        ProcessSequence {
            processors: self.processors.into_iter().map(|p| (0.0, p)).collect(),
            idx: 0,
            per_sample: 0.0,
        }
    }
}

impl Node for ProcessSequence {
    type Input = Value<(f32x8, (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((gate, signal)) = input;
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            for (i, (t, _)) in self.processors.iter_mut().enumerate() {
                if i == self.idx {
                    *t = (*t + self.per_sample).min(0.01);
                } else {
                    *t = (*t - self.per_sample).max(0.0);
                }
            }
            if gate > 0.5 {
                self.idx = (self.idx + 1) % self.processors.len();
            }
        }
        let mut r = Value((f32x8::splat(0.0), f32x8::splat(0.0)));
        for (i, (t, processor)) in self.processors.iter_mut().enumerate() {
            let v = processor.process(Value(signal));
            let t = f32x8::splat(*t / 0.01);
            (r.0).0 += (v.0).0 * t;
            (r.0).1 += (v.0).1 * t;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
        for (_, p) in &mut self.processors {
            p.set_sample_rate(rate);
        }
    }
}

#[derive(Clone)]
pub struct Delay(DelayLine<f32x8>, DelayLine<f32x8>, f32);
impl Delay {
    pub fn new(len: f32) -> Self {
        let l = DelayLine::new((len * 48000.0) as usize / 8);
        let r = DelayLine::new((len * 48000.0) as usize / 8);
        Self(l, r, len)
    }

    pub fn get(&self, idx: usize) -> Option<(f32x8, f32x8)> {
        if let Some(l) = (self.0).buffer.get(idx) {
            if let Some(r) = (self.0).buffer.get(idx) {
                return Some((*l, *r));
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.0.buffer.len()
    }
}

impl Node for Delay {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((l, r)) = input;
        Value((self.0.pre_swap(l), self.1.pre_swap(r)))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        let len = (rate * self.2) as usize / 8;
        if len != (self.0).buffer.len() {
            self.0 = DelayLine::new(len);
            self.1 = DelayLine::new(len);
        }
    }
}

#[derive(Clone)]
pub struct MonoDelay(DelayLine<f32x8>, f32);
impl MonoDelay {
    pub fn new(len: f32) -> Self {
        let l = DelayLine::new((len * 48000.0) as usize / 8);
        Self(l, len)
    }
}

impl Node for MonoDelay {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((v,)) = input;
        Value((self.0.pre_swap(v),))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        let len = (rate * self.1) as usize / 8;
        if len != (self.0).buffer.len() {
            self.0 = DelayLine::new(len);
        }
    }
}

#[derive(Clone)]
pub struct Stutter {
    delay: Delay,
    pans: Vec<f32>,
    count: u32,
}
impl Stutter {
    pub fn new(count: u32, max_delay: f32) -> Self {
        Self {
            delay: Delay::new(max_delay),
            count,
            pans: vec![0.5; count as usize],
        }
    }

    pub fn rand_pan(count: u32, max_delay: f32) -> Self {
        let mut rng = thread_rng();
        Self {
            delay: Delay::new(max_delay),
            count,
            pans: (0..count).map(|_| rng.gen()).collect(),
        }
    }
}
impl Node for Stutter {
    type Input = Value<(f32x8, (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((len_mult, signal)) = input;
        self.delay.process(Value(signal));
        let (mut l, mut r) = signal;
        let d = (self.delay).0.buffer.len() as f32 / self.count as f32;
        for i in 1..self.count {
            let d = d*len_mult.max_element();
            let p = self.pans[i as usize];
            let m = f32x8::splat((1.0 - ((i as f32 + 1.0) / self.count as f32)).powi(3));
            let i = i as f32 * d;
            let f = i.fract();
            let j = i.ceil() as usize;
            let i = i.floor() as usize;
            let i = ((self.delay.0.idx + (self.delay).0.buffer.len()) - i)
                % (self.delay).0.buffer.len();
            let j = ((self.delay.0.idx + (self.delay).0.buffer.len()) - j)
                % (self.delay).0.buffer.len();
            let v = (self.delay).0.buffer[i] * (1.0-f) + (self.delay).0.buffer[j] * f;
            l += v * m * f32x8::splat(p);
            let v = (self.delay).1.buffer[i] * (1.0-f) + (self.delay).1.buffer[j] * f;
            r += v * m * f32x8::splat(1.0 - p);
        }
        Value((l, r))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.delay.set_sample_rate(rate);
    }
}

#[derive(Copy, Clone)]
pub struct Transpose;
impl Node for Transpose {
    type Input = Value<((f32x8, f32x8), (f32x8, f32x8))>;
    type Output = Value<((f32x8, f32x8), (f32x8, f32x8))>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((a, b), (c, d))) = input;
        Value(((a, c), (b, d)))
    }
}

#[derive(Copy, Clone)]
pub struct Rescale(pub f32, pub f32);
impl Node for Rescale {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut v = (input.0).0;
        v *= f32x8::splat(self.1 - self.0);
        v += f32x8::splat(self.0);
        Value((v,))
    }
}
*/
#[derive(Copy, Clone)]
pub struct ModulatedRescale;
impl Node for ModulatedRescale {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (min, max, mut v) = (input[0], input[1], input[2]);
        v *= max - min;
        v += min;
        arr![f32x8; v]
    }
}
/*
#[derive(Copy, Clone)]
pub struct FnRescale<F>(pub F);
impl<F: Fn() -> (f32, f32) + Clone> Node for FnRescale<F> {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (a, b) = (self.0)();
        let mut v = (input.0).0;
        v *= f32x8::splat(b - a);
        v += f32x8::splat(a);
        Value((v,))
    }
}

#[derive(Copy, Clone)]
pub struct FnSRescale<F>(pub F);
impl<F: Fn() -> (f32, f32) + Clone> Node for FnSRescale<F> {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (a, b) = (self.0)();
        let mut v = (input.0).0;
        v *= b - a;
        v += a;
        Value((v,))
    }
}

#[derive(Copy, Clone)]
pub struct SRescale(pub f32, pub f32);
impl Node for SRescale {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut v = (input.0).0;
        v *= self.1 - self.0;
        v += self.0;
        Value((v,))
    }
}

#[derive(Copy, Clone)]
pub struct FunctionTrigger<F>(pub F);
impl<F> Node for FunctionTrigger<F>
where
    F: FnMut() + Clone,
{
    type Input = Value<(f32x8,)>;
    type Output = NoValue;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        if input.car().max_element() > 0.5 {
            (self.0)();
        }
        NoValue
    }
}
*/
#[derive(Clone)]
pub struct EuclidianPulse {
    pulses: u32,
    len: u32,
    steps: Vec<bool>,
    idx: usize,
    triggered: bool,
}
impl Default for EuclidianPulse {
    fn default() -> Self {
        Self {
            pulses: 0,
            len: 0,
            steps: vec![],
            idx: 0,
            triggered: false,
        }
    }
}

impl Node for EuclidianPulse {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let pulses = input[0].max_element();
        let len = input[1].max_element();
        let gate = input[2];
        let pulses = pulses as u32;
        let len = len as u32;
        if pulses != self.pulses || len != self.len {
            make_euclidian_rhythm(pulses, len, &mut self.steps);
            self.pulses = pulses;
            self.len = len;
        }
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            if gate > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.idx = (self.idx + 1) % self.len as usize;
                    if self.steps[self.idx] {
                        r = r.replace(i, 1.0);
                    }
                }
            } else {
                self.triggered = false;
            }
        }
        arr![f32x8; r]
    }
}

fn make_euclidian_rhythm(pulses: u32, len: u32, steps: &mut Vec<bool>) {
    steps.resize(len as usize, false);
    steps.fill(false);
    let mut bucket = 0;
    for (i, step) in steps.iter_mut().enumerate() {
        bucket += pulses;
        if bucket >= len {
            bucket -= len;
            *step = true;
        }
    }
}

#[derive(Clone)]
pub struct Folder;
impl Node for Folder {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = input[0];
        for i in 0..f32x8::lanes() {
            let mut v = r.extract(i);
            while v.abs() > 1.0 {
                v = v.signum() - (v - v.signum());
            }
            r = r.replace(i, v);
        }
        arr![f32x8; r ]
    }
}

/*

#[derive(Clone, Default)]
pub struct Swing {
    ratio: f32,
    last_dur: f32,
    switched: bool,
    time: f32,
    to_release: f32,
    per_sample: f32,
}
impl Node for Swing {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((ratio, gate)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            let ratio = ratio.extract(i);
            if gate > 0.5 {
                self.last_dur = self.time;
                self.time = 0.0;
            }
            self.time += self.per_sample;
            if (0.0..=self.per_sample).contains(&self.to_release) {
                let dur = self.last_dur;
                if self.switched {
                    self.to_release = dur * ratio;
                } else {
                    self.to_release = dur / ratio;
                }
                self.switched = !self.switched;
                r = r.replace(i, 1.0);
            } else {
                self.to_release -= self.per_sample;
            }
        }
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Clone)]
pub struct Interpolator {
    start_time: f32,
    start_value: f32,
    stop_time: f32,
    stop_value: f32,
    time: f32,
    per_sample: f32,
    reset: Arc<RefCell<Option<(f32, f32, f32, f32)>>>,
}

impl Interpolator {
    pub fn new(v: f32) -> (Self, Arc<RefCell<Option<(f32, f32, f32, f32)>>>) {
        let cell = Arc::new(RefCell::new(None));
        (
            Self {
                start_time: 0.0,
                start_value: v,
                stop_time: 0.0,
                stop_value: v,
                time: 0.0,
                per_sample: 0.0,
                reset: cell.clone(),
            },
            cell,
        )
    }
}

impl Node for Interpolator {
    type Input = NoValue;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        if let Some((start_time, start_value, stop_time, stop_value)) = self.reset.replace(None) {
            self.time = 0.0;
            self.start_time = start_time;
            self.start_value = start_value;
            self.stop_time = stop_time;
            self.stop_value = stop_value;
        }

        let v = if self.time < self.start_time {
            self.start_value
        } else if self.time < self.stop_time {
            let t = self.time - self.start_time;
            let t = t / (self.stop_time - self.start_time);
            t * self.stop_value + (1.0 - t) * self.start_value
        } else {
            self.stop_value
        };

        self.time += self.per_sample * f32x8::lanes() as f32;

        Value((f32x8::splat(v),))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Clone)]
pub struct RandomSequencer<A> {
    sequences: Vec<Vec<A>>,
    sequence_idx: usize,
    element_idx: usize,
    triggered: bool,
}
impl<A> RandomSequencer<A> {
    pub fn new(sequences: Vec<Vec<A>>) -> Self {
        Self {
            sequences,
            sequence_idx: 0,
            element_idx: 1000000,
            triggered: false,
        }
    }
}
impl<A: Clone> Node for RandomSequencer<A> {
    type Input = Value<(f32x8,)>;
    type Output = Value<(A,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut seq = &self.sequences[self.sequence_idx];
        let gate = (input.0).0.extract(0);
        if !self.triggered && gate > 0.5 {
            self.element_idx = self.element_idx + 1;
            if self.element_idx >= seq.len() {
                self.sequence_idx = thread_rng().gen_range(0..self.sequences.len());
                self.element_idx = 0;
                seq = &self.sequences[self.sequence_idx];
            }
            self.triggered = true;
        } else if gate < 0.5 {
            self.triggered = false;
        }
        Value((seq[self.element_idx].clone(),))
    }
}

#[derive(Copy, Clone)]
pub struct Func<F>(pub F);
impl<F: Fn(f32) -> f32 + Clone> Node for Func<F> {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((v,)) = input;
        Value(((self.0)(v),))
    }
}
#[derive(Copy, Clone)]
pub struct BFunc<F>(pub F);
impl<F: Fn(f32x8) -> f32x8 + Clone> Node for BFunc<F> {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((v,)) = input;
        Value(((self.0)(v),))
    }
}

#[derive(Clone)]
pub struct Mixer<A: ValueT, B: ValueT>(
    Vec<(Box<dyn Node<Input = A, Output = B>>, Arc<RefCell<Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>>>>)>,
);

impl<A: ValueT, B: ValueT> Default for Mixer<A, B> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<A: ValueT, B: ValueT> Mixer<A, B> {
    pub fn add_track(
        &mut self,
        track: impl Node<Input = A, Output = B> + 'static,
    ) -> Arc<RefCell<Box< dyn Node<Input=NoValue, Output=Value<(f32,)>>>>> {
        let gain = Constant(1.0);
        self.add_track_with_gain(track, gain)
    }
    pub fn add_track_with_gain(
        &mut self,
        track: impl Node<Input = A, Output = B> + 'static,
        gain: impl Node<Input=NoValue, Output=Value<(f32,)>> + 'static,
    ) -> Arc<RefCell<Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>>>> {
        let gain: Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>> = Box::new(gain);
        let gain = Arc::new(RefCell::new(gain));
        self.0.push((Box::new(track), Arc::clone(&gain)));
        gain
    }
}
impl<A, B> Node for Mixer<A, B>
where
    A: ValueT,
    B: ValueT + std::ops::Add<Output = B> + std::ops::Mul<f32, Output = B>,
{
    type Input = A;
    type Output = B;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (track, gain) = &mut self.0[0];
        let s = track.process(input.clone()).clone();
        let mut r = s * *gain.borrow_mut().process(NoValue).car();
        for (track, gain) in self.0.iter_mut().skip(1) {
            let s = track.process(input.clone());
            r = r + s * *gain.borrow_mut().process(NoValue).car();
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for (track, gain) in &mut self.0 {
            track.set_sample_rate(rate);
            gain.borrow_mut().set_sample_rate(rate);
        }
    }
}

#[derive(Copy, Clone)]
pub struct AnalogToDigital(pub u32);
impl Node for AnalogToDigital {
    type Input = Value<(f32x8,)>;
    type Output = Value<(u32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let a = *input.car();
        let mut r = u32x8::splat(0);
        for i in 0..f32x8::lanes() {
            let a = a.extract(i);
            r = r.replace(i, (a.min(1.0).max(0.0) * 2.0f32.powi(self.0 as i32)) as u32);
        }
        Value((r,))
    }
}
#[derive(Clone)]
pub struct BitsToImpulse(pub Vec<u32>, bool);
impl BitsToImpulse {
    pub fn new(bits: Vec<u32>) -> Self {
        Self(bits, false)
    }
}
impl Node for BitsToImpulse {
    type Input = Value<(u32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let input = *input.car();
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let  input = input.extract(i);
            let mut triggered = false;
            if self.0.iter().any(|k| (input & (1 << (k - 1))) != 0) {
                if !self.1 {
                    self.1 = true;
                    triggered = true;
                }
            } else {
                self.1 = false;
            }
            if triggered {
                r = r.replace(i, 1.0);
            }
        }
        Value((r,))
    }
}

#[derive(Copy, Clone)]
pub struct StereoBoost;
impl Node for StereoBoost {
    type Input = Value<(f32x8, (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((amount, (mut left, mut right))) = input;
        let side = left - right;
        left += side * amount;
        right += -side * amount;
        Value((left, right))
    }
}
*/

#[derive(Copy, Clone)]
pub struct SumSequencer([(u32, f32); 4], u32, bool);
impl SumSequencer {
    pub fn new(f:[(u32, f32); 4]) -> Self {
        Self(f, 0, false)
    }
}
impl Node for SumSequencer {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let gate = input[0];
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            if gate > 0.5 {
                if !self.2 {
                    self.2 = true;
                    self.1 += 1;
                    if self.0.iter().any(|(d,p)| *d != 0 && self.1 % d == 0 && thread_rng().gen::<f32>() < *p) {
                        r = r.replace(i, 10.0);
                    }
                }
            } else {
                self.2 = false;
            }
        }
        arr![f32x8; r]
    }
}
#[derive(Copy, Clone, Debug)]
pub enum Interpolation {
    Constant { value: f32, duration: f32 },
    Linear { start: f32, end: f32, duration: f32 },
}
impl Interpolation {
    pub fn to_rust(&self) -> String {
        match self {
            Interpolation::Constant { value, duration } => {
                format!("Interpolation::Constant {{ value: {:.4}, duration: {:.4} }}", value, duration)
            }
            Interpolation::Linear { start, end, duration } => {
                format!("Interpolation::Linear {{ start: {:.4}, end: {:.4}, duration: {:.4} }}", start, end, duration)
            }
        }
    }

    fn evaluate(&self, time: f32) -> (f32, bool) {
        match self {
            Interpolation::Linear { start, end, duration } => {
                if time >= *duration {
                    (*end, false)
                } else {
                    let t = (duration - time) / duration;
                    (start * t + end * (1.0-t), true)
                }
            }
            Interpolation::Constant { value, duration } => {
                (*value, time <= *duration)
            }
        }
    }
}

#[derive(Clone)]
pub struct Automation {
    steps: Vec<Interpolation>,
    step: usize,
    time: f64,
    pub do_loop: bool,
    per_sample: f64,
}
impl Automation {
    pub fn new(steps: &[Interpolation]) -> Self {
        Self {
            steps: steps.to_vec(),
            step: 0,
            time: 0.0,
            do_loop: true,
            per_sample: 0.0,
        }
    }
}

impl Node for Automation {
    type Input = U0;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let step = &self.steps[self.step];
        let (v, running) = step.evaluate(self.time as f32);
        if !running {
            self.time = 0.0;
            self.step = (self.step + 1) % self.steps.len();
        }
        self.time += self.per_sample;
        arr![f32x8; f32x8::splat(v) ]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = f32x8::lanes() as f64 / rate as f64;
    }
}

#[derive(Copy,Clone)]
pub struct SoftClip;
impl Node for SoftClip {
    type Input = U2;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (l,r) = (input[0], input[1]);
        arr![f32x8; l.tanh(), r.tanh()]
    }
}

#[derive(Clone, Default)]
pub struct Looper {
    samples: Vec<(f32x8,f32x8)>,
    triggered: bool,
    idx: f32
}
impl Node for Looper {
    type Input = U4;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (gate, speed, left, right) = (input[0], input[1], input[2], input[3]);
        if gate.max_element() > 0.5 {
            if !self.triggered {
                self.triggered = true;
                self.samples.clear();
                self.idx=0.0;
            }
            self.samples.push((left, right));
            arr![f32x8; left, right]
        } else {
            self.triggered = false;
            if !self.samples.is_empty() {
                self.idx = self.idx + speed.max_element();
                if self.idx.ceil() >= self.samples.len() as f32 {
                    self.idx -= self.samples.len() as f32;
                } else if self.idx.floor() < 0.0 {
                    self.idx += self.samples.len() as f32;
                }
                let f = f32x8::splat(self.idx.fract());
                let a = self.samples[self.idx.floor() as usize];
                let b = self.samples[self.idx.ceil() as usize % self.samples.len()];
                arr![f32x8; (a.0*(1.0-f) + b.0*f), (a.1*(1.0-f) + b.1*f)]
            } else {
                arr![f32x8; f32x8::splat(0.0), f32x8::splat(0.0)]
            }
        }
    }
}
#[derive(Clone, Default)]
pub struct Toggle {
    value: bool,
    triggered: bool,
}
impl Node for Toggle {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        if input[0].max_element() > 0.5 {
            if !self.triggered {
                self.triggered = true;
                self.value = !self.value;
            }
        } else {
            self.triggered = false;
        }
        arr![f32x8; f32x8::splat(if self.value { 1.0 } else { 0.0 })]
    }
}
/*

#[derive(Copy, Clone, Default)]
pub struct ProbImpulse(bool);
impl Node for ProbImpulse {
    type Input = Value<(f32x8, f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((prob, gate)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            let mut switched = false;
            if !self.0 && gate > 0.5 {
                if random::<f32>() < prob.extract(i).abs() {
                    self.0= true;
                    switched = true;
                }
            } else if gate < 0.5 {
                self.0 = false;
            }
            if switched {
                r = unsafe { r.replace_unchecked(i, 1.0) };
            }
        }
        Value((r,))
    }
}
#[derive(Copy, Clone, Default)]
pub struct AccumulatorImpulse(bool,f32,f32);
impl Node for AccumulatorImpulse {
    type Input = Value<(f32x8, f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((sig, gate)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            let sig = sig.extract(i);
            self.1 += sig.abs() * self.2;
            let mut switched = false;
            if !self.0 && gate > 0.5 {
                if random::<f32>() < self.1 {
                    self.1=0.0;
                    self.0= true;
                    switched = true;
                }
            } else if gate < 0.5 {
                self.0 = false;
            }
            if switched {
                r = unsafe { r.replace_unchecked(i, 1.0) };
            }
        }
        Value((r,))
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.2= 1.0 / rate;
    }
}

#[derive(Copy, Clone, Default)]
pub struct ToPower;
impl Node for ToPower {
    type Input = Value<(f32x8, f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((sig, pow)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let pow = pow.extract(i);
            let sig = sig.extract(i);
            r = r.replace(i, sig.abs().powf(pow)*sig.signum());
        }
        Value((r,))
    }
}

#[derive(Copy, Clone, Default)]
pub struct Register(f32,f32);
impl Node for Register {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.0 = (input.0).0;
        Value((self.1,))
    }

    #[inline]
    fn post_process(&mut self) {
        println!("{}", self.0);
        self.1 = self.0;
    }
}
*/

#[derive(Copy, Clone, Default)]
pub struct Comparator;
impl Node for Comparator {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a,b) = (input[0], input[1]);

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            if a.extract(i) > b.extract(i) {
                r = r.replace(i, 1.0)
            } else {
                r = r.replace(i, 0.0)
            }
        }
        arr![f32x8; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct CXor;
impl Node for CXor {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a,b) = (input[0], input[1]);

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let a = a.extract(i);
            let b = b.extract(i);
            let v = a.max(b).min(-a.min(b));
            r = r.replace(i, v);
        }
        arr![f32x8; r]
    }
}

//https://www.musicdsp.org/en/latest/Filters/92-state-variable-filter-double-sampled-stable.html
#[derive(Copy, Clone, Default)]
pub struct SimperSvf {
    notch: f32,
    low: f32,
    high: f32,
    band: f32,
    sample_rate: f32,
    ty: u32,
}
impl SimperSvf {
    pub fn low_pass() -> Self {
        Self::default()
    }
    pub fn high_pass() -> Self {
        let mut r = Self::default();
        r.ty = 1;
        r
    }
    pub fn band_pass() -> Self {
        let mut r = Self::default();
        r.ty = 2;
        r
    }
    pub fn notch() -> Self {
        let mut r = Self::default();
        r.ty = 3;
        r
    }
}
impl Node for SimperSvf {
    type Input = U4;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (fc, res, drive, sig) = (input[0], input[1], input[2], input[3]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let fc = fc.extract(i);
            let res = res.extract(i);
            let drive = drive.extract(i);
            let sig = sig.extract(i);

            let freq = 2.0*(PI*0.25f32.min(fc/(self.sample_rate*2.0))).sin();
            let damp = (2.0f32*(1.0 - res.powf(0.25))).min(2.0f32.min(2.0/freq - freq*0.5));

            self.notch = sig - damp*self.band;
            self.low   = self.low + freq*self.band;
            self.high  = self.notch - self.low;
            self.band  = freq*self.high + self.band - drive*self.band.powi(3);
            let mut out = 0.5*match self.ty {
                0 => self.low,
                1 => self.high,
                2 => self.band,
                3 => self.notch,
                _ => panic!(),
            };
            self.notch = sig - damp*self.band;
            self.low   = self.low + freq*self.band;
            self.high  = self.notch - self.low;
            self.band  = freq*self.high + self.band - drive*self.band.powi(3);
            out += 0.5*match self.ty {
                0 => self.low,
                1 => self.high,
                2 => self.band,
                3 => self.notch,
                _ => panic!(),
            };
            out  += 0.5*self.low;
            r = r.replace(i,out);
        }
        if !self.notch.is_finite() {
            self.notch = 0.0;
        }
        if !self.low.is_finite() {
            self.low = 0.0;
        }
        if !self.high.is_finite() {
            self.high = 0.0;
        }
        if !self.band.is_finite() {
            self.band = 0.0;
        }
        arr![f32x8; r]
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate=rate;
    }
}

#[derive(Copy, Clone, Default)]
pub struct Portamento {
    current: f32,
    target: f32,
    remaining: u32,
    delta: f32,
    per_sample: f32
}
impl Node for Portamento {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (transition_time,sig) = (input[0], input[1]);
        let mut r = sig;
        for i in 0..f32x8::lanes() {
            let sig = sig.extract(i);
            if sig != self.target {
                let transition_time = transition_time.extract(i);
                self.remaining = (transition_time / self.per_sample) as u32;
                self.delta = (sig-self.current) / self.remaining as f32;
                self.target = sig;
            }
            if self.remaining > 0 {
                self.current += self.delta;
                self.remaining -= 1;
                r = r.replace(i, self.current);
            }
        }
        arr![f32x8; r]
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate;
    }
}

#[derive(Clone)]
pub struct Reverb {
    delay_l: Vec<f32x8>,
    delay_r: Vec<f32x8>,
    allpass_l: AllPass,
    allpass_r: AllPass,
    taps: Vec<(f32,f32)>,
    tap_total: f32,
    idx_l: usize,
    idx_r: usize,
    per_sample: f32,
}
impl Reverb {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(2);
        let taps:Vec<_> = (0..10).map(|_| (rng.gen_range(0.001..1.0),rng.gen::<f32>())).collect();
        let tap_total = taps.iter().map(|(_,t)| t).sum::<f32>();
        Self {
            delay_l: vec![f32x8::splat(0.0); ((0.09*44100.0)/8.0) as usize],
            delay_r: vec![f32x8::splat(0.0); ((0.0904*44100.0)/8.0) as usize],
            allpass_l: AllPass::default(),
            allpass_r: AllPass::default(),
            taps,
            tap_total,
            idx_l: 0,
            idx_r: 0,
            per_sample: 0.0,
        }
    }
}
impl Node for Reverb {
    type Input = U4;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len,gain,left,right) = (input[0], input[1], input[2], input[3]);
        let mut r_l = f32x8::splat(0.0);
        let mut r_r = f32x8::splat(0.0);
        let Reverb {
            delay_l,
            delay_r,
            taps,
            tap_total,
            idx_l,
            idx_r,
            per_sample,
            ..
        } = self;
        let len = len.max_element().max(0.001);
        delay_l.resize(((len / *per_sample)/8.0) as usize, f32x8::splat(0.0));
        delay_r.resize((((len-0.0008).max(0.001) / *per_sample)/8.0) as usize, f32x8::splat(0.0));
        *idx_l = (*idx_l+1) % delay_l.len();
        *idx_r = (*idx_r+1) % delay_r.len();
        for (fti,g) in taps {
            for (dl, idx, r) in [(&mut *delay_l, *idx_l, &mut r_l), (delay_r, *idx_r, &mut r_r)] {
                let l = 1.0/(1.0+ *fti*dl.len() as f32 * *per_sample*8.0).powi(2);
                let l = (l* *g * gain)/ *tap_total;
                let mut i = idx as i32 - (*fti * dl.len() as f32) as i32;
                if i < 0 {
                    i += dl.len() as i32;
                }
                let dl = dl[i as usize % dl.len()];
                *r += dl*l;
            }
        }
        r_l = left - r_l;
        r_r = right - r_r;
        self.delay_l[self.idx_l] = self.allpass_l.process(arr![f32x8; f32x8::splat(1.0), r_l])[0];
        self.delay_r[self.idx_r] = self.allpass_r.process(arr![f32x8; f32x8::splat(1.0), r_r])[0];
        arr![f32x8; r_l, r_r ]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate;
    }
}

#[derive(Clone)]
pub struct ModableDelay {
    line: DelayLine,
    rate: f32,
}
impl ModableDelay {
    pub fn new() -> Self {
        Self {
            line: DelayLine::default(),
            rate: 0.0,
        }
    }
}
impl Node for ModableDelay {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len,sig) = (input[0], input[1]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let len = len.extract(i).max(0.000001);
            self.line.set_delay((len*self.rate) as f64);
            let sig = sig.extract(i);
            self.line.tick(sig as f64);
            r = r.replace(i, self.line.next() as f32);
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate;
    }
}

#[derive(Clone)]
pub struct GenericIdentity<A: ArrayLength<f32x8>>(PhantomData<A>);
impl<A: ArrayLength<f32x8>> Default for GenericIdentity<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ArrayLength<f32x8>> Node for GenericIdentity<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
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
pub struct Inputs;
impl Node for Inputs {
    type Input = U10;
    type Output = U10;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        input
    }
}

const C0: f32 = 16.35;
#[derive(Clone)]
pub struct Quantizer {
    values: Vec<f32>,
}

impl Quantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec()
        }
    }
}

impl Node for Quantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let input = input[0].max_element().max(0.0);
        let freq = C0 * 2.0f32.powf(input);

        let octave = input.floor();
        let mut min_d = f32::INFINITY;
        let mut min_freq = 0.0;
        for v in &self.values {
            let v = v * 2.0f32.powf(octave);
            let d = (v - freq).abs();
            if d < min_d {
                min_d = d;
                min_freq = v;
            }
        }
        arr![f32x8; f32x8::splat(min_freq)]
    }
}

#[derive(Clone)]
pub struct DegreeQuantizer {
    values: Vec<f32>,
}

impl DegreeQuantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec()
        }
    }

    pub fn chromatic() -> Self {
        let mut notes = Vec::with_capacity(12);
        for i in 0..12 {
            notes.push(16.35*2.0f32.powf((i * 100) as f32/1200.0));
        }
        Self::new(&notes)
    }
}

impl Node for DegreeQuantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let input = input[0].max_element();
        let degree = (input + self.values.len() as f32 * 4.0).max(0.0).round() as usize;

        let octave = degree / self.values.len();
        let idx = degree % self.values.len();

        arr![f32x8; f32x8::splat(self.values[idx]*2.0f32.powi(octave as i32))]
    }
}

#[derive(Clone)]
pub struct TritaveDegreeQuantizer {
    values: Vec<f32>,
}

impl TritaveDegreeQuantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec()
        }
    }
}

impl Node for TritaveDegreeQuantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let input = input[0].max_element();
        let degree = (input + self.values.len() as f32 * 2.0).max(0.0).round() as usize;

        let octave = degree / self.values.len();
        let idx = degree % self.values.len();

        arr![f32x8; f32x8::splat(self.values[idx]*3.0f32.powi(octave as i32))]
    }
}

#[derive(Clone, Debug)]
pub enum Subsequence {
    Item(f32, f32),
    Rest(f32),
    Tuplet(Vec<Subsequence>, usize),
    Iter(Vec<Subsequence>, usize),
    Choice(Vec<Subsequence>, usize),
    ClockMultiplier(Box<Subsequence>, f32),
}
impl Subsequence {
    pub fn to_rust(&self) -> String {
        match self {
            Subsequence::Item(a, b) => format!("Subsequence::Item({:.4}, {:.4})",a, b),
            Subsequence::Rest(a) => format!("Subsequence::Rest({:.4})",a),
            Subsequence::Tuplet(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Tuplet(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::Iter(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Iter(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::Choice(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Choice(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::ClockMultiplier(seq, m) => {
                format!("Subsequence::ClockMultiplier(Box::new({}), {:.4})", seq.to_rust(), m)
            }
        }
    }

    fn current(&mut self, pulse: bool, clock_division: f32) -> (Option<f32>, bool, bool, bool, bool, f32) {
        match self {
            Subsequence::Rest(clock) => {
                if pulse {
                    *clock += 1.0;
                }
                let do_tick = if *clock >= clock_division {
                    *clock = 0.0;
                    true
                } else {
                    false
                };
                (None, do_tick, false, false, false, clock_division)
            },
            Subsequence::Item(v, clock) => {
                if pulse {
                    *clock += 1.0;
                }
                let (do_tick, do_trigger, gate) = if *clock >= clock_division {
                    *clock = 0.0;
                    (true, true, true)
                } else {
                    (false, false, true)
                };
                (Some(*v), do_tick, do_trigger, gate, false, clock_division)
            },
            Subsequence::Tuplet(sub_sequence, sub_idx) => {
                let clock_division = clock_division / sub_sequence.len() as f32;
                let (v, do_tick, do_trigger, gate, _, len) = sub_sequence[*sub_idx].current(pulse, clock_division);
                let do_tick = if do_tick {
                    *sub_idx += 1;
                    if *sub_idx >= sub_sequence.len() {
                        *sub_idx = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                (v, do_tick, do_trigger, gate, do_tick, len)
            }
            Subsequence::Iter(sub_sequence, sub_idx) => {
                let (v, do_tick, do_trigger, gate, _, len) = sub_sequence[*sub_idx].current(pulse, clock_division);
                let (do_tick, end_of_cycle) = if do_tick {
                    *sub_idx += 1;
                    let end_of_cycle = if *sub_idx >= sub_sequence.len() {
                        *sub_idx = 0;
                        true
                    } else { false };
                    (true, end_of_cycle)
                } else {
                    (false, false)
                };
                (v, do_tick, do_trigger, gate, end_of_cycle, len)
            }
            Subsequence::Choice(sub_sequence, sub_idx) => {
                let (v, do_tick, do_trigger, gate, _, len) = sub_sequence[*sub_idx].current(pulse, clock_division);
                let do_tick = if do_tick {
                    *sub_idx = thread_rng().gen_range(0..sub_sequence.len());
                    true
                } else {
                    false
                };
                (v, do_tick, do_trigger, gate, false, len)
            }
            Subsequence::ClockMultiplier(sub_sequence, mul) => {
                let mut r = sub_sequence.current(pulse, clock_division* *mul);
                r.5 *= *mul;
                r
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Subsequence::Item(..) | Subsequence::Iter(..) | Subsequence::Choice(..) | Subsequence::Rest(..) => 1,
            Subsequence::Tuplet(sub_sequence, ..) => sub_sequence.len(),
            Subsequence::ClockMultiplier(sub_sequence, mul) => sub_sequence.len(),
        }
    }
}

#[derive(Clone)]
pub struct PatternSequencer {
    sequence: Subsequence,
    per_sample: f32,
    triggered: bool,
    previous_value: f32,
}

impl PatternSequencer {
    pub fn new(sequence: Subsequence) -> Self {
        Self {
            sequence,
            per_sample: 0.0,
            triggered: false,
            previous_value: 0.0,
        }
    }
}

impl Node for PatternSequencer {
    type Input = U1;
    type Output = U5;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r_trig = f32x8::splat(0.0);
        let mut r_gate = f32x8::splat(0.0);
        let mut r_eoc = f32x8::splat(0.0);
        let mut r_value = f32x8::splat(0.0);
        let mut r_len = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let trigger = input[0].extract(i);
            let mut pulse = false;
            if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    pulse = true;
                }
            } else {
                self.triggered = false;
            }

            let (v,_,t,g,eoc,len) = self.sequence.current(pulse, 24.0 * self.sequence.len() as f32);
            if g {
                r_gate = r_gate.replace(i, 1.0);
            }
            if t {
                r_trig = r_trig.replace(i, 1.0);
                r_gate = r_gate.replace(i, 0.0);
            }
            if eoc {
                r_eoc = r_eoc.replace(i, 1.0);
            }
            self.previous_value = v.unwrap_or(self.previous_value);
            r_value = r_value.replace(i, self.previous_value);
            r_len = r_len.replace(i, len/24.0);
        }
        arr![f32x8; r_value, r_trig, r_gate, r_eoc, r_len]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 8.0/rate;
    }
}

#[derive(Clone, Default)]
pub struct BernoulliGate {
    trigger: bool,
    active_gate: bool
}

impl Node for BernoulliGate {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (prob, sig) = (input[0], input[1]);
        let mut r = [f32x8::splat(0.0), f32x8::splat(0.0)];
        for i in 0..f32x8::lanes() {
            let prob = prob.extract(i);
            let sig = sig.extract(i);
            if sig > 0.5 {
                if !self.trigger {
                    self.trigger = true;
                    self.active_gate = thread_rng().gen::<f32>() < prob;
                }
            } else {
                self.trigger = false;
            }

            if self.active_gate {
                r[0] = r[0].replace(i, sig);
            } else {
                r[1] = r[1].replace(i, sig);
            }
        }
        arr![f32x8; r[0], r[1]]
    }
}

#[derive(Clone, Default)]
pub struct Noise {
    clock: f32,
    value: f32,
    positive: bool,
    per_sample: f32,
}
impl Noise {
    pub fn positive() -> Self {
        let mut r = Self::default();
        r.positive = true;
        r
    }
}

impl Node for Noise {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let period = 1.0/freq.extract(i);
            self.clock += self.per_sample;
            if self.clock >= period {
                if self.positive {
                    self.value = thread_rng().gen_range(0.0..1.0);
                } else {
                    self.value = thread_rng().gen_range(-1.0..1.0);
                }
                self.clock = 0.0;
            }
            r = r.replace(i, self.value);
        }

        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate;
    }
}

#[derive(Clone, Default)]
pub struct QuadSwitch {
    triggered: bool,
    pidx: usize,
    slew: f32,
    idx: usize,
    per_sample: f32,
}

impl Node for QuadSwitch {
    type Input = U3;
    type Output = U4;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (trigger, index, sig) = (input[0], input[1], input[2]);
        let mut r = [
            f32x8::splat(0.0),
            f32x8::splat(0.0),
            f32x8::splat(0.0),
            f32x8::splat(0.0),
        ];
        for i in 0..f32x8::lanes() {
            let trigger = trigger.extract(i);
            let sig = sig.extract(i);
            let mut index = index.extract(i);
            if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.pidx = self.idx;
                    while index < 0.0 {
                        index += 4.0;
                    }
                    self.idx =  index as usize % 4;
                    self.slew = 0.0;
                }
            } else {
                self.triggered = false;
            }
            self.slew += self.per_sample*100.0;

            r[self.idx] = r[self.idx].replace(i, r[self.idx].extract(i) + sig * self.slew.min(1.0));
            r[self.pidx] = r[self.pidx].replace(i, r[self.pidx].extract(i) + sig * (1.0-self.slew.min(1.0)));
        }
        arr![f32x8; r[0], r[1], r[2], r[3] ]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate;
    }
}

#[cfg(feature = "square")]
#[derive(Clone)]
pub struct SquareWave {
    osc: hexodsp::dsp::helpers::PolyBlepOscillator,
    per_sample: f32,
}

#[cfg(feature = "square")]
impl Default for SquareWave {
    fn default() -> Self {
        Self {
            osc: hexodsp::dsp::helpers::PolyBlepOscillator::new(0.0),
            per_sample: 0.0,
        }
    }
}


#[cfg(feature = "square")]
impl Node for SquareWave {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (freq, pw) = (input[0], input[1]);
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let freq = freq.extract(i);
            let pw = pw.extract(i);
            r = r.replace(i, self.osc.next_pulse(freq, self.per_sample, pw));
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate;
    }
}

#[derive(Clone, Default)]
pub struct MidSideEncoder;

impl Node for MidSideEncoder {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (l,r) = (input[0], input[1]);
        arr![f32x8; l+r, l-r]
    }
}

#[derive(Clone, Default)]
pub struct MidSideDecoder;

impl Node for MidSideDecoder {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (m, s) = (input[0], input[1]);
        arr![f32x8; m+s,m-s]
    }
}

#[derive(Clone, Default)]
pub struct Pan;

impl Node for Pan {
    type Input = U3;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (pan, l, r) = (input[0], input[1], input[2]);
        let pan_mapped = ((pan + 1.0) / 2.0) * (PI / 2.0);

        arr![f32x8; l*pan_mapped.sin(),r*pan_mapped.cos()]
    }
}

#[derive(Clone, Default)]
pub struct MonoPan;

impl Node for MonoPan {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (pan, i) = (input[0], input[1]);
        let pan_mapped = ((pan + 1.0) / 2.0) * (PI / 2.0);

        arr![f32x8; i*pan_mapped.sin(),i*pan_mapped.cos()]
    }
}

#[derive(Clone)]
pub struct Brownian {
    current: f32,
    triggered: bool,
}

impl Default for Brownian {
    fn default() -> Self {
        Self {
            current: f32::NAN,
            triggered: false,
        }
    }
}

impl Node for Brownian {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (rate, min, max, trig) = (input[0], input[1], input[2], input[3]);

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let rate = rate.extract(i);
            let min = min.extract(i);
            let max = max.extract(i);
            let trig = trig.extract(i);
            if !self.current.is_finite() {
                self.current = thread_rng().gen_range(min..max);
            }
            if trig > 0.5 {
                if !self.triggered {
                    self.current += thread_rng().gen_range(-1.0..1.0) * rate;
                    self.triggered = true;
                }
            } else {
                self.triggered = false;
            }
            if self.current > max {
                self.current -= self.current - max;
            }
            if self.current < min {
                self.current += min - self.current;
            }
            r = r.replace(i, self.current);
        }
        arr![f32x8; r]
    }
}

#[derive(Clone)]
pub struct IndexPort<A: ArrayLength<f32x8>>(usize, PhantomData<A>);
impl<A: ArrayLength<f32x8>> IndexPort<A> {
    pub fn new(port: usize) -> Self {
        IndexPort(port, Default::default())
    }
}

impl<A: ArrayLength<f32x8>> Node for IndexPort<A> {
    type Input = A;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![f32x8; input[self.0]]
    }
}

#[derive(Clone)]
pub struct Markov {
    transitions: Vec<(f32,Vec<(usize,f32)>)>,
    current_state: usize,
    trigger: bool,
}

impl Markov {
    pub fn new(transitions: &[(f32, Vec<(usize, f32)>)]) -> Self {
        Self {
            transitions: transitions.iter().cloned().collect(),
            current_state: 0,
            trigger: false,
        }
    }

    pub fn major_key_chords() -> Self {
        let transitions = vec![
            (3.0, vec![(1, 1.0)]),
            (6.0, vec![(2, 0.5), (3, 0.5)]),
            (4.0, vec![(4, 0.5), (3, 0.5)]),
            (2.0, vec![(4, 0.5), (5, 0.5)]),
            (7.0, vec![(6, 0.5), (5, 0.5)]),
            (5.0, vec![(6, 1.0)]),
            (1.0, vec![(0, 1.0/7.0),(1, 1.0/7.0),(2, 1.0/7.0),(3, 1.0/7.0),(4, 1.0/7.0),(5, 1.0/7.0),(6, 1.0/7.0),]),
        ];
        Self {
            transitions,
            current_state: 0,
            trigger: false,
        }
    }
}

impl Node for Markov {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            if input[0].extract(i) > 0.5 {
                if !self.trigger {
                    self.trigger = true;
                    let transitions = &self.transitions[self.current_state].1;
                    let mut rng = thread_rng();
                    let new_state = transitions.choose_weighted(&mut rng, |(_, w)| *w).unwrap().0;
                    self.current_state = new_state;
                }
            } else {
                self.trigger = false;
            }
            r = r.replace(i, self.transitions[self.current_state].0);
        }
        arr![f32x8; r]
    }
}

#[derive(Copy, Clone)]
pub struct PulseOnChange(f32);
impl Default for PulseOnChange {
    fn default() -> Self {
        Self(f32::NAN)
    }
}

impl Node for PulseOnChange {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let v = input[0].extract(i);
            if v != self.0 {
                self.0 = v;
                r = r.replace(i, 1.0);
            }
        }
        arr![f32x8; r]
    }
}

#[derive(Clone, Default)]
pub struct QuantizedImpulse {
    pending: bool,
    next_imp: f32,
    current_imp: f32,
    aux_current: f32,
    aux_next: f32,
    triggered: bool,
    clock_triggered: bool,
}
impl Node for QuantizedImpulse {
    type Input = U3;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let src_imp = input[0];
        let clock = input[1];
        let src_aux = input[2];

        let mut imp = f32x8::splat(0.0);
        let mut aux = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let src_imp = src_imp.extract(i);
            let clock = clock.extract(i);
            if src_imp > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.aux_next = src_aux.extract(i);
                    self.pending = true;
                    self.next_imp = 1.0;
                }
            } else {
                self.triggered = false;
                self.pending = true;
                self.next_imp = 0.0;
            }
            if clock > 0.5 {
                if !self.clock_triggered {
                    self.clock_triggered = true;
                    if self.pending {
                        self.aux_current = self.aux_next;
                        self.current_imp = self.next_imp;
                        self.pending = false;
                    }
                }
            } else {
                self.clock_triggered = false;
            }
            aux = aux.replace(i, self.aux_current);
            imp = imp.replace(i, self.current_imp);
        }
        arr![f32x8; imp, aux]
    }
}

#[derive(Clone)]
pub struct BowedString {
    nut_to_bow: DelayLine,
    bow_to_bridge: DelayLine,
    per_sample: f64,
    string_filter: OnePole,
    body_filters: [Biquad; 6],
}

impl Default for BowedString {
    fn default() -> Self {
        Self {
            nut_to_bow: DelayLine::default(),
            bow_to_bridge: DelayLine::default(),
            per_sample: 1.0/44100.0,
            string_filter: OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
            body_filters: [
                Biquad::new(1.0,  1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.8852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.9888, -1.9867, 0.9923),
            ]
        }
    }
}

impl Node for BowedString {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let bow_velocity = input[1];
        let bow_force = input[2];
        let bow_position = input[3];
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let freq = freq.extract(i) as f64;
            let bow_velocity = bow_velocity.extract(i) as f64;
            let bow_force = bow_force.extract(i) as f64;
            let total_l = (1.0/freq.max(20.0))/self.per_sample;
            let bow_position = ((bow_position.extract(i) as f64 + 1.0)/2.0).max(0.01).min(0.99);

            let bow_nut_l = total_l*(1.0-bow_position);
            let bow_bridge_l = total_l*bow_position;

            self.nut_to_bow.set_delay(bow_nut_l);
            self.bow_to_bridge.set_delay(bow_bridge_l);

            let nut = -self.nut_to_bow.next();
            let bridge = -self.string_filter.tick(self.bow_to_bridge.next());


            let dv = bow_velocity - (nut+bridge);

            let phat = ((dv+0.001)*bow_force + 0.75).powf(-4.0).max(0.0).min(0.98);



            self.bow_to_bridge.tick(nut + phat*dv);
            self.nut_to_bow.tick(bridge + phat*dv);


            let mut output = bridge;
            for f in self.body_filters.iter_mut() {
                output = f.tick(output);
            }

            r = r.replace(i, output as f32);
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
        self.string_filter = OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), 0.9);
    }
}

#[derive(Clone)]
pub struct ImaginaryGuitar {
    strings: Vec<(DelayLine, f64, OnePole, bool)>,
    per_sample: f64,
    body_filters: [Biquad; 6],
}

impl Default for ImaginaryGuitar {
    fn default() -> Self {
        Self {
            strings: vec![
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/329.63)*44100.0);
                    l
                }, 329.63, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/246.94)*44100.0);
                    l
                }, 246.94, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/196.0)*44100.0);
                    l
                }, 196.0, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/146.83)*44100.0);
                    l
                }, 146.83, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/110.0)*44100.0);
                    l
                }, 110.0, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
                ({
                    let mut l = DelayLine::default();
                    l.set_delay((1.0/82.40)*44100.0);
                    l
                }, 82.0, OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9), false),
            ],
            body_filters: [
                Biquad::new(1.0,  1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.8852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.9888, -1.9867, 0.9923),
            ],
            per_sample: 1.0/44100.0,
        }
    }
}

impl Node for ImaginaryGuitar {
    type Input = U12;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let crossover = 0.005;

            let mut diffusion = 0.0;
            for (s, _, _, _) in &mut self.strings {
                let v = s.next();
                diffusion += v;
            }
            diffusion /= self.strings.len() as f64;
            for f in self.body_filters.iter_mut() {
                diffusion = f.tick(diffusion);
            }
            r = r.replace(i, diffusion as f32);
            for (j, (s, base_freq, f, triggered)) in self.strings.iter_mut().enumerate() {
                let fret = input[j*2].extract(i);
                s.set_delay(((1.0/(*base_freq))/self.per_sample)*(1.0-fret as f64));
                let trigger = input[j*2+1].extract(i);
                if trigger > 0.5 {
                    if !*triggered {
                        *triggered = true;
                        s.add_at(10.0, 1.0);
                    }
                } else {
                    *triggered = false;
                }
                let mut v = s.next();
                v = -f.tick(v +diffusion*crossover as f64);
                s.tick(v);
            }
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
        for (d,freq,f,_) in &mut self.strings {
           *f  = OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), 0.99);
           d.set_delay((1.0/(*freq))/self.per_sample);
        }
    }
}

#[derive(Clone)]
pub struct StringBodyFilter {
    filters: [Biquad; 6],
}

impl Default for StringBodyFilter {
    fn default() -> Self {
        Self {
            filters: [
                Biquad::new(1.0,  1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.8852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.9888, -1.9867, 0.9923),
            ],
        }
    }
}

impl Node for StringBodyFilter {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let mut v = input[0].extract(i) as f64;
            for f in self.filters.iter_mut() {
                v = f.tick(v);
            }
            r = r.replace(i, v as f32);
        }
        arr![f32x8; r]
    }
}

#[derive(Clone)]
pub struct PluckedString {
    line: DelayLine,
    string_filter: OnePole,
    triggered: bool,
    per_sample: f64,
}

impl Default for PluckedString {
    fn default() -> Self {
        Self {
            line: DelayLine::default(),
            string_filter: OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
            triggered: false,
            per_sample: 1.0/44100.0,
        }
    }
}

impl Node for PluckedString {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let freq = input[0].extract(i);
            let trigger = input[1].extract(i);
            let mut slap_threshold = input[2].extract(i);
            if slap_threshold == 0.0 {
                slap_threshold = 1.0;
            }
            self.line.set_delay((1.0/(freq as f64))/self.per_sample);
            let pluck = if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    trigger as f64
                } else {
                    0.0
                }
            } else {
                self.triggered = false;
                0.0
            };
            let v = -self.string_filter.tick(self.line.next());
            self.line.tick(v.min(slap_threshold as f64) + pluck);
            r = r.replace(i, v as f32);
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
        self.string_filter  = OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), 0.99);
    }
}

#[derive(Clone)]
pub struct SympatheticString {
    line: DelayLine,
    string_filter: OnePole,
    per_sample: f64,
}

impl Default for SympatheticString {
    fn default() -> Self {
        Self {
            line: DelayLine::default(),
            string_filter: OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
            per_sample: 1.0/44100.0,
        }
    }
}

impl Node for SympatheticString {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let freq = input[0].extract(i);
            let driver = input[1].extract(i);
            self.string_filter.set_gain(input[2].extract(i) as f64);
            self.line.set_delay((1.0/(freq as f64))/self.per_sample);
            let v = -self.string_filter.tick(self.line.next());
            self.line.tick(v + driver as f64);
            r = r.replace(i, v as f32);
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
        self.string_filter  = OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), 0.99);
    }
}

#[derive(Copy, Clone, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    gain: f64,

    x0: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            gain: 1.0,

            x0: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn tick(&mut self, input: f64) -> f64 {
        self.x0 = self.gain * input;
        let mut output = self.b0 * self.x0 + self.b1 * self.x1 + self.b2 * self.x2;
        output -= self.a2 * self.y2 + self.a1 * self.y1;
        self.x2 = self.x1;
        self.x1 = self.x0;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}


#[derive(Copy, Clone, Default)]
struct OnePole {
    b0: f64,
    a1: f64,

    y1: f64,

    gain: f64,
}

impl OnePole {
    fn new(pole: f64, gain: f64) -> Self {
        let b0 = if pole > 0.0 {
            1.0 - pole
        } else {
            1.0 + pole
        };
        Self {
            b0,
            a1: -pole,

            y1: 0.0,

            gain
        }
    }

    fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    fn tick(&mut self, input: f64) -> f64 {
        let output = self.b0 * self.gain * input - self.a1 * self.y1;
        self.y1 = output;
        output
    }
}

#[derive(Clone, Default)]
struct DelayLine {
    line: Vec<f64>,
    in_index: usize,
    out_index: f64,
    desired_out_index: f64,
    delay: f64,
}

impl DelayLine {
    fn set_delay(&mut self, delay: f64) {
        let delay = delay.max(1.0);
        if self.delay == delay {
            return
        }

        self.delay = delay;
        if delay as usize > self.line.len() {
            self.line.resize(self.delay as usize * 2, 0.0);
        }

        let mut out_index = self.in_index as f64 - delay;
        while out_index < 0.0 {
            out_index += self.line.len() as f64;
        }
        self.desired_out_index = out_index;
    }

    fn add_at(&mut self, amount: f64, position: f64) {
        let mut index = self.in_index as f64 - self.delay * position;
        while index < 0.0 {
            index += self.line.len() as f64;
        }
        self.line[index as usize] += amount;
    }

    fn tick(&mut self, input: f64) {
        self.line[self.in_index] = input;
        self.in_index += 1;
        if self.in_index >= self.line.len() {
            self.in_index = 0;
        }
        self.out_index += 1.0;
        if self.out_index >= self.line.len() as f64{
            self.out_index = 0.0;
        }
        self.desired_out_index += 1.0;
        if self.desired_out_index >= self.line.len() as f64{
            self.desired_out_index = 0.0;
        }
    }

    fn next(&mut self) -> f64 {
        self.out_index += 0.9*(self.desired_out_index - self.out_index);
        let alpha = self.out_index.fract();
        let mut out = self.line[self.out_index as usize] * (1.0-alpha);
        out += if self.out_index + 1.0 >= self.line.len() as f64 {
            self.line[0]
        } else {
            self.line[self.out_index as usize+ 1]
        } * alpha;
        out
    }
}

#[derive(Clone, Default)]
struct BlockDelayLine {
    lines: Vec<f64>,
    width: usize,
    len: f64,
    in_index: usize,
    out_index: f64,
    output_buffer: Vec<f64>,
    delay: f64,
}

impl BlockDelayLine {
    fn new(width: usize, delay: f64) -> Self {
        let mut s = Self {
            lines: vec![0.0; width],
            len: 1.0,
            output_buffer: vec![0.0; width],
            width,
            in_index: 0,
            out_index: 0.0,
            delay: 1.0,
        };
        s.set_delay(delay);
        s
    }

    fn set_delay(&mut self, mut delay: f64) {
        if !delay.is_finite() {
            delay = 1.0;
        }
        delay = delay.max(1.0).min(44100.0*1.0);
        if self.delay == delay {
            return
        }

        self.delay = delay;
        if delay as usize > self.len as usize {
            self.len = (self.delay.ceil() as usize * 2) as f64;
            self.lines.resize(self.len as usize * self.width, 0.0);
        }

        self.out_index = self.in_index as f64 - delay;
        while self.out_index <= 0.0 {
            self.out_index += self.len;
        }
    }

    fn input_buffer(&mut self) -> &mut [f64] {
        let i = self.in_index * self.width;
        &mut self.lines[i .. i+self.width]
    }

    fn tick(&mut self) {
        self.in_index += 1;
        if self.in_index >= self.len as usize {
            self.in_index = 0;
        }
        self.out_index += 1.0;
        if self.out_index >= self.len {
            self.out_index = 0.0;
        }
    }

    fn next(&mut self) -> &[f64] {
        let alpha = self.out_index.fract();
        let i = self.out_index as usize * self.width;
        self.output_buffer.copy_from_slice(&self.lines[i .. i + self.width]);
        let other = if self.out_index + 1.0 >= self.len {
            &self.lines[0 .. self.width]
        } else {
            &self.lines[i+self.width..i+self.width*2]
        };
        let alpha_p = 1.0-alpha;
        self.output_buffer.iter_mut().zip(other).for_each(|(b,o)| *b = *b*alpha_p + *o*alpha);
        &self.output_buffer
    }
}


#[cfg(test)]
mod block_delay_tests {
    use super::*;

    #[test]
    fn no_interp() {
        let mut l = BlockDelayLine::new(10, 10.0);
        println!("{} {} {}", l.len, l.in_index, l.out_index);
        l.input_buffer().copy_from_slice(&[1.0; 10]);
        l.tick();
        assert_eq!(l.next(), &[0.0; 10]);
        for _ in 0..9 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert_eq!(l.next(), &[1.0; 10]);
        l.tick();
        for _ in 0..20 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
    }

    #[test]
    fn small_interp() {
        let mut l = BlockDelayLine::new(10, 10.1);
        l.input_buffer().copy_from_slice(&[1.0; 10]);
        println!("{} {} {}", l.len, l.in_index, l.out_index);
        assert_eq!(l.next(), &[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert!(l.next().iter().all(|v| (1.0-*v).abs() < 0.2));
        l.input_buffer().copy_from_slice(&[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        assert_eq!(l.next(), &[0.0; 10]);
    }
}

#[derive(Clone)]
pub struct WaveMesh {
    lines: BlockDelayLine,
    junctions: Vec<(Option<OnePole>, Vec<usize>, Vec<usize>)>,
    lines_buffer: Vec<f64>,
    sample_rate: f64,
    gain: f64,
}

impl Default for WaveMesh {
    fn default() -> Self {
        let width = 10i32;
        let height = 10i32;

        let mut nodes = indexmap::IndexMap::new();
        let rate = 44100.0;
        for x in 0..width {
            nodes.insert((x,-1), (Some(OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), -0.85)),vec![
                (x, 0)
            ]));
            nodes.insert((x,height), (Some(OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), -0.85)),vec![
                (x, height-1)
            ]));
            for y in 0..width {
                let inputs = vec![
                    (x-1,y),
                    (x+1,y),
                    (x,y-1),
                    (x,y+1),
                ];
                nodes.insert((x,y),(None,inputs));
            }
        }
        for y in 0..width {
            nodes.insert((-1,y),(Some(OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), -0.85)),vec![
                (0,y)
            ]));
            nodes.insert((width,y),(Some(OnePole::new(0.75 - (0.2 * (rate as f64/2.0) / rate as f64), -0.85)),vec![
                (width-1,y)
            ]));
        }
        let mut lines_map = indexmap::IndexMap::new();
        let mut lines = 0;
        let mut junctions = Vec::new();
        for (src, (reflective, ns)) in nodes {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for dst in ns {
                let input = *lines_map.entry((src,dst)).or_insert_with(|| {
                    lines += 1;
                    lines - 1
                });
                inputs.push(input);
                let output = *lines_map.entry((dst,src)).or_insert_with(|| {
                    let mut d = DelayLine::default();
                    lines += 1;
                    lines - 1
                });
                outputs.push(output);
            }
            junctions.push((reflective, inputs, outputs));
        }

        Self {
            lines_buffer: vec![0.0; lines],
            lines: BlockDelayLine::new(lines, 100.0),
            junctions,
            sample_rate: 44100.0,
            gain: 0.85,
        }
    }
}

impl Node for WaveMesh {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gain = input[1].extract(i) as f64;
            if gain != self.gain {
                for (f,_,_) in &mut self.junctions {
                    if let Some(f) = f {
                        f.set_gain(gain);
                    }
                }
            }
            let line_len = input[2].extract(i) as f64 * self.sample_rate * (10.0/self.lines.len);
            self.lines.set_delay(line_len);
            {
                let Self { junctions, lines_buffer, lines, .. } = self;
                let v = lines.next();
                for (reflective, in_edges, out_edges) in junctions {
                    let mut b = 0.0;
                    for (e,o) in in_edges.iter().zip(out_edges.iter()) {
                        let v = v[*e];
                        if reflective.is_none() {
                            lines_buffer[*o] = -v;
                            b += v;
                        } else {
                            lines_buffer[*o] = 0.0;
                            b -= v;
                        }
                    }
                    if let Some(f) = reflective {
                        b = f.tick(b);
                    } else {
                        b *= 0.5;
                    }
                    for o in out_edges {
                        lines_buffer[*o] += b;
                    }
                }
            }
            let trigger = input[0].extract(i) as f64 / self.lines.width as f64;
            let input_idx = self.lines.width / 2;
            r = r.replace(i, self.lines_buffer[0] as f32);
            {
                let Self { lines_buffer, lines, .. } = self;
                let b = lines.input_buffer();
                b.copy_from_slice(&lines_buffer);
                b.iter_mut().for_each(|b| *b += trigger);
                //b[input_idx] += trigger as f64;
                lines.tick();
            }
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate as f64;
        for (f,_,_) in &mut self.junctions {
            if let Some(f) = f {
                *f = OnePole::new(0.75 - (0.2 * 22050.0 / rate as f64), 0.8);
            }
        }
    }
}

// Based on: https://ccrma.stanford.edu/software/clm/compmus/clm-tutorials/pm.html#s-f
#[derive(Clone)]
pub struct Flute {
    embouchure: DelayLine,
    body: DelayLine,
    y2: f64,
    y1: f64,
    rng: rand::rngs::StdRng,
    sample_rate: f64,
}

impl Default for Flute {
    fn default() -> Self {
        Self {
            embouchure: DelayLine::default(),
            body: DelayLine::default(),
            y1: 0.0,
            y2: 0.0,
            rng: rand::rngs::StdRng::from_rng(thread_rng()).unwrap(),
            sample_rate: 44100.0,
        }
    }
}

impl Node for Flute {
    type Input = U7;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let flow = input[1];
        let noise_amount = input[2];
        let feedback_1 = input[3];
        let feedback_2 = input[4];
        let lowpass_cutoff = input[5];
        let embouchure_ratio = input[6];

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let freq = freq.extract(i) as f64;
            self.body.set_delay((1.0/freq)*self.sample_rate);
            self.embouchure.set_delay((1.0/freq)*self.sample_rate*embouchure_ratio.extract(i) as f64);
            let flow = flow.extract(i) as f64;
            let n = (self.rng.gen_range(-1.0..1.0) + self.y2) * 0.5;
            self.y2 = n;
            let excitation = n * flow * noise_amount.extract(i) as f64 + flow;
            let body_out = self.body.next();
            let embouchure_out = self.embouchure.next();
            let embouchure_out = embouchure_out - embouchure_out.powi(3);

            self.embouchure.tick(body_out * feedback_1.extract(i) as f64 + excitation);
            let body_in = embouchure_out + body_out * feedback_2.extract(i) as f64;
            let a = lowpass_cutoff.extract(i) as f64;
            let body_in = a * body_in + (1.0 - a) * self.y1;
            self.y1 = body_in;
            self.body.tick(body_in);

            r = r.replace(i, body_out as f32);
        }
        arr![f32x8; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate as f64;
    }
}
