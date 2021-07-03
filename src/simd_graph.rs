use dyn_clone::DynClone;
use packed_simd_2::{f32x8, u32x8};
use rand::prelude::*;
use linkme::distributed_slice;
use std::{cell::RefCell, f32::consts::TAU, marker::PhantomData, rc::Rc};

use crate::{
    dynamic_graph::{BoxedDynamicNode, MODULES},
    type_list::{Combine, NoValue, Value, ValueT, DynamicValue},
};

pub trait Node: DynClone {
    type Input: ValueT;
    type Output: ValueT;
    fn process(&mut self, input: Self::Input) -> Self::Output;
    fn set_sample_rate(&mut self, rate: f32) {}
}
dyn_clone::clone_trait_object!(<A, B> Node<Input=A, Output=B>);

#[derive(Clone)]
pub struct LfoSine {
    phase: f32,
    per_sample: f32,
    positive: bool,
}
impl Default for LfoSine {
    fn default() -> Self {
        Self {
            phase: thread_rng().gen(),
            per_sample: 0.0,
            positive: false,
        }
    }
}
impl LfoSine {
    pub fn new(phase: f32) -> Self {
        Self {
            phase,
            per_sample: 0.0,
            positive: false,
        }
    }

    pub fn positive_random_phase() -> Self {
        Self {
            phase: thread_rng().gen(),
            per_sample: 0.0,
            positive: true,
        }
    }

    pub fn positive(phase: f32) -> Self {
        Self {
            phase,
            per_sample: 0.0,
            positive: true,
        }
    }
}

impl Node for LfoSine {
    type Input = Value<(f32,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.phase = self.phase + self.per_sample * *input.car() * f32x8::lanes() as f32;
        if self.phase > 500.0 {
            self.phase %= 1.0;
        }
        let mut r = (self.phase * std::f32::consts::TAU).sin();
        if self.positive {
            r = (r + 1.0) / 2.0;
        }
        Value((f32x8::splat(r),))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Copy, Clone)]
pub struct Mul<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Mul<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A, B, C> Node for Mul<A, B>
where
    A: std::ops::Mul<B, Output = C> + Clone,
    B: Clone,
    C: Clone,
{
    type Input = Value<(A, B)>;
    type Output = Value<(C,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.car().clone() * input.cdr().clone(),))
    }
}
#[derive(Copy, Clone)]
pub struct Div<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Div<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A, B, C> Node for Div<A, B>
where
    A: std::ops::Div<B, Output = C> + Copy,
    B: Clone,
    C: Clone,
{
    type Input = Value<(A, B)>;
    type Output = Value<(C,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((*input.car() / input.cdr().clone(),))
    }
}
#[derive(Copy, Clone)]
pub struct Add<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Add<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A, B, C> Node for Add<A, B>
where
    A: std::ops::Add<B, Output = C> + Copy,
    B: Clone,
    C: Clone,
{
    type Input = Value<(A, B)>;
    type Output = Value<(C,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((*input.car() + input.cdr().clone(),))
    }
}
#[distributed_slice(MODULES)]
fn add() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Add::<f32x8, f32x8>::default());
    ("add".to_string(), n)
}
#[distributed_slice(MODULES)]
fn mul() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Mul::<f32x8, f32x8>::default());
    ("mul".to_string(), n)
}
#[derive(Copy, Clone)]
pub struct Sub<A, B>(PhantomData<A>, PhantomData<B>);
impl<A, B> Default for Sub<A, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
impl<A, B, C> Node for Sub<A, B>
where
    A: std::ops::Sub<B, Output = C> + Copy,
    B: Clone,
    C: Clone,
{
    type Input = Value<(A, B)>;
    type Output = Value<(C,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((*input.car() - input.cdr().clone(),))
    }
}

#[derive(Clone)]
pub struct Branch<A: Clone, B: Clone, C: Clone>(A, B, PhantomData<C>);
impl<A, B> Branch<A, B, (A::Output, B::Output)>
where
    A: Node + Clone,
    B: Node + Clone,
{
    pub fn new(a: A, b: B) -> Self {
        Self(a, b, Default::default())
    }
}
impl<A, B, C> Node for Branch<A, B, C>
where
    A: Node + Clone,
    B: Node<Input = A::Input> + Clone,
    C: Combine<A::Output, B::Output> + Clone,
{
    type Input = A::Input;
    type Output = C::Output;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        C::combine(self.0.process(input.clone()), self.1.process(input))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}
#[derive(Clone)]
pub struct Pass<A>(PhantomData<A>);
impl<A> Default for Pass<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ValueT> Node for Pass<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        input
    }
}
#[derive(Clone)]
pub struct Sink<A>(PhantomData<A>);
impl<A> Default for Sink<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A: ValueT> Node for Sink<A> {
    type Input = A;
    type Output = NoValue;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        NoValue
    }
}

#[derive(Clone)]
pub struct Stack<A: Clone, B: Clone, C: Clone, D: Clone>(A, B, PhantomData<C>, PhantomData<D>);
impl<A, B> Stack<A, B, (A::Input, B::Input), (A::Output, B::Output)>
where
    A: Node + Clone,
    B: Node + Clone,
{
    pub fn new(a: A, b: B) -> Self {
        Self(a, b, Default::default(), Default::default())
    }
}
impl<A, B, C, D> Node for Stack<A, B, C, D>
where
    A: Node + Clone,
    B: Node + Clone,
    C: Combine<A::Input, B::Input> + Clone,
    D: Combine<A::Output, B::Output> + Clone,
{
    type Input = C::Output;
    type Output = D::Output;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let (a, b) = C::split(input);
        D::combine(self.0.process(a), self.1.process(b))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct Curry<A, B, C, D, E>(A, B, PhantomData<C>, PhantomData<D>, PhantomData<E>);
impl<A, B, C, D, E> Node for Curry<A, B, C, D, E>
where
    A: Node + Clone,
    B: Node<Input = C::Output> + Clone,
    C: Combine<A::Output, E> + Clone,
    D: Combine<A::Input, E> + Clone,
    E: Clone,
{
    type Input = D::Output;
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
    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.1.process(self.0.process(input))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
        self.1.set_sample_rate(rate);
    }
}
pub fn pipe<I, O, A: Node<Input=I, Output=B::Input> + Clone, B: Node<Output=O> + Clone>(a: A, b: B) -> Pipe<A, B> {
    Pipe(a, b)
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

#[derive(Clone)]
pub struct Constant<A>(pub A);
impl<A: Clone> Node for Constant<A> {
    type Input = NoValue;
    type Output = Value<(A,)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        Value((self.0.clone(),))
    }
}
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
pub struct One;
impl Node for One {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((input.car().extract(0),))
    }
}

#[derive(Clone)]
pub struct RcConstant<A>(pub Rc<RefCell<A>>);
impl<A> RcConstant<A> {
    pub fn new(a: A) -> (Self, Rc<RefCell<A>>) {
        let cell = Rc::new(RefCell::new(a));
        (Self(Rc::clone(&cell)), cell)
    }
}
impl<A: Clone> Node for RcConstant<A> {
    type Input = NoValue;
    type Output = Value<(A,)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        Value((self.0.borrow().clone(),))
    }
}

#[derive(Clone)]
pub struct RawRcConstant<A:ValueT>(pub Rc<RefCell<A::Inner>>);
impl<A:ValueT> RawRcConstant<A> {
    pub fn new(a: A::Inner) -> (Self, Rc<RefCell<A::Inner>>) {
        let cell = Rc::new(RefCell::new(a));
        (Self(Rc::clone(&cell)), cell)
    }
}
impl<A: ValueT> Node for RawRcConstant<A> {
    type Input = NoValue;
    type Output = A;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        A::from_inner(self.0.borrow().clone())
    }
}

#[derive(Clone)]
pub struct Bridge<A: ValueT>(pub Rc<RefCell<A::Inner>>);
impl<A: ValueT> Bridge<A> {
    pub fn new(a: A::Inner) -> (Self, RcConstant<A::Inner>) {
        let (constant, cell) = RcConstant::new(a);
        (Self(cell), constant)
    }
}
impl<A: ValueT> Node for Bridge<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        *self.0.borrow_mut() = input.inner().clone();
        input
    }
}

#[derive(Clone)]
pub struct RawBridge<A: ValueT>(pub Rc<RefCell<A::Inner>>);
impl<A: ValueT> RawBridge<A> {
    pub fn new(a: A::Inner) -> (Self, RawRcConstant<A>) {
        let (constant, cell) = RawRcConstant::new(a);
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

#[distributed_slice(MODULES)]
fn dynamic_short_ad() -> (String, BoxedDynamicNode) {
    let f = move |t: f32, off_time: Option<f32>| {
        let attack = 0.05;
        let release = 0.1;
        if t < attack {
            (t / attack).min(1.0)
        } else {
            let t = t - attack;
            1.0 - (t / release).min(1.0)
        }
    };
    let envelope = ThreshEnvelope::new(f);
    let n = BoxedDynamicNode::new(envelope);
    ("ad".to_string(), n)
}
#[distributed_slice(MODULES)]
fn dynamic_pad_ad() -> (String, BoxedDynamicNode) {
    let f = move |t: f32, off_time: Option<f32>| {
        let attack = 0.5;
        let release = 3.75;
        0.2 * if t < attack {
            (t / attack).min(1.0)
        } else {
            let t = t - attack;
            1.0 - (t / release).min(1.0)
        }
    };
    let envelope = ThreshEnvelope::new(f);
    let n = BoxedDynamicNode::new(envelope);
    ("pad_ad".to_string(), n)
}

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
        let sz = 1024; // the size of the table
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
        let mut rng = thread_rng();
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
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let d = 1.0 / (self.sample_rate / self.len);

        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            if self.idx >= self.len {
                self.idx -= self.len;
            }
            r = unsafe { r.replace_unchecked(i, self.table[self.idx as usize % self.table.len()]) };
            self.idx += (input.0).0.extract(i) * d;
        }
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }
}

#[derive(Clone)]
pub struct Split;
impl Node for Split {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((*input.car(), *input.car()))
    }
}
#[distributed_slice(MODULES)]
fn split() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Split);
    ("split".to_string(), n)
}

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
    PeakingEq(Rc<RefCell<(f32,)>>),
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
                    b2: (1.0 - alpha / a) as f64,
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

    pub fn peaking_eq(db_gain: Rc<RefCell<(f32,)>>) -> Self {
        Self {
            config: BiquadConfig::PeakingEq(db_gain),
            coefficients: Default::default(),

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,

            sample_rate: 44100.0,
        }
    }
}
#[distributed_slice(MODULES)]
fn lpf() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Biquad::lowpass());
    ("lpf".to_string(), n)
}

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
            //println!("{} {:?}", self.sample_rate, self.coefficients);
            let y = (self.coefficients.b0 / self.coefficients.a0) * x
                + (self.coefficients.b1 / self.coefficients.a0) * self.x1
                + (self.coefficients.b2 / self.coefficients.a0) * self.x2
                - (self.coefficients.a1 / self.coefficients.a0) * self.y1
                - (self.coefficients.a2 / self.coefficients.a0) * self.y2;
            //println!("{} {} {} {} {} {}", x, self.x1, self.x2, y, self.y1, self.y2);
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

#[macro_export]
macro_rules! file_constants {
    ($($name:ident : $t:ty = $initial_value:expr),* $(,)*) => {
        {
            use serde::Deserialize;
            use std::{
                rc::Rc,
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
                    $name: Rc<RefCell<$t>>,
                )*
            }
            $(
                let $name = RcConstant::new($initial_value);
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
pub struct RcChoice(pub Rc<RefCell<Vec<f32>>>);
impl Node for RcChoice {
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

#[derive(Copy, Clone, Default)]
pub struct SampleAndHold(f32, bool, bool);
impl Node for SampleAndHold {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = (input.0).0.extract(i);
            let signal = (input.0).1.extract(i);
            if !self.1 && gate > 0.5 || !self.2 {
                self.0 = signal;
                self.1 = true;
                self.2 = true;
            } else if gate < 0.5 {
                self.1 = false;
            }
            r = unsafe { r.replace_unchecked(i, self.0) };
        }
        Value((r,))
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
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = (input.0).0.extract(i);
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
        Value((r,))
    }
}

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
#[derive(Copy, Clone)]
pub struct Invert;
impl Node for Invert {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        Value((f32x8::splat(1.0) - *input.car(),))
    }
}

#[derive(Clone)]
pub struct GateSequencer(Rc<RefCell<Vec<bool>>>, usize, bool);
impl GateSequencer {
    pub fn new(values: Rc<RefCell<Vec<bool>>>) -> Self {
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
pub struct Sequencer<A>(Rc<RefCell<Vec<A>>>, usize, bool);
impl<A> Sequencer<A> {
    pub fn new(values: Rc<RefCell<Vec<A>>>) -> Self {
        Self(values, 0, false)
    }
}
impl<A: Clone> Node for Sequencer<A> {
    type Input = Value<(f32x8,)>;
    type Output = Value<(A,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let seq = self.0.borrow();
        let gate = (input.0).0.extract(0);
        if !self.2 && gate > 0.5 {
            self.1 = (self.1 + 1) % seq.len();
            self.2 = true;
        } else if gate < 0.5 {
            self.2 = false;
        }
        Value((seq[self.1 % seq.len()].clone(),))
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

#[derive(Copy, Clone, Default)]
pub struct AllPass(f32, f32);

impl Node for AllPass {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let scale = *input.car();
        let signal = *input.cdr();
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let scale = scale.extract(i);
            let signal = signal.extract(i);
            let v = scale * signal + self.0 - scale * self.1;
            self.0 = signal;
            self.1 = v;
            r = unsafe { r.replace_unchecked(i, v) };
        }
        Value((r,))
    }
}

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

#[derive(Copy, Clone, Default)]
pub struct PulseDivider(u32, bool);

impl Node for PulseDivider {
    type Input = Value<(u32, f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((division, gate)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            if gate > 0.5 {
                if !self.1 {
                    self.0 += 1;
                    self.1 = true;
                }
            } else if self.1 {
                self.1 = false;
                if self.0 == division {
                    self.0 = 0;
                }
            }
            if self.1 && self.0 == division {
                r = r.replace(i, gate);
            }
        }
        Value((r,))
    }
}

#[derive(Clone)]
pub struct Compressor {
    threshold: f32,
    time: f32,
    decaying: bool,
    per_sample: f32,
}
impl Compressor {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            time: 0.0,
            decaying: false,
            per_sample: 0.0,
        }
    }
}
impl Node for Compressor {
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
            if left.abs() > self.threshold || right.abs() > self.threshold {
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

#[derive(Clone)]
pub struct SidechainCompressor {
    threshold: f32,
    time: f32,
    decaying: bool,
    per_sample: f32,
}
impl SidechainCompressor {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            time: 0.0,
            decaying: false,
            per_sample: 0.0,
        }
    }
}
impl Node for SidechainCompressor {
    type Input = Value<(f32x8, (f32x8, f32x8))>;
    type Output = Value<(f32x8, f32x8)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((sidechain, (left, right))) = input;
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
                let t = (self.time / 0.02).min(1.0);
                let m = (1.0 - t) + t * 0.75;
                l = l.replace(i, left * m);
                r = r.replace(i, right * m);
                self.time += self.per_sample;
            } else {
                if !self.decaying {
                    self.decaying = true;
                    self.time = 0.0;
                }
                let t = (self.time / 0.02).min(1.0);
                let m = (1.0 - t) * 0.75 + t;
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

#[derive(Copy, Clone)]
pub struct Log<A>(PhantomData<A>);
impl<A> Default for Log<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}
#[distributed_slice(MODULES)]
fn dynamic_log() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Log::<Value<(f32x8,)>>::default());
    ("log".to_string(), n)
}

impl<A: ValueT + std::fmt::Debug> Node for Log<A> {
    type Input = A;
    type Output = A;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        println!("{:?}", input);
        input
    }
}

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
            let i = (i as f32 * d) as usize;
            let i = ((self.delay.0.idx + (self.delay).0.buffer.len()) - i)
                % (self.delay).0.buffer.len();
            l += (self.delay).0.buffer[i] * m * f32x8::splat(p);
            r += (self.delay).1.buffer[i] * m * f32x8::splat(1.0 - p);
        }
        Value((l, r))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.delay.set_sample_rate(rate);
    }
}
#[distributed_slice(MODULES)]
fn dynamic_stutter_reverb() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(Stutter::rand_pan(50, 0.35));
    ("stutter_reverb".to_string(), n)
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
#[derive(Copy, Clone)]
pub struct ModulatedRescale;
impl Node for ModulatedRescale {
    type Input = Value<((f32x8, f32x8), f32x8,)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((min, max), mut v)) = input;
        v *= max - min;
        v += min;
        Value((v,))
    }
}
#[distributed_slice(MODULES)]
fn rescale() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode::new(ModulatedRescale);
    ("rescale".to_string(), n)
}

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

#[derive(Clone)]
pub struct EuclidianPulse {
    pulses: u32,
    len: u32,
    steps: Vec<bool>,
    idx: usize,
}
impl Default for EuclidianPulse {
    fn default() -> Self {
        Self {
            pulses: 0,
            len: 0,
            steps: vec![],
            idx: 0,
        }
    }
}

impl Node for EuclidianPulse {
    type Input = Value<((f32, f32), f32x8)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value(((pulses, len), gate)) = input;
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
                self.idx = (self.idx + 1) % self.len as usize;
                if self.steps[self.idx] {
                    r = r.replace(i, 1.0);
                }
            }
        }
        Value((r,))
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
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((mut r,)) = input;
        for i in 0..f32x8::lanes() {
            let mut v = r.extract(i);
            while v.abs() > 1.0 {
                v = v.signum() - (v - v.signum());
            }
            r = r.replace(i, v);
        }
        Value((r,))
    }
}

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
    reset: Rc<RefCell<Option<(f32, f32, f32, f32)>>>,
}

impl Interpolator {
    pub fn new(v: f32) -> (Self, Rc<RefCell<Option<(f32, f32, f32, f32)>>>) {
        let cell = Rc::new(RefCell::new(None));
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
    Vec<(Box<dyn Node<Input = A, Output = B>>, Rc<RefCell<Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>>>>)>,
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
    ) -> Rc<RefCell<Box< dyn Node<Input=NoValue, Output=Value<(f32,)>>>>> {
        let gain = Constant(1.0);
        self.add_track_with_gain(track, gain)
    }
    pub fn add_track_with_gain(
        &mut self,
        track: impl Node<Input = A, Output = B> + 'static,
        gain: impl Node<Input=NoValue, Output=Value<(f32,)>> + 'static,
    ) -> Rc<RefCell<Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>>>> {
        let gain: Box<dyn Node<Input=NoValue, Output=Value<(f32,)>>> = Box::new(gain);
        let gain = Rc::new(RefCell::new(gain));
        self.0.push((Box::new(track), Rc::clone(&gain)));
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

#[derive(Copy, Clone)]
pub struct SumSequencer(fn() -> [(u32, f32); 4], u32, bool);
impl SumSequencer {
    pub fn new(f: fn() -> [(u32, f32); 4]) -> Self {
        Self(f, 0, false)
    }
}
impl Node for SumSequencer {
    type Input = Value<(f32x8,)>;
    type Output = Value<(f32x8,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let Value((gate,)) = input;
        let mut r = f32x8::splat(0.0);
        for i in 0..f32x8::lanes() {
            let gate = gate.extract(i);
            if gate > 0.5 {
                if !self.2 {
                    self.2 = true;
                    self.1 += 1;
                    if self.0().iter().any(|(d,p)| *d != 0 && self.1 % d == 0 && thread_rng().gen::<f32>() < *p) {
                        r = r.replace(i, 10.0);
                    }
                }
            } else {
                self.2 = false;
            }
        }
        Value((r,))
    }
}

#[derive(Copy, Clone)]
pub enum Interpolation {
    Constant { value: f32, duration: f32 },
    Linear { start: f32, end: f32, duration: f32 },
}
impl Interpolation {
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
                (*value, time < *duration)
            }
        }
    }
}

#[derive(Clone)]
pub struct Automation {
    steps: Vec<Interpolation>,
    step: usize,
    time: f32,
    pub do_loop: bool,
    per_sample: f32,
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
    type Input = NoValue;
    type Output = Value<(f32,)>;

    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let step = &self.steps[self.step];
        let (v, running) = step.evaluate(self.time);
        if !running {
            self.time = 0.0;
            self.step = (self.step + 1) % self.steps.len();
        } else {
            self.time += self.per_sample;
        }
        Value((v,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = f32x8::lanes() as f32 / rate;
    }
}
