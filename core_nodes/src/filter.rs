use generic_array::{arr, typenum::*};
use std::f32::consts::PI;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};
use crate::{DelayLine, utils::make_coprime};

#[derive(Clone, Default)]
pub struct AllPass {
    delay: DelayLine,
    pub gain: f64,
    rate: f64,
}

impl AllPass {
    pub fn new(gain: f64, delay: f64) -> Self {
        let mut line = DelayLine::default();
        line.set_delay(delay);
        Self {
            delay: line,
            gain,
            rate: 48000.0,
        }
    }

    pub fn set_delay(&mut self, delay: f64) {
        self.delay.set_delay(delay);
    }

    pub fn tick(&mut self, signal: f64) -> f64 {
        let v = -self.gain * signal + self.delay.current();
        self.delay.tick(signal + self.gain * v);
        v
    }
}

impl Node for AllPass {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (scale, signal, delay) = (input[0], input[1], input[2]);
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            self.gain = scale[i] as f64;
            self.set_delay(delay[i] as f64 * self.rate);
            let signal = signal[i] as f64;
            *r = self.tick(signal) as f32;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as f64;
    }
}

#[derive(Clone, Default)]
pub struct Comb {
    delay: DelayLine,
    pub gain: f64,
    rate: f64,
}

impl Comb {
    pub fn new(gain: f64, delay: f64) -> Self {
        let mut line = DelayLine::default();
        line.set_delay(delay);
        Self {
            delay: line,
            gain,
            rate: 48000.0,
        }
    }

    pub fn set_delay(&mut self, delay: f64) {
        self.delay.set_delay(delay);
    }

    pub fn tick(&mut self, signal: f64) -> f64 {
        let v = signal + self.gain*self.delay.current();
        self.delay.tick(v);
        v
    }
}

impl Node for Comb {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (scale, signal, delay) = (input[0], input[1], input[2]);
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            self.gain = scale[i] as f64;
            self.set_delay(delay[i] as f64 * self.rate);
            let signal = signal[i] as f64;
            *r = self.tick(signal) as f32;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as f64;
    }
}

//TODO: unify with the main comb filter implementation
#[derive(Clone, Default)]
pub struct LPComb {
    delay: DelayLine,
    filter: OnePole,
    pub gain: f64,
}

impl LPComb {
    pub fn new(gain: f64, damping: f64, delay: f64) -> Self {
        let mut line = DelayLine::default();
        line.set_delay(delay);
        Self {
            delay: line,
            filter: OnePole::new(1.0 - damping, -damping),
            gain,
        }
    }

    pub fn set_delay(&mut self, delay: f64) {
        self.delay.set_delay(delay);
    }

    pub fn tick(&mut self, signal: f64) -> f64 {
        let v = signal + self.gain*self.filter.tick(self.delay.current());
        self.delay.tick(v);
        v
    }
}

impl Node for LPComb {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (scale, signal, delay) = (input[0], input[1], input[2]);
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            self.gain = scale[i] as f64;
            self.set_delay(delay[i] as f64);
            let signal = signal[i] as f64;
            *r = self.tick(signal) as f32;
        }
        r
    }
}

#[derive(Clone, Default)]
pub struct ParallelCombs {
    combs: Vec<(f64, Comb)>,
    rate: f64,
}

impl Node for ParallelCombs {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let signal = input[0];
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            let signal = signal[i] as f64;
            for (_, comb) in &mut self.combs {
                *r += comb.tick(signal) as f32;
            }
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as f64;
        let mut delays:Vec<_> = self.combs.iter().map(|(d, _)| (d*self.rate) as u32).collect()
 ;
        make_coprime(&mut delays);
        for (d, (_, comb)) in delays.iter().zip(&mut self.combs) {
            comb.set_delay(*d as f64);
        }
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.combs = parse_combs(parameters)?;
        self.set_sample_rate(self.rate as f32);
        Ok(())
    }
}

fn parse_combs(data: &str) -> Result<Vec<(f64, Comb)>, String> {
    use pom::parser::Parser;
    use pom::parser::*;
    use std::str::{self, FromStr};
    fn number<'a>() -> Parser<'a, u8, f64> {
        let integer = one_of(b"0123456789").repeat(0..);
        let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
        let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
        let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
        number
            .collect()
            .convert(str::from_utf8)
            .convert(f64::from_str)
            .name("number")
    }
    fn whitespace<'a>() -> Parser<'a, u8, Vec<u8>> {
        sym(b' ').repeat(0..).name("whitespace")
    }
    fn item<'a>() -> Parser<'a, u8, (f64, Comb)> {
        (sym(b'(') * whitespace() * number() - whitespace() - sym(b',') - whitespace() + number() - whitespace() - sym(b')')).map(|(gain, delay)| (delay, Comb::new(gain, delay)))
    }
    let parsed = list(item(), whitespace())
        .parse(data.as_bytes())
        .map_err(|e| format!("{:?}", e));
    parsed
}

#[derive(Clone, Default)]
pub struct SerialAllPasses {
    all_passes: Vec<(f64, AllPass)>,
    rate: f64,
}

impl Node for SerialAllPasses {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let signal = input[0];
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            let mut signal = signal[i] as f64;
            for (_, all_pass) in &mut self.all_passes {
                signal  = all_pass.tick(signal);
            }
            *r = signal as f32;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as f64;
        let mut delays:Vec<_> = self.all_passes.iter().map(|(d, _)| (d*self.rate) as u32).collect()
 ;
        make_coprime(&mut delays);
        for (d, (_, all_pass)) in delays.iter().zip(&mut self.all_passes) {
            all_pass.set_delay(*d as f64);
        }
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.all_passes = parse_allpass(parameters)?;
        self.set_sample_rate(self.rate as f32);
        Ok(())
    }
}

fn parse_allpass(data: &str) -> Result<Vec<(f64, AllPass)>, String> {
    use pom::parser::Parser;
    use pom::parser::*;
    use std::str::{self, FromStr};
    fn number<'a>() -> Parser<'a, u8, f64> {
        let integer = one_of(b"0123456789").repeat(0..);
        let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
        let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
        let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
        number
            .collect()
            .convert(str::from_utf8)
            .convert(f64::from_str)
            .name("number")
    }
    fn whitespace<'a>() -> Parser<'a, u8, Vec<u8>> {
        sym(b' ').repeat(0..).name("whitespace")
    }
    fn item<'a>() -> Parser<'a, u8, (f64, AllPass)> {
        (sym(b'(') * whitespace() * number() - whitespace() - sym(b',') - whitespace() + number() - whitespace() - sym(b')')).map(|(gain, delay)| (delay, AllPass::new(gain, delay)))
    }
    let parsed = list(item(), whitespace())
        .parse(data.as_bytes())
        .map_err(|e| format!("{:?}", e));
    parsed
}
// Based on https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
#[derive(Copy, Clone, Default)]
pub struct Simper {
    cutoff: f32,
    resonance: f32,
    k: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    ic1eq: f32,
    ic2eq: f32,
    sample_rate: f32,
}
impl Simper {
    pub fn set_parameters(&mut self, cutoff: f32, resonance: f32) {
        if cutoff != self.cutoff || resonance != self.resonance {
            self.cutoff = cutoff;
            self.resonance = resonance;
            let g = (PI * (cutoff / self.sample_rate)).tan();
            self.k = 2.0 - 2.0 * resonance.min(1.0).max(0.0);

            self.a1 = 1.0 / (1.0 + g * (g + self.k));
            self.a2 = g * self.a1;
            self.a3 = g * self.a2;
        }
    }

    pub fn tick(&mut self, input: f32) -> (f32, f32) {
        let v3 = input - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;

        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;
        if !(self.ic1eq.is_finite() && self.ic2eq.is_finite()) {
            self.ic1eq = 0.0;
            self.ic2eq = 0.0;
        }

        (v1, v2)
    }

    pub fn low(&mut self, input: f32) -> f32 {
        self.tick(input).1
    }

    pub fn band(&mut self, input: f32) -> f32 {
        self.tick(input).0
    }

    pub fn high(&mut self, input: f32) -> f32 {
        let (v1, v2) = self.tick(input);
        input - self.k * v1 - v2
    }

    pub fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }
}

macro_rules! simper {
    ($struct_name:ident, $tick:ident) => {
        #[derive(Copy, Clone, Default)]
        pub struct $struct_name(Simper);
        impl Node for $struct_name {
            type Input = U3;
            type Output = U1;

            #[inline]
            fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
                let cutoff = input[0];
                let resonance = input[1];
                let input = input[2];
                let mut r = [0.0; BLOCK_SIZE];
                for i in 0..BLOCK_SIZE {
                    let input = input[i];
                    let cutoff = cutoff[i];
                    let resonance = resonance[i];
                    self.0.set_parameters(cutoff, resonance);
                    r[i] = self.0.$tick(input);
                }
                arr![[f32; BLOCK_SIZE]; r]
            }

            fn set_sample_rate(&mut self, rate: f32) {
                self.0.set_sample_rate(rate);
            }
        }
    }
}
simper!(SimperLowPass, low);
simper!(SimperHighPass, high);
simper!(SimperBandPass, band);

#[derive(Clone, Default)]
pub struct DCBlocker {
    x: f64,
    y: f64,
}

impl DCBlocker {
    pub fn tick(&mut self, x: f64) -> f64 {
        let y = x - self.x + 0.995 * self.y;
        self.x = x;
        self.y = y;
        y
    }
}
#[derive(Copy, Clone, Default)]
pub struct Biquad {
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
    pub fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
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

    pub fn tick(&mut self, input: f64) -> f64 {
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
pub struct OnePole {
    b0: f64,
    a1: f64,

    y1: f64,

    gain: f64,
}

impl OnePole {
    pub fn new(pole: f64, gain: f64) -> Self {
        let b0 = if pole > 0.0 { 1.0 - pole } else { 1.0 + pole };
        Self {
            b0,
            a1: -pole,

            y1: 0.0,

            gain,
        }
    }

    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    pub fn tick(&mut self, input: f64) -> f64 {
        let output = self.b0 * self.gain * input - self.a1 * self.y1;
        self.y1 = output;
        output
    }
}
