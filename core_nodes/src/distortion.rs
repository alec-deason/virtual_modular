use generic_array::{arr, typenum::*};
use std::f32::consts::PI;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone)]
pub struct Folder;
impl Node for Folder {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = input[0];
        for r in r.iter_mut() {
            while r.abs() > 1.0 {
                *r = r.signum() - (*r - r.signum());
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone)]
pub struct EvenHarmonicDistortion;
impl Node for EvenHarmonicDistortion {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let signal = input[0];
        let gain = input[1];
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let signal = signal[i];
            *r = signal;
            if signal > 0.0 {
                let gain = gain[i];
                *r *= gain;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Clone, Default)]
pub struct MidSideEncoder;

impl Node for MidSideEncoder {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (l, r) = (input[0], input[1]);

        let mut r_l = [0.0; BLOCK_SIZE];
        let mut r_r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let l = l[i];
            let r = r[i];

            r_l[i] = l + r;
            r_r[i] = l - r;
        }
        arr![[f32; BLOCK_SIZE]; r_l, r_r]
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
        let mut r_m = [0.0; BLOCK_SIZE];
        let mut r_s = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let m = m[i];
            let s = s[i];
            r_m[i] = m + s;
            r_s[i] = m - s;
        }
        arr![[f32; BLOCK_SIZE]; r_m, r_s]
    }
}

#[derive(Clone, Default)]
pub struct Pan;

impl Node for Pan {
    type Input = U3;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r_l = [0.0; BLOCK_SIZE];
        let mut r_r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let (pan, l, r) = (input[0][i], input[1][i], input[2][i]);
            let pan_mapped = ((pan + 1.0) / 2.0) * (PI / 2.0);

            r_l[i] = l * pan_mapped.sin();
            r_r[i] = r * pan_mapped.cos();
        }
        arr![[f32; BLOCK_SIZE]; r_l, r_r]
    }
}

#[derive(Clone, Default)]
pub struct MonoPan;

impl Node for MonoPan {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r_l = [0.0; BLOCK_SIZE];
        let mut r_r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let (pan, input) = (input[0][i], input[1][i]);
            let pan_mapped = ((pan + 1.0) / 2.0) * (PI / 2.0);
            r_l[i] = input * pan_mapped.sin();
            r_r[i] = input * pan_mapped.cos();
        }

        arr![[f32; BLOCK_SIZE]; r_l, r_r]
    }
}

#[derive(Clone, Default)]
pub struct SineToSine(crate::Simper);

impl Node for SineToSine {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, mut input: Ports<Self::Input>) -> Ports<Self::Output> {
        self.0.set_parameters(200.0, 0.0);
        for r in input[0].iter_mut() {
            let v = r.sin();
            *r = self.0.low(v).asin();
        }
        input
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}
