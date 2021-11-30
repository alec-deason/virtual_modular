#![feature(hash_drain_filter)]
#[macro_use]
pub mod dynamic_graph;
pub mod code_generator;
use generic_array::{
    arr,
    typenum::{U0, U2},
};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

use std::{cell::RefCell, collections::HashMap};

pub struct InstrumentSynth {
    synth: RefCell<Box<dyn Node<Input = U0, Output = U2>>>,
    synth_sample: (usize, Ports<U2>),
    float_parameters: HashMap<String, Box<dyn FnMut(f64)>>,
    float_float_parameters: HashMap<String, Box<dyn FnMut(f64, f64)>>,
    sample_rate: f32,
}
unsafe impl Send for InstrumentSynth {}
unsafe impl Sync for InstrumentSynth {}

#[derive(Default)]
pub struct InstrumentSynthBuilder {
    float_parameters: HashMap<String, Box<dyn FnMut(f64)>>,
    float_float_parameters: HashMap<String, Box<dyn FnMut(f64, f64)>>,
}

impl InstrumentSynthBuilder {
    pub fn float_parameter(mut self, name: &str, f: impl FnMut(f64) + 'static) -> Self {
        self.float_parameters.insert(name.to_string(), Box::new(f));
        self
    }

    pub fn float_float_parameter(mut self, name: &str, f: impl FnMut(f64, f64) + 'static) -> Self {
        self.float_float_parameters
            .insert(name.to_string(), Box::new(f));
        self
    }

    pub fn build_with_synth(
        self,
        synth: impl Node<Input = U0, Output = U2> + 'static,
    ) -> InstrumentSynth {
        InstrumentSynth {
            synth: RefCell::new(Box::new(synth)),
            synth_sample: (
                9,
                arr![[f32; BLOCK_SIZE]; [0.0f32; BLOCK_SIZE], [0.0f32; BLOCK_SIZE]],
            ),
            float_parameters: self.float_parameters,
            float_float_parameters: self.float_float_parameters,
            sample_rate: 4100.0,
        }
    }
}

impl InstrumentSynth {
    pub fn builder() -> InstrumentSynthBuilder {
        InstrumentSynthBuilder::default()
    }

    pub fn set_float_parameter(&mut self, name: &str, value: f64) {
        if let Some(p) = self.float_parameters.get_mut(name) {
            p(value);
        }
    }

    pub fn set_float_float_parameter(&mut self, name: &str, a: f64, b: f64) {
        if let Some(p) = self.float_float_parameters.get_mut(name) {
            p(a, b);
        }
    }

    pub fn replace_synth(
        &self,
        mut synth: impl Node<Input = U0, Output = U2> + 'static,
    ) {
        synth.set_sample_rate(self.sample_rate);
        let synth = Box::new(synth);
        *self.synth.borrow_mut() = synth;
    }
}

impl InstrumentSynth {
    pub fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        self.synth.borrow_mut().set_sample_rate(rate);
    }

    pub fn process(&mut self, out_left: &mut [f32], out_right: &mut [f32]) {
        for (left, right) in out_left.iter_mut().zip(out_right) {
            if self.synth_sample.0 >= BLOCK_SIZE {
                self.synth_sample = (0, self.synth.borrow_mut().process(arr![[f32; BLOCK_SIZE];]));
            }
            let i = self.synth_sample.0;
            *left = (self.synth_sample.1)[0][i];
            *right = (self.synth_sample.1)[1][i];
            self.synth_sample.0 += 1;
        }
    }
}

fn variance(vs: &[f32]) -> f32 {
    let t = vs.iter().sum::<f32>();
    let m = t / vs.len() as f32;
    let e = vs.iter().map(|v| (v - m).powi(2)).sum::<f32>();
    e / vs.len() as f32
}
