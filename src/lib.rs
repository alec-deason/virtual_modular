pub mod instruments;
pub mod simd_graph;
pub mod type_list;
use packed_simd_2::f32x8;
use type_list::{NoValue, Value};

use std::collections::HashMap;

pub struct InstrumentSynth {
    synth: Box<dyn simd_graph::Node<Input = NoValue, Output = Value<(f32x8, f32x8)>>>,
    synth_sample: (usize, (f32x8, f32x8)),
    float_parameters: HashMap<String, Box<dyn FnMut(f64)>>,
    float_float_parameters: HashMap<String, Box<dyn FnMut(f64, f64)>>,
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
        synth: impl simd_graph::Node<Input = NoValue, Output = Value<(f32x8, f32x8)>> + 'static,
    ) -> InstrumentSynth {
        InstrumentSynth {
            synth: Box::new(synth),
            synth_sample: (9, (f32x8::splat(0.0), f32x8::splat(0.0))),
            float_parameters: self.float_parameters,
            float_float_parameters: self.float_float_parameters,
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
}

impl InstrumentSynth {
    pub fn set_sample_rate(&mut self, rate: f32) {
        self.synth.set_sample_rate(rate);
    }

    pub fn process(&mut self, out_left: &mut [f32], out_right: &mut [f32]) {
        for (left, right) in out_left.iter_mut().zip(out_right) {
            if self.synth_sample.0 >= f32x8::lanes() {
                self.synth_sample = (0, self.synth.process(NoValue).0);
            }
            let i = self.synth_sample.0;
            *left = (self.synth_sample.1).0.extract(i);
            *right = (self.synth_sample.1).1.extract(i);
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
