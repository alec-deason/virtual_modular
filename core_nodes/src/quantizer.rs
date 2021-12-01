use generic_array::{
    arr,
    typenum::*,
};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

const C0: f32 = 16.35;
#[derive(Clone)]
pub struct Quantizer {
    values: Vec<f32>,
}

impl Quantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }
}

impl Node for Quantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let input = input[0][i].max(0.0);
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
            r[i] = min_freq;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Clone)]
pub struct DegreeQuantizer {
    values: Vec<f32>,
}

impl DegreeQuantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }

    pub fn chromatic() -> Self {
        let mut notes = Vec::with_capacity(12);
        for i in 0..12 {
            notes.push(16.35 * 2.0f32.powf((i * 100) as f32 / 1200.0));
        }
        Self::new(&notes)
    }
}

impl Node for DegreeQuantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let input = input[0][i];
            let degree = (input + self.values.len() as f32 * 4.0).max(0.0).round() as usize;

            let octave = degree / self.values.len();
            let idx = degree % self.values.len();
            r[i] = self.values[idx] * 2.0f32.powi(octave as i32);
        }

        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Clone)]
pub struct TritaveDegreeQuantizer {
    values: Vec<f32>,
}

impl TritaveDegreeQuantizer {
    pub fn new(values: &[f32]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }
}

impl Node for TritaveDegreeQuantizer {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let input = input[0][i];
            let degree = (input + self.values.len() as f32 * 2.0).max(0.0).round() as usize;

            let octave = degree / self.values.len();
            let idx = degree % self.values.len();
            r[i] = self.values[idx] * 3.0f32.powi(octave as i32);
        }
        arr![[f32; BLOCK_SIZE]; r]
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

        let mut imp = [0.0; BLOCK_SIZE];
        let mut aux = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let src_imp = src_imp[i];
            let clock = clock[i];
            if src_imp > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.aux_next = src_aux[i];
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
            aux[i] = self.aux_current;
            imp[i] = self.current_imp;
        }
        arr![[f32; BLOCK_SIZE]; imp, aux]
    }
}
