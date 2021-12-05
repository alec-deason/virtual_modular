use generic_array::{arr, typenum::*};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

const A0: f32 = 27.50;
const SEMITONE: f32 = 1.05946;

#[derive(Clone)]
pub struct DegreeQuantizer {
    pitches: Vec<f32>,
    cache_key: (f32, f32),
    cached_value: f32,
}

impl Default for DegreeQuantizer {
    fn default() -> Self {
        Self {
            pitches: Vec::new(),
            cache_key: (f32::NAN, f32::NAN),
            cached_value: 0.0
        }
    }
}

impl Node for DegreeQuantizer {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let degree = input[0];
        let root = input[1];
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let degree = degree[i];
            let root = root[i];
            if (degree, root) != self.cache_key {
                self.cache_key = (degree, root);
                let degree = (degree + self.pitches.len() as f32 * 4.0).max(0.0).round() as usize;

                let octave = degree / self.pitches.len();
                let idx = degree % self.pitches.len();
                self.cached_value = self.pitches[idx] * 2.0f32.powi(octave as i32) * SEMITONE.powi(root as i32);
            }
            *r = self.cached_value;
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        let mut new_pitches = vec![A0];
        for n in parameters.split(" ") {
            let next_value = match n {
                "W" => new_pitches[new_pitches.len()-1] * SEMITONE.powi(2),
                "H" => new_pitches[new_pitches.len()-1] * SEMITONE,
                _ => return Err(format!("Unknown interval {}", n))
            };
            new_pitches.push(next_value);
        }
        self.pitches = new_pitches;
        Ok(())
    }
}

#[cfg(test)]
mod quantizer_tests {
    use super::*;

    #[test]
    fn basic() {
        let mut quantizer = DegreeQuantizer::default();
        // major scale
        quantizer.set_static_parameters("W W H W W W").unwrap();

        for (degree, expected) in &[(0, 440.0), (1,  493.88), (2,  554.37), (3, 587.33), (4,  659.25), (5,  739.99), (6,  830.61)] {
            // first degree, a as root
            let input = arr![[f32; BLOCK_SIZE]; [*degree as f32; BLOCK_SIZE], [0.0f32; BLOCK_SIZE]];
            let output = quantizer.process(input);
            for i in 0..BLOCK_SIZE {
                assert!(output[0][i] - expected < 0.03);
            }
            // again up an octave
            let input = arr![[f32; BLOCK_SIZE]; [*degree as f32 + 7.0; BLOCK_SIZE], [0.0f32; BLOCK_SIZE]];
            let output = quantizer.process(input);
            for i in 0..BLOCK_SIZE {
                assert!(output[0][i] - expected*2.0 < 0.03, "Got {}, expected {}", output[0][i], expected*2.0);
            }
        }
    }

    #[test]
    fn root_shift() {
        let mut quantizer = DegreeQuantizer::default();
        // natural minor scale
        quantizer.set_static_parameters("W H W W H W").unwrap();

        for (degree, expected) in &[(0, 261.63), (1, 293.66), (2, 311.13), (3, 349.23), (4, 392.00), (5, 415.30), (6, 466.16)] {
            // first degree, c as root
            let input = arr![[f32; BLOCK_SIZE]; [*degree as f32; BLOCK_SIZE], [-9.0f32; BLOCK_SIZE]];
            let output = quantizer.process(input);
            for i in 0..BLOCK_SIZE {
                assert!(output[0][i] - expected < 0.03, "Got {}, expected {}", output[0][i], expected);
            }
            // again up an octave
            let input = arr![[f32; BLOCK_SIZE]; [*degree as f32 + 7.0; BLOCK_SIZE], [-9.0f32; BLOCK_SIZE]];
            let output = quantizer.process(input);
            for i in 0..BLOCK_SIZE {
                assert!(output[0][i] - expected*2.0 < 0.03, "Got {}, expected {}", output[0][i], expected*2.0);
            }
        }
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
