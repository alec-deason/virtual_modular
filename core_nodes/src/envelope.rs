use generic_array::{arr, typenum::*};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Copy, Clone)]
pub struct ADEnvelope {
    time: f32,
    triggered: bool,
    current: f32,
    per_sample: f32,
}

impl Default for ADEnvelope {
    fn default() -> Self {
        Self {
            time: f32::INFINITY,
            triggered: false,
            current: 0.0,
            per_sample: 0.0,
        }
    }
}

impl Node for ADEnvelope {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (attack, decay, trigger) = (input[0], input[1], input[2]);
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let attack = attack[i];
            let decay = decay[i];
            let trigger = trigger[i];
            if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.time = 0.0;
                }
            } else if self.triggered {
                self.triggered = false
            }
            self.time += self.per_sample;
            let v = if self.time < attack {
                self.time / attack
            } else if self.time < attack + decay {
                let t = (self.time - attack) / decay;
                1.0 - t
            } else {
                0.0
            };
            self.current = self.current * 0.001 + v * 0.999;
            r[i] = self.current;
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Copy, Clone)]
pub struct ADSREnvelope {
    time: f32,
    triggered: bool,
    current: f32,
    per_sample: f32,
}

impl Default for ADSREnvelope {
    fn default() -> Self {
        Self {
            time: f32::INFINITY,
            triggered: false,
            current: 0.0,
            per_sample: 0.0,
        }
    }
}

impl Node for ADSREnvelope {
    type Input = U5;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (attack, decay, sustain, release, gate) =
            (input[0], input[1], input[2], input[3], input[4]);
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let attack = attack[i];
            let decay = decay[i];
            let release = release[i];
            let sustain = sustain[i];
            let gate = gate[i];
            if gate > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.time = 0.0;
                }
            } else if self.triggered {
                self.triggered = false;
                self.time = 0.0;
            }
            self.time += self.per_sample;
            if self.triggered {
                let target = if self.time < attack {
                    self.time / attack
                } else if self.time < attack + decay {
                    let t = (self.time - attack) / decay;
                    (1.0 - t) + t * sustain
                } else {
                    sustain
                };
                self.current = self.current * 0.001 + target * 0.999;
                r[i] = self.current;
            } else if self.time < release {
                r[i] = (1.0 - self.time / release) * sustain;
            }
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
