use generic_array::{arr, typenum::*};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Default)]
pub struct Constant(pub f32);
impl Node for Constant {
    type Input = U0;
    type Output = U1;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        arr![[f32; BLOCK_SIZE]; [self.0; BLOCK_SIZE]]
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        let n: f32 = parameters.parse().map_err(|e| format!("{}", e))?;
        self.0 = n;
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct Accumulator {
    value: f32,
    sum_triggered: bool,
    reset_triggered: bool,
}
impl Node for Accumulator {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let value = input[0];
        let sum_trigger = input[1];
        let reset_value = input[2];
        let reset_trigger = input[3];

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            if sum_trigger[i] > 0.5 {
                if !self.sum_triggered {
                    self.sum_triggered = true;
                    self.value += value[i];
                }
            } else {
                self.sum_triggered = false;
            }
            if reset_trigger[i] > 0.5 {
                if !self.reset_triggered {
                    self.reset_triggered = true;
                    self.value = reset_value[i];
                }
            } else {
                self.reset_triggered = false;
            }
            *r = self.value;
        }

        arr![[f32; BLOCK_SIZE]; r]
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
        let mut r = [0.0f32; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            if input[0][i] > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.value = !self.value;
                }
            } else {
                self.triggered = false;
            }
            *r = if self.value { 1.0 } else { 0.0 };
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct Comparator;
impl Node for Comparator {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a, b) = (input[0], input[1]);

        let mut r = [0.0f32; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            if a[i] > b[i] {
                *r = 1.0
            } else {
                *r = 0.0
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct CXor;
impl Node for CXor {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (a, b) = (input[0], input[1]);

        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let a = a[i];
            let b = b[i];
            let v = a.max(b).min(-a.min(b));
            r[i] = v;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct SampleAndHold(f32, bool, bool);
impl Node for SampleAndHold {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (gate, signal) = (input[0], input[1]);
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let gate = gate[i];
            let signal = signal[i];
            if !self.1 && gate > 0.5 || !self.2 {
                self.0 = signal;
                self.1 = true;
                self.2 = true;
            } else if gate < 0.5 {
                self.1 = false;
            }
            *r = self.0;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone)]
pub struct ModulatedRescale;
impl Node for ModulatedRescale {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (min, max, mut v) = (input[0], input[1], input[2]);
        for i in 0..BLOCK_SIZE {
            v[i] *= max[i] - min[i];
            v[i] += min[i];
        }
        arr![[f32; BLOCK_SIZE]; v]
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
            [0.0; BLOCK_SIZE],
            [0.0; BLOCK_SIZE],
            [0.0; BLOCK_SIZE],
            [0.0; BLOCK_SIZE],
        ];
        for i in 0..BLOCK_SIZE {
            let trigger = trigger[i];
            let sig = sig[i];
            let mut index = index[i];
            if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.pidx = self.idx;
                    while index < 0.0 {
                        index += 4.0;
                    }
                    self.idx = index as usize % 4;
                    self.slew = 0.0;
                }
            } else {
                self.triggered = false;
            }
            self.slew += self.per_sample * 100.0;

            r[self.idx][i] += sig * self.slew.min(1.0);
            r[self.pidx][i] += sig * (1.0 - self.slew.min(1.0));
        }
        arr![[f32; BLOCK_SIZE]; r[0], r[1], r[2], r[3]]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Copy, Clone, Default)]
pub struct Portamento {
    current: f32,
    target: f32,
    remaining: u32,
    delta: f32,
    per_sample: f32,
}
impl Node for Portamento {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (transition_time, sig) = (input[0], input[1]);
        let mut r = sig;
        for (i, r) in r.iter_mut().enumerate() {
            let sig = sig[i];
            if sig != self.target {
                let transition_time = transition_time[i];
                self.remaining = (transition_time / self.per_sample) as u32;
                self.delta = (sig - self.current) / self.remaining as f32;
                self.target = sig;
            }
            if self.remaining > 0 {
                self.current += self.delta;
                self.remaining -= 1;
                *r = self.current;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
