use generic_array::{arr, typenum::*};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Default)]
pub struct PulseOnLoad(bool);
impl Node for PulseOnLoad {
    type Input = U0;
    type Output = U1;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        if self.0 {
            arr![[f32; BLOCK_SIZE]; [0.0f32; BLOCK_SIZE]]
        } else {
            self.0 = true;
            let mut r = [0.0f32; BLOCK_SIZE];
            r[0] = 1.0;
            arr![[f32; BLOCK_SIZE]; r]
        }
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
        let mut r = [0.0f32; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let gate = input[i];
            let mut switched = false;
            if !self.1 && gate > self.0 {
                self.1 = true;
                switched = true;
            } else if gate < self.0 {
                self.1 = false;
            }
            if switched {
                *r = 1.0;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct PulseDivider(u64, bool);

impl Node for PulseDivider {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (division, gate) = (input[0], input[1]);
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let gate = gate[i];
            let division = division[i].round() as u64;
            if gate > 0.5 {
                if !self.1 {
                    self.0 += 1;
                    self.1 = true;
                }
            } else if self.1 {
                self.1 = false;
            }
            if self.1 && division > 0 && self.0 % division == 0 {
                *r = gate;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
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
        let mut r = [0.0f32; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let v = input[0][i];
            if v != self.0 {
                self.0 = v;
                *r = 1.0;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct BurstTrigger {
    triggered: bool,
    spacing: f64,
    clock: f64,
    per_sample: f64,
    remaining: u32,
}

impl Node for BurstTrigger {
    type Input = U3;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let trigger = input[0];
        let count = input[1];
        let spacing = input[2];
        let mut r = <Ports<Self::Output> >::default();
        for i in 0..BLOCK_SIZE {
            if self.remaining > 0 {
                self.clock += self.per_sample;
            }
            if trigger[i] > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.remaining = count[i] as u32;
                    self.spacing = spacing[i] as f64;
                    self.clock = self.spacing;
                }
            } else {
                self.triggered = false;
            }
            if self.remaining > 0 && self.clock >= self.spacing {
                self.remaining -= 1;
                self.clock = 0.0;
                r[0][i] = 1.0;
                if self.remaining == 0 {
                    r[1][i] = 1.0;
                }
            }
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
    }
}
