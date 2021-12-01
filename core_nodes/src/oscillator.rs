use generic_array::{arr, typenum::*};
use rand::prelude::*;
use std::f32::consts::TAU;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone)]
pub struct WaveTable {
    sample_rate: f32,
    table: Vec<f32>,
    len: f32,
    pub idx: f32,
}
impl WaveTable {
    pub fn saw() -> Self {
        let sz = 1024;
        let mut table = vec![0.0; 1024];
        let scale = 1.0;
        let omega = 1.0 / sz as f32;

        for (i, v) in table.iter_mut().enumerate() {
            let mut amp = scale;
            let mut x = 0.0;
            let mut h = 1.0;
            let mut dd;
            let pd = i as f32 / sz as f32;
            let mut hpd = pd;
            loop {
                if (omega * h) < 0.5 {
                    dd = ((omega * h * std::f32::consts::PI).sin() * 0.5 * std::f32::consts::PI)
                        .cos();
                    x += amp * dd * (hpd * 2.0 * std::f32::consts::PI).sin();
                    h += 1.0;
                    hpd = pd * h;
                    amp = scale / h;
                } else {
                    break;
                }
            }
            *v = x;
        }

        Self {
            len: table.len() as f32,
            idx: thread_rng().gen_range(0.0..table.len() as f32),
            table,
            sample_rate: 0.0,
        }
    }

    pub fn noise() -> Self {
        let mut rng = StdRng::seed_from_u64(2);
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
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let reset = input[1];
        let input = input[0];
        let d = 1.0 / (self.sample_rate / self.len);

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            if reset[i] > 0.5 {
                self.idx = 0.0;
            }
            if self.idx >= self.len {
                self.idx -= self.len;
            }
            *r = self.table[self.idx as usize % self.table.len()];
            self.idx += input[i] * d;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
    }
}

#[derive(Clone)]
pub struct Sine {
    phase: f64,
    per_sample: f64,
}

impl Default for Sine {
    fn default() -> Self {
        Self {
            phase: thread_rng().gen(),
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for Sine {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let reset = input[1];

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let v = (TAU as f64 * self.phase).sin();
            if reset[i] > 0.5 {
                self.phase = 0.0;
            }
            self.phase += self.per_sample * freq[i] as f64;
            if self.phase > 1.0 {
                self.phase -= 1.0;
            }
            *r = v as f32;
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
    }
}

#[derive(Clone, Default)]
pub struct PositiveSine {
    sine: Sine,
}

impl Node for PositiveSine {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = self.sine.process(input);
        r[0].iter_mut().for_each(|v| *v = *v * 0.5 + 0.5);
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sine.set_sample_rate(rate);
    }
}

#[derive(Clone, Default)]
pub struct Noise {
    clock: f32,
    value: f32,
    positive: bool,
    per_sample: f32,
}
impl Noise {
    pub fn positive() -> Self {
        Self {
            positive: true,
            ..Default::default()
        }
    }
}

impl Node for Noise {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let period = 1.0 / freq[i];
            self.clock += self.per_sample;
            if self.clock >= period {
                if self.positive {
                    self.value = thread_rng().gen_range(0.0..1.0);
                } else {
                    self.value = thread_rng().gen_range(-1.0..1.0);
                }
                self.clock = 0.0;
            }
            *r = self.value;
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[cfg(feature = "square")]
#[derive(Clone)]
pub struct SquareWave {
    osc: hexodsp::dsp::helpers::PolyBlepOscillator,
    per_sample: f32,
}

#[cfg(feature = "square")]
impl Default for SquareWave {
    fn default() -> Self {
        Self {
            osc: hexodsp::dsp::helpers::PolyBlepOscillator::new(0.0),
            per_sample: 0.0,
        }
    }
}

#[cfg(feature = "square")]
impl Node for SquareWave {
    type Input = U2;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (freq, pw) = (input[0], input[1]);
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let freq = freq[i];
            let pw = pw[i];
            *r = self.osc.next_pulse(freq, self.per_sample, pw);
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
