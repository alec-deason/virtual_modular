use generic_array::{arr, typenum::*};
use rand::prelude::*;
use std::f64::consts::{TAU, PI};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone)]
pub struct WaveTable {
    sample_rate: f32,
    table: Vec<f32>,
    len: f32,
    pub idx: f32,
}
impl WaveTable {
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

// Based on: http://till.com/articles/sineshaper/
#[derive(Copy, Clone)]
pub struct TanhShaper {
    phase: f64,
    per_sample: f64,
}

impl Default for TanhShaper {
    fn default() -> Self {
        Self {
            phase: thread_rng().gen(),
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for TanhShaper {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let param = input[1];

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let x = if self.phase < 0.5 {
                self.phase as f32 * 4.0 - 1.0
            } else {
                (1.0 - self.phase as f32) * 4.0 - 1.0
            } * std::f32::consts::PI
                * 0.5;
            *r = 9.0 * (0.3833 * x).tanh() - 3.519 * param[i] * x;
            self.phase += self.per_sample * freq[i] as f64;
            while self.phase > 1.0 {
                self.phase -= 1.0;
            }
        }

        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
    }
}

// Based on: http://www.martin-finke.de/blog/articles/audio-plugins-018-polyblep-oscillator/
fn poly_blep(mut t: f64, dt: f64) -> f64 {
    if t < dt {
        t /= dt;
        2.0 * t - t.powi(2) - 1.0
    } else if t > 1.0 - dt {
        t = (t - 1.0) / dt;
        t.powi(2) + 2.0 * t + 1.0
    } else {
        0.0
    }
}

macro_rules! oscillator {
    ($name:ident, $aux_inputs:ty, $body:expr) => {
        #[derive(Clone)]
        pub struct $name {
            phase: f64,
            per_sample: f64,
            previous: f64,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    phase: thread_rng().gen(),
                    per_sample: 1.0 / 44100.0,
                    previous: 0.0,
                }
            }
        }

        impl Node for $name {
            type Input = <$aux_inputs as core::ops::Add<U1>>::Output;
            type Output = U1;

            #[inline]
            fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
                let freq = input[0];

                let mut r = [0.0; BLOCK_SIZE];
                for (i, r) in r.iter_mut().enumerate() {
                    let d = self.per_sample * freq[i] as f64;
                    self.phase = (self.phase + d).fract();
                    self.previous = $body(self.phase, d, &input, i, self.previous);
                    *r = self.previous as f32;
                }

                arr![[f32; BLOCK_SIZE]; r]
            }

            fn set_sample_rate(&mut self, rate: f32) {
                self.per_sample = 1.0 / rate as f64;
            }
        }
    };
}

oscillator! {
    SawWave,
    U0,
    |phase, delta, _input, _i, _previous| {
        let r = 2.0 * phase - 1.0;
        r - poly_blep(phase, delta)
    }
}

oscillator! {
    TriangleWave,
    U0,
    |phase, delta, _input, _i, previous| {
        let mut r = if phase < 0.5 { 1.0 } else { -1.0 };
        r += poly_blep(phase, delta);
        r -= poly_blep((phase + 0.5).fract(), delta);
        // FIXME: The amplitude of this oscillator is much lower than the others. Why?
        delta * r + (1.0 - delta) * previous
    }
}

oscillator! {
    SquareWave,
    U1,
    |phase, delta, input:&Ports<U2>, i, _previous| {
        let width = input[1][i] as f64;
        let mut r = if phase < width { 1.0 } else { -1.0 };
        r += poly_blep(phase, delta);
        r - poly_blep((phase + (1.0-width)).fract(), delta)
    }
}

oscillator! {
    SineWave,
    U0,
    |phase:f64, _delta, _input, _i, _previous| {
        (TAU as f64 * phase).sin()
    }
}

oscillator! {
    PositiveSineWave,
    U0,
    |phase:f64, _delta, _input, _i, _previous| {
        (TAU as f64 * phase).sin() * 0.5 + 0.5
    }
}
