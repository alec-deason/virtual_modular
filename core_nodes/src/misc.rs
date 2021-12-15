use generic_array::{arr, typenum::*};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Copy, Clone)]
pub struct Log;

impl Node for Log {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        println!("{:?}", input[0]);
        input
    }
}

#[derive(Copy, Clone, Default)]
pub struct LogTrigger {
    triggered: bool
}

impl Node for LogTrigger {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        for r in &input[0] {
            if *r > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    println!("Trigger");
                }
            } else {
                self.triggered = false;
            }
        }
        input
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Interpolation {
    Constant { value: f32, duration: f32 },
    Linear { start: f32, end: f32, duration: f32 },
}
impl Interpolation {
    pub fn to_rust(&self) -> String {
        match self {
            Interpolation::Constant { value, duration } => {
                format!(
                    "Interpolation::Constant {{ value: {:.4}, duration: {:.4} }}",
                    value, duration
                )
            }
            Interpolation::Linear {
                start,
                end,
                duration,
            } => {
                format!(
                    "Interpolation::Linear {{ start: {:.4}, end: {:.4}, duration: {:.4} }}",
                    start, end, duration
                )
            }
        }
    }

    fn evaluate(&self, time: f32) -> (f32, bool) {
        match self {
            Interpolation::Linear {
                start,
                end,
                duration,
            } => {
                if time >= *duration {
                    (*end, false)
                } else {
                    let t = (duration - time) / duration;
                    (start * t + end * (1.0 - t), true)
                }
            }
            Interpolation::Constant { value, duration } => (*value, time <= *duration),
        }
    }
}

#[derive(Clone)]
pub struct Automation {
    steps: Vec<Interpolation>,
    step: usize,
    time: f64,
    pub do_loop: bool,
    per_sample: f64,
}
impl Automation {
    pub fn new(steps: &[Interpolation]) -> Self {
        Self {
            steps: steps.to_vec(),
            step: 0,
            time: 0.0,
            do_loop: true,
            per_sample: 0.0,
        }
    }
}

impl Node for Automation {
    type Input = U0;
    type Output = U1;

    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        let step = &self.steps[self.step];
        let (v, running) = step.evaluate(self.time as f32);
        if !running {
            self.time = 0.0;
            self.step = (self.step + 1) % self.steps.len();
        }
        self.time += self.per_sample;
        arr![[f32; BLOCK_SIZE]; [v; BLOCK_SIZE]]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = BLOCK_SIZE as f64 / rate as f64;
    }
}

#[derive(Copy, Clone)]
pub struct RMS {
    mean_squared: f64,
    decay: f64
}

impl Default for RMS {
    fn default() -> Self {
        Self {
            mean_squared: 0.0,
            decay: 0.999,
        }
    }
}

impl RMS {
    pub fn tick(&mut self, sample: f64) -> f64 {
        self.mean_squared = self.mean_squared * self.decay + (1.0-self.decay) * sample.powi(2);
        self.mean_squared.sqrt()
    }
}

impl Node for RMS {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0f32; BLOCK_SIZE];
        for (sample, r) in input[0].iter().zip(&mut r) {
            *r = self.tick(*sample as f64) as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Copy, Clone, Default)]
pub struct Compressor {
    rms: RMS,
    triggered: bool,
    current: f64,
    time: f64,
    per_sample: f64,
}

impl Compressor {
    pub fn tick(&mut self, attack: f64, decay: f64, threshold: f64, gain: f64, sample: f64) -> f64 {
        let rms = self.rms.tick(sample);
        self.time += self.per_sample;
        if rms > threshold {
            if !self.triggered {
                self.triggered = true;
                self.time = 0.0;
            }
        } else {
            if self.triggered {
                self.triggered = false;
                self.time = (1.0-self.current) * attack;
            }
        }
        let r = if self.triggered {
            if self.time < attack {
                self.time / attack
            } else {
                1.0
            }
        } else {
            if self.time < decay {
                1.0 - self.time / decay
            } else {
                0.0
            }
        };
        self.current = self.current * 0.1 + r * 0.9;
        10.0f64.powf((self.current*gain)/10.0)
    }
}

impl Node for Compressor {
    type Input = U5;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let attack = input[0];
        let decay = input[1];
        let threshold = input[2];
        let gain = input[3];
        let sample = input[4];

        let mut r = [0.0f32; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            *r = self.tick(attack[i] as f64, decay[i] as f64, threshold[i] as f64, gain[i] as f64, sample[i] as f64) as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0/rate as f64;
    }
}

