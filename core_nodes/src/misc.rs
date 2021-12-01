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
