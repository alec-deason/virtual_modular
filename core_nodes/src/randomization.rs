use generic_array::{arr, typenum::*};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Default)]
pub struct BernoulliGate {
    trigger: bool,
    active_gate: bool,
}

impl Node for BernoulliGate {
    type Input = U2;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (prob, sig) = (input[0], input[1]);
        let mut r = [[0.0; BLOCK_SIZE], [0.0; BLOCK_SIZE]];
        for i in 0..BLOCK_SIZE {
            let prob = prob[i];
            let sig = sig[i];
            if sig > 0.5 {
                if !self.trigger {
                    self.trigger = true;
                    self.active_gate = thread_rng().gen::<f32>() < prob;
                }
            } else {
                self.trigger = false;
            }

            if self.active_gate {
                r[0][i] = sig;
            } else {
                r[1][i] = sig;
            }
        }
        arr![[f32; BLOCK_SIZE]; r[0], r[1]]
    }
}

#[derive(Clone, Default)]
pub struct EuclidianPulse {
    pulses: u32,
    len: u32,
    steps: Vec<bool>,
    idx: usize,
    triggered: bool,
}

impl Node for EuclidianPulse {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let pulses = input[0];
        let len = input[1];
        let gate = input[2];
        let mut r = [0.0f32; BLOCK_SIZE];

        for (i, r) in r.iter_mut().enumerate() {
            let pulses = pulses[i] as u32;
            let len = len[i] as u32;
            if pulses != self.pulses || len != self.len {
                make_euclidian_rhythm(pulses, len, &mut self.steps);
                self.pulses = pulses;
                self.len = len;
            }
            let gate = gate[i];
            if gate > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.idx = (self.idx + 1) % self.len as usize;
                    if self.steps[self.idx] {
                        *r = 1.0;
                    }
                }
            } else {
                self.triggered = false;
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

fn make_euclidian_rhythm(pulses: u32, len: u32, steps: &mut Vec<bool>) {
    steps.resize(len as usize, false);
    steps.fill(false);
    let mut bucket = 0;
    for step in steps.iter_mut() {
        bucket += pulses;
        if bucket >= len {
            bucket -= len;
            *step = true;
        }
    }
}

#[derive(Clone)]
pub struct Brownian {
    current: f32,
    triggered: bool,
}

impl Default for Brownian {
    fn default() -> Self {
        Self {
            current: f32::NAN,
            triggered: false,
        }
    }
}

impl Node for Brownian {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (rate, min, max, trig) = (input[0], input[1], input[2], input[3]);

        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let rate = rate[i];
            let min = min[i];
            let max = max[i];
            let trig = trig[i];
            if trig > 0.5 {
                if !self.triggered {
                    self.current += thread_rng().gen_range(-1.0..1.0) * rate;
                    if !self.current.is_finite() {
                        self.current = thread_rng().gen_range(min..max);
                    }
                    self.current = self.current.min(max).max(min);
                    self.triggered = true;
                }
            } else {
                self.triggered = false;
            }
            *r = self.current;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Clone)]
pub struct Markov {
    transitions: Vec<(f32, Vec<(usize, f32)>)>,
    current_state: usize,
    trigger: bool,
}

impl Markov {
    pub fn new(transitions: &[(f32, Vec<(usize, f32)>)]) -> Self {
        Self {
            transitions: transitions.to_vec(),
            current_state: 0,
            trigger: false,
        }
    }

    pub fn major_key_chords() -> Self {
        let transitions = vec![
            (3.0, vec![(1, 1.0)]),
            (6.0, vec![(2, 0.5), (3, 0.5)]),
            (4.0, vec![(4, 0.5), (3, 0.5)]),
            (2.0, vec![(4, 0.5), (5, 0.5)]),
            (7.0, vec![(6, 0.5), (5, 0.5)]),
            (5.0, vec![(6, 1.0)]),
            (
                1.0,
                vec![
                    (0, 1.0 / 7.0),
                    (1, 1.0 / 7.0),
                    (2, 1.0 / 7.0),
                    (3, 1.0 / 7.0),
                    (4, 1.0 / 7.0),
                    (5, 1.0 / 7.0),
                    (6, 1.0 / 7.0),
                ],
            ),
        ];
        Self {
            transitions,
            current_state: 0,
            trigger: false,
        }
    }
}

impl Node for Markov {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            if input[0][i] > 0.5 {
                if !self.trigger {
                    self.trigger = true;
                    let transitions = &self.transitions[self.current_state].1;
                    let mut rng = thread_rng();
                    let new_state = transitions
                        .choose_weighted(&mut rng, |(_, w)| *w)
                        .unwrap()
                        .0;
                    self.current_state = new_state;
                }
            } else {
                self.trigger = false;
            }
            *r = self.transitions[self.current_state].0;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}
