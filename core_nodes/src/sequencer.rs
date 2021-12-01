use std::{
    hash::{Hash, Hasher},
    collections::hash_map::DefaultHasher
};
use generic_array::{
    arr,
    typenum::*,
};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Debug)]
pub enum Subsequence {
    Item(f32, f32),
    Rest(f32),
    Tuplet(Vec<Subsequence>, usize),
    Iter(Vec<Subsequence>, usize),
    Choice(Vec<Subsequence>, usize),
    ClockMultiplier(Box<Subsequence>, f32),
}

impl Default for Subsequence {
    fn default() -> Self {
        Subsequence::Item(0.0, 0.0)
    }
}

impl Subsequence {
    pub fn to_rust(&self) -> String {
        match self {
            Subsequence::Item(a, b) => format!("Subsequence::Item({:.4}, {:.4})", a, b),
            Subsequence::Rest(a) => format!("Subsequence::Rest({:.4})", a),
            Subsequence::Tuplet(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Tuplet(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::Iter(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Iter(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::Choice(seq, c) => {
                let seq: Vec<_> = seq.iter().map(|s| s.to_rust()).collect();
                format!("Subsequence::Choice(vec![{}], {})", seq.join(","), c)
            }
            Subsequence::ClockMultiplier(seq, m) => {
                format!(
                    "Subsequence::ClockMultiplier(Box::new({}), {:.4})",
                    seq.to_rust(),
                    m
                )
            }
        }
    }

    fn reset(&mut self) {
        match self {
            Subsequence::Rest(clock) |
            Subsequence::Item(_, clock) => *clock = 0.0,
            Subsequence::Tuplet(sub_sequence, sub_idx) |
            Subsequence::Iter(sub_sequence, sub_idx) |
            Subsequence::Choice(sub_sequence, sub_idx) => {
                sub_sequence.iter_mut().for_each(|s| s.reset());
                *sub_idx = 0;
            }
            Subsequence::ClockMultiplier(sub_sequence, mul) => sub_sequence.reset(),
        }
    }

    fn current(
        &mut self,
        pulse: bool,
        clock_division: f32,
    ) -> (Option<f32>, bool, bool, bool, bool, f32) {
        match self {
            Subsequence::Rest(clock) => {
                if pulse {
                    *clock += 1.0;
                }
                let do_tick = if *clock >= clock_division {
                    *clock = 0.0;
                    true
                } else {
                    false
                };
                (None, do_tick, false, false, false, clock_division)
            }
            Subsequence::Item(v, clock) => {
                if pulse {
                    *clock += 1.0;
                }
                let (do_tick, do_trigger, gate) = if *clock >= clock_division {
                    *clock = 0.0;
                    (true, true, true)
                } else {
                    (false, false, true)
                };
                (Some(*v), do_tick, do_trigger, gate, false, clock_division)
            }
            Subsequence::Tuplet(sub_sequence, sub_idx) => {
                let clock_division = clock_division / sub_sequence.len() as f32;
                let (v, do_tick, do_trigger, gate, _, len) =
                    sub_sequence[*sub_idx].current(pulse, clock_division);
                let do_tick = if do_tick {
                    *sub_idx += 1;
                    if *sub_idx >= sub_sequence.len() {
                        *sub_idx = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                (v, do_tick, do_trigger, gate, do_tick, len)
            }
            Subsequence::Iter(sub_sequence, sub_idx) => {
                let (v, do_tick, do_trigger, gate, _, len) =
                    sub_sequence[*sub_idx].current(pulse, clock_division);
                let (do_tick, end_of_cycle) = if do_tick {
                    *sub_idx += 1;
                    let end_of_cycle = if *sub_idx >= sub_sequence.len() {
                        *sub_idx = 0;
                        true
                    } else {
                        false
                    };
                    (true, end_of_cycle)
                } else {
                    (false, false)
                };
                (v, do_tick, do_trigger, gate, end_of_cycle, len)
            }
            Subsequence::Choice(sub_sequence, sub_idx) => {
                let (v, do_tick, do_trigger, gate, _, len) =
                    sub_sequence[*sub_idx].current(pulse, clock_division);
                let do_tick = if do_tick {
                    *sub_idx = thread_rng().gen_range(0..sub_sequence.len());
                    true
                } else {
                    false
                };
                (v, do_tick, do_trigger, gate, false, len)
            }
            Subsequence::ClockMultiplier(sub_sequence, mul) => {
                let mut r = sub_sequence.current(pulse, clock_division * *mul);
                r.5 *= *mul;
                r
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Subsequence::Item(..)
            | Subsequence::Iter(..)
            | Subsequence::Choice(..)
            | Subsequence::Rest(..) => 1,
            Subsequence::Tuplet(sub_sequence, ..) => sub_sequence.len(),
            Subsequence::ClockMultiplier(sub_sequence, mul) => sub_sequence.len(),
        }
    }
}

#[derive(Clone, Default)]
pub struct PatternSequencer {
    sequence: Subsequence,
    parameter_hash: u64,
    per_sample: f32,
    triggered: bool,
    previous_value: f32,
}

impl PatternSequencer {
    pub fn new(sequence: Subsequence) -> Self {
        Self {
            sequence,
            parameter_hash: 0,
            per_sample: 0.0,
            triggered: false,
            previous_value: 0.0,
        }
    }

    fn tick(&mut self, trigger: f32) -> [f32; 5] {
        let mut r_trig = 0.0;
        let mut r_gate = 0.0;
        let mut r_eoc = 0.0;
        let mut r_value = 0.0;
        let mut r_len = 0.0;

        let mut pulse = false;
        if trigger > 0.5 {
            if !self.triggered {
                self.triggered = true;
                pulse = true;
            }
        } else {
            self.triggered = false;
        }

        let (v, _, t, g, eoc, len) = self
            .sequence
            .current(pulse, 24.0 * self.sequence.len() as f32);
        if g {
            r_gate = 1.0;
        }
        if t {
            r_trig = 1.0;
            r_gate = 0.0;
        }
        if eoc {
            r_eoc = 1.0;
        }
        self.previous_value = v.unwrap_or(self.previous_value);
        r_value = self.previous_value;
        r_len = len / 24.0;
        [r_trig, r_gate, r_eoc, r_value, r_len]
    }

    fn reset(&mut self) {
        self.sequence.reset();
    }
}

impl Node for PatternSequencer {
    type Input = U1;
    type Output = U5;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r_trig = [0.0f32; BLOCK_SIZE];
        let mut r_gate = [0.0f32; BLOCK_SIZE];
        let mut r_eoc = [0.0f32; BLOCK_SIZE];
        let mut r_value = [0.0f32; BLOCK_SIZE];
        let mut r_len = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let trigger = input[0][i];
            let result = self.tick(trigger);
            r_trig[i] = result[0];
            r_gate[i] = result[1];
            r_eoc[i] = result[2];
            r_value[i] = result[3];
            r_len[i] = result[4];
        }
        arr![[f32; BLOCK_SIZE]; r_value, r_trig, r_gate, r_eoc, r_len]
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        let mut s = DefaultHasher::new();
        parameters.hash(&mut s);
        let h = s.finish();
        if h != self.parameter_hash {
            self.parameter_hash = h;
            let seq = parse_sequence(parameters)?;
            self.sequence = seq;
        }
        Ok(())
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 8.0 / rate;
    }
}

#[derive(Clone, Default)]
pub struct BurstSequencer {
    seq: PatternSequencer,
    burst_triggered: bool,
    firing: bool,
}

impl BurstSequencer {
    pub fn new(sequence: Subsequence) -> Self {
        Self {
            seq: PatternSequencer::new(sequence),
            burst_triggered: false,
            firing: false,
        }
    }
}

impl Node for BurstSequencer {
    type Input = U2;
    type Output = U5;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r_trig = [0.0f32; BLOCK_SIZE];
        let mut r_gate = [0.0f32; BLOCK_SIZE];
        let mut r_eoc = [0.0f32; BLOCK_SIZE];
        let mut r_value = [0.0f32; BLOCK_SIZE];
        let mut r_len = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            if input[1][i] > 0.5 {
                if !self.burst_triggered {
                    self.burst_triggered = true;
                    self.firing = true;
                    self.seq.reset();
                }
            } else {
                self.burst_triggered = false;
            }
            if self.firing {
                let trigger = input[0][i];
                let result = self.seq.tick(trigger);
                r_trig[i] = result[0];
                r_gate[i] = result[1];
                r_eoc[i] = result[2];
                if result[2] == 1.0 {
                    self.firing = false;
                }
                r_value[i] = result[3];
                r_len[i] = result[4];
            }
        }
        arr![[f32; BLOCK_SIZE]; r_value, r_trig, r_gate, r_eoc, r_len]
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.seq.set_static_parameters(parameters)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.seq.set_sample_rate(rate);
    }
}

fn parse_sequence(data: &str) -> Result<Subsequence, String> {
    use pom::parser::Parser;
    use pom::parser::*;
    use std::str::{self, FromStr};
    fn modified_event<'a>() -> Parser<'a, u8, Subsequence> {
        (call(raw_event) + one_of(b"*/!") + number()).convert(|((e, o), n)| match o {
            b'*' => Ok(Subsequence::ClockMultiplier(Box::new(e), 1.0 / n)),
            b'/' => Ok(Subsequence::ClockMultiplier(Box::new(e), n)),
            b'!' => Ok(Subsequence::ClockMultiplier(
                Box::new(Subsequence::Tuplet(
                    (0..n.max(1.0) as usize).map(|_| e.clone()).collect(),
                    0,
                )),
                1.0 / n,
            )),
            _ => Err(()),
        })
    }

    fn raw_event<'a>() -> Parser<'a, u8, Subsequence> {
        call(bracketed_pattern)
            | sym(b'~').map(|_| Subsequence::Rest(0.0))
            | number().map(|n| Subsequence::Item(n, 0.0))
    }

    fn event<'a>() -> Parser<'a, u8, Subsequence> {
        call(modified_event) | call(raw_event)
    }

    fn sequence<'a>() -> Parser<'a, u8, Vec<Subsequence>> {
        list(call(event), sym(b' ').repeat(1..))
    }

    fn repeat<'a>() -> Parser<'a, u8, Subsequence> {
        (sym(b'[') * whitespace() * call(sequence) - whitespace() - sym(b']')).convert(|s| {
            if s.is_empty() {
                Err(())
            } else {
                Ok(Subsequence::Tuplet(s, 0))
            }
        })
    }

    fn cycle<'a>() -> Parser<'a, u8, Subsequence> {
        (sym(b'<') * whitespace() * call(sequence) - whitespace() - sym(b'>')).convert(|s| {
            if s.is_empty() {
                Err(())
            } else {
                Ok(Subsequence::Iter(s, 0))
            }
        })
    }

    fn choice<'a>() -> Parser<'a, u8, Subsequence> {
        (sym(b'|') * whitespace() * call(sequence) - whitespace() - sym(b'|')).convert(|s| {
            if s.is_empty() {
                Err(())
            } else {
                Ok(Subsequence::Choice(s, 0))
            }
        })
    }

    fn bracketed_pattern<'a>() -> Parser<'a, u8, Subsequence> {
        call(repeat) | call(cycle) | call(choice)
    }

    fn whitespace<'a>() -> Parser<'a, u8, Vec<u8>> {
        sym(b' ').repeat(0..).name("whitespace")
    }

    fn number<'a>() -> Parser<'a, u8, f32> {
        let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
        let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
        let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
        let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
        number
            .collect()
            .convert(str::from_utf8)
            .convert(f32::from_str)
            .name("number")
    }

    let parsed = (whitespace() * sequence() - whitespace() - end())
        .parse(data.as_bytes())
        .map(|s| Subsequence::Tuplet(s, 0))
        .map_err(|e| format!("{:?}", e));
    parsed
}
