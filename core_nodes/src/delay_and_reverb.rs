use generic_array::{
    arr,
    typenum::*,
};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};
use crate::filter::AllPass;

#[derive(Clone)]
pub struct Reverb {
    delay_l: Vec<f32>,
    delay_r: Vec<f32>,
    allpass_l: AllPass,
    allpass_r: AllPass,
    taps: Vec<(f32, f32)>,
    tap_total: f32,
    idx_l: usize,
    idx_r: usize,
    per_sample: f32,
}
impl Reverb {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(2);
        let taps: Vec<_> = (0..10)
            .map(|_| (rng.gen_range(0.001..1.0), rng.gen::<f32>()))
            .collect();
        let tap_total = taps.iter().map(|(_, t)| t).sum::<f32>();
        Self {
            delay_l: vec![],
            delay_r: vec![],
            allpass_l: AllPass::default(),
            allpass_r: AllPass::default(),
            taps,
            tap_total,
            idx_l: 0,
            idx_r: 0,
            per_sample: 0.0,
        }
    }
}
impl Node for Reverb {
    type Input = U4;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len, gain, left, right) = (input[0], input[1], input[2], input[3]);
        let mut r_l = [0.0; BLOCK_SIZE];
        let mut r_r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let Reverb {
                delay_l,
                delay_r,
                taps,
                tap_total,
                idx_l,
                idx_r,
                per_sample,
                ..
            } = self;
            let len = len[i].max(0.001);
            delay_l.resize((len / *per_sample) as usize, 0.0);
            delay_r.resize((len / *per_sample) as usize, 0.0);
            *idx_l = (*idx_l + 1) % delay_l.len();
            *idx_r = (*idx_r + 1) % delay_r.len();
            for (fti, g) in taps {
                for (dl, idx, r) in [
                    (&mut *delay_l, *idx_l, &mut r_l[i]),
                    (delay_r, *idx_r, &mut r_r[i]),
                ] {
                    let l = 1.0 / (1.0 + *fti * dl.len() as f32 * *per_sample).powi(2);
                    let l = (l * *g * gain[i]) / *tap_total;
                    let mut i = idx as i32 - (*fti * dl.len() as f32) as i32;
                    if i < 0 {
                        i += dl.len() as i32;
                    }
                    let dl = dl[i as usize % dl.len()];
                    *r += dl * l;
                }
            }
            r_l[i] = left[i] - r_l[i];
            r_r[i] = right[i] - r_r[i];
            delay_l[*idx_l] = self.allpass_l.tick(1.0, r_l[i]);
            delay_r[*idx_r] = self.allpass_r.tick(1.0, r_r[i]);
        }
        arr![[f32; BLOCK_SIZE]; r_l, r_r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}

#[derive(Clone)]
pub struct ModableDelay {
    line: DelayLine,
    rate: f32,
}
impl ModableDelay {
    pub fn new() -> Self {
        Self {
            line: DelayLine::default(),
            rate: 0.0,
        }
    }
}
impl Node for ModableDelay {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len, sig) = (input[0], input[1]);
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let len = len[i].max(0.000001);
            self.line.set_delay((len * self.rate) as f64);
            let sig = sig[i];
            self.line.tick(sig as f64);
            r[i] = self.line.next() as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate;
    }
}

#[derive(Clone, Debug)]
pub struct DelayLine {
    line: Vec<f64>,
    in_index: usize,
    out_index: f64,
    delay: f64,
}


impl Default for DelayLine {
    fn default() -> Self {
        let mut s = Self {
            line: vec![0.0; 41],
            in_index: 0,
            out_index: 0.0,
            delay: 1.0,
        };
        s.set_delay(30.0);
        s
    }
}

impl DelayLine {
    pub fn set_delay(&mut self, delay: f64) {
        let delay = delay.max(1.0);
        if self.delay == delay {
            return;
        }

        self.delay = delay;
        if delay as usize > self.line.len() {
            self.line.resize(self.delay as usize * 2, 0.0);
        }

        self.out_index = self.in_index as f64 - delay;
        while self.out_index < 0.0 {
            self.out_index += self.line.len() as f64;
        }
    }

    pub fn add_at(&mut self, amount: f64, position: f64) {
        let mut index = self.in_index as f64 - self.delay * position;
        while index < 0.0 {
            index += self.line.len() as f64;
        }
        self.line[index as usize] += amount;
    }

    pub fn tick(&mut self, input: f64) {
        self.line[self.in_index] = input;
        self.in_index += 1;
        if self.in_index >= self.line.len() {
            self.in_index = 0;
        }
        self.out_index += 1.0;
        if self.out_index >= self.line.len() as f64 {
            self.out_index -= self.line.len() as f64;
        }
    }

    pub fn next(&self) -> f64 {
        let alpha = self.out_index.fract();
        let mut out = self.line[self.out_index as usize] * (1.0 - alpha);
        out += if self.out_index + 1.0 >= self.line.len() as f64 {
            self.line[0]
        } else {
            self.line[self.out_index as usize + 1]
        } * alpha;
        out
    }
}

#[cfg(test)]
mod delay_tests {
    use super::*;

    #[test]
    fn no_interp() {
        let mut l = DelayLine::default();
        l.set_delay(10.0);
        l.tick(1.0);
        assert_eq!(l.next(), 0.0);
        for _ in 0..9 {
            assert_eq!(l.next(), 0.0);
            l.tick(0.0);
        }
        assert_eq!(l.next(), 1.0);
        l.tick(0.0);
        for _ in 0..20 {
            assert_eq!(l.next(), 0.0);
            l.tick(0.0);
        }
    }

    #[test]
    fn small_interp() {
        let mut l = DelayLine::default();
        l.set_delay(10.1);
        assert_eq!(l.next(), 0.0);
        l.tick(1.0);
        for _ in 0..9 {
            assert_eq!(l.next(), 0.0);
            l.tick(0.0);
        }
        assert!((1.0 - l.next()).abs() < 0.2);
        l.tick(0.0);
        for _ in 0..9 {
            println!("{:?}", l);
            assert!((0.0 - l.next()).abs() < 0.2);
            l.tick(0.0);
        }
        assert_eq!(l.next(), 0.0);
    }
}

#[derive(Clone, Default)]
pub struct BlockDelayLine {
    lines: Vec<f64>,
    width: usize,
    len: f64,
    in_index: usize,
    out_index: f64,
    output_buffer: Vec<f64>,
    delay: f64,
}

impl BlockDelayLine {
    pub fn new(width: usize, delay: f64) -> Self {
        let mut s = Self {
            lines: vec![0.0; width],
            len: 1.0,
            output_buffer: vec![0.0; width],
            width,
            in_index: 0,
            out_index: 0.0,
            delay: 1.0,
        };
        s.set_delay(delay);
        s
    }

    pub fn set_delay(&mut self, mut delay: f64) {
        if !delay.is_finite() {
            delay = 1.0;
        }
        delay = delay.max(1.0).min(44100.0 * 1.0);
        if self.delay == delay {
            return;
        }

        self.delay = delay;
        if delay as usize > self.len as usize {
            self.len = (self.delay.ceil() as usize * 2) as f64;
            self.lines.resize(self.len as usize * self.width, 0.0);
        }

        self.out_index = self.in_index as f64 - delay;
        while self.out_index <= 0.0 {
            self.out_index += self.len;
        }
    }

    pub fn input_buffer(&mut self) -> &mut [f64] {
        let i = self.in_index * self.width;
        &mut self.lines[i..i + self.width]
    }

    pub fn tick(&mut self) {
        self.in_index += 1;
        if self.in_index >= self.len as usize {
            self.in_index = 0;
        }
        self.out_index += 1.0;
        if self.out_index >= self.len {
            self.out_index -= self.len;
        }
    }

    pub fn next(&mut self) -> &[f64] {
        let alpha = self.out_index.fract();
        let i = self.out_index as usize * self.width;
        self.output_buffer
            .copy_from_slice(&self.lines[i..i + self.width]);
        let other = if self.out_index + 1.0 >= self.len {
            &self.lines[0..self.width]
        } else {
            &self.lines[i + self.width..i + self.width * 2]
        };
        let alpha_p = 1.0 - alpha;
        self.output_buffer
            .iter_mut()
            .zip(other)
            .for_each(|(b, o)| *b = *b * alpha_p + *o * alpha);
        &self.output_buffer
    }
}

#[cfg(test)]
mod block_delay_tests {
    use super::*;

    #[test]
    fn no_interp() {
        let mut l = BlockDelayLine::new(10, 10.0);
        println!("{} {} {}", l.len, l.in_index, l.out_index);
        l.input_buffer().copy_from_slice(&[1.0; 10]);
        l.tick();
        assert_eq!(l.next(), &[0.0; 10]);
        for _ in 0..9 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert_eq!(l.next(), &[1.0; 10]);
        l.tick();
        for _ in 0..20 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
    }

    #[test]
    fn small_interp() {
        let mut l = BlockDelayLine::new(10, 10.1);
        l.input_buffer().copy_from_slice(&[1.0; 10]);
        println!("{} {} {}", l.len, l.in_index, l.out_index);
        assert_eq!(l.next(), &[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert_eq!(l.next(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert!(l.next().iter().all(|v| (1.0 - *v).abs() < 0.2));
        l.input_buffer().copy_from_slice(&[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert!(l.next().iter().all(|v| (0.0 - *v).abs() < 0.2));
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        assert_eq!(l.next(), &[0.0; 10]);
    }
}
