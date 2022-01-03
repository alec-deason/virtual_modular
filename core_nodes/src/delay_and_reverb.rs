use crate::filter::AllPass;
use generic_array::{arr, typenum::*};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};
use crate::{LPComb, OnePole, utils::make_coprime};



#[cfg(test)]
mod make_coprime_tests {
    use super::*;

    #[test]
    fn basic() {
        let mut set = vec![1, 2, 3, 4];
        make_coprime(&mut set);
        assert_eq!(set, vec![2, 3, 5, 7]);
    }
}

#[derive(Clone)]
enum ReverbBlock {
    Combs(Vec<(f64, f64, LPComb)>),
    AllPasses(Vec<(f64, f64, AllPass)>),
    Parallel(Vec<ReverbBlock>),
    Series(Vec<ReverbBlock>),
}

impl ReverbBlock {
    fn tick(&mut self, signal: f64) -> f64 {
        match self {
            ReverbBlock::Combs(combs) => {
                let mut out = 0.0;
                for (_, _, comb) in combs{
                    out += comb.tick(signal);
                }
                out
            },
            ReverbBlock::AllPasses(all_passes) => {
                let mut out = signal;
                for (_, _, all_pass) in all_passes {
                    out = all_pass.tick(out);
                }
                out
            },
            ReverbBlock::Parallel(blocks) => {
                let mut out = 0.0;
                for block in blocks {
                    out += block.tick(signal);
                }
                out
            },
            ReverbBlock::Series(blocks) => {
                let mut out = signal;
                for block in blocks {
                    out = block.tick(out);
                }
                out
            },
        }
    }

    fn set_sample_rate(&mut self, delay_mul: f64, gain_mul: f64, rate: f64) {
        match self {
            ReverbBlock::Combs(combs) => {
                let mut delays:Vec<_> = combs.iter().map(|(d, _, _)| (d*rate*delay_mul) as u32).collect();
                make_coprime(&mut delays);
                for (d, (_, gain, comb)) in delays.iter().zip(combs) {
                    comb.set_delay(*d as f64);
                    comb.gain = *gain * gain_mul;
                }
            }
            ReverbBlock::AllPasses(all_passes) => {
                let mut delays:Vec<_> = all_passes.iter().map(|(d, _, _)| (d*rate*delay_mul) as u32).collect();
                make_coprime(&mut delays);
                for (d, (_, gain, ap)) in delays.iter().zip(all_passes) {
                    ap.set_delay(*d as f64);
                    ap.gain = *gain * gain_mul;
                }
            }
            ReverbBlock::Parallel(blocks) | ReverbBlock::Series(blocks) => {
                for block in blocks {
                    block.set_sample_rate(delay_mul, gain_mul, rate);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Reverb {
    blocks: ReverbBlock,
    delay_mul: f64,
    gain_mul: f64,
    rate: f64,
}

impl Reverb {
    // Based on: https://ccrma.stanford.edu/~jos/pasp/Example_Schroeder_Reverberators.html
    pub fn satrev() -> Self {
        Self {
            blocks: ReverbBlock::Series(vec![
                ReverbBlock::Combs([(0.805, 0.03604), (0.827, 0.03112), (0.783, 0.04044), (0.764, 0.04492)].iter().map(|(g, d)| (*d, *g, LPComb::new(*g, 1.0, d*48000.0))).collect()),
                ReverbBlock::AllPasses([(0.7, 0.005), (0.7, 0.00168), (0.7, 0.0048)].iter().map(|(g, d)| (*d, *g, AllPass::new(*g, d*48000.0))).collect())
            ]),
            delay_mul: 1.0,
            gain_mul: 1.0,
            rate: 48000.0,
        }
    }

    pub fn poop() -> Self {
        Self {
            blocks: ReverbBlock::Series(vec![
                ReverbBlock::Combs([(0.805, 0.03604), (0.827, 0.03112), (0.783, 0.04044), (0.764, 0.04492), (0.7, 0.08), (0.9, 0.03), (0.44, 0.1), (0.3, 0.08)].iter().map(|(g, d)| (*d, *g, LPComb::new(*g, 1.0, d*48000.0))).collect()),
                ReverbBlock::AllPasses([(0.6, 0.005), (0.778, 0.00168), (0.8, 0.0048)].iter().map(|(g, d)| (*d, *g, AllPass::new(*g, d*48000.0))).collect())
            ]),
            delay_mul: 1.0,
            gain_mul: 1.0,
            rate: 48000.0,
        }
    }

    // Based on: https://ccrma.stanford.edu/~jos/pasp/Freeverb.html
    pub fn freeverb() -> Self {
        Self {
            blocks: ReverbBlock::Series(vec![
                ReverbBlock::Combs([0.0353, 0.0366, 0.0338, 0.0322, 0.0289, 0.0307, 0.0269, 0.0253].iter().map(|d| (*d, 0.84, LPComb::new(0.84, 0.2, d*48000.0))).collect()),
                ReverbBlock::AllPasses([0.0051, 0.0126, 0.01, 0.0077].iter().map(|d| (*d, 0.5, AllPass::new(0.5, d*48000.0))).collect())
            ]),
            delay_mul: 1.0,
            gain_mul: 1.0,
            rate: 48000.0,
        }
    }

    fn set_params(&mut self, delay_mul: f64, gain_mul: f64) {
        if self.gain_mul != gain_mul || self.delay_mul != delay_mul {
            self.gain_mul = gain_mul;
            self.delay_mul = delay_mul;
            self.blocks.set_sample_rate(self.delay_mul, self.gain_mul, self.rate);
        }
    }
}

impl Node for Reverb {
    type Input = U3;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len, gain, sig) = (input[0], input[1], input[2]);
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in r[0].iter_mut().enumerate() {
            self.set_params(len[i] as f64, gain[i] as f64);
            let sig = sig[i] as f64;
            *r = self.blocks.tick(sig) as f32;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as f64;
        self.blocks.set_sample_rate(self.delay_mul, self.gain_mul, self.rate);
    }
}

#[derive(Clone, Default)]
pub struct ModableDelay {
    line: DelayLine,
    rate: f32,
}
impl Node for ModableDelay {
    type Input = U3;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len, sig, feedback) = (input[0], input[1], input[2]);
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            let len = len[i].max(0.000001);
            self.line.set_delay((len * self.rate) as f64);
            *r = self.line.current() as f32;
            let sig = sig[i] + *r * feedback[i];
            self.line.tick(sig as f64);
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate;
    }
}

#[derive(Clone, Default)]
pub struct ModablePingPong {
    left: DelayLine,
    right: DelayLine,
    rate: f32,
}
impl Node for ModablePingPong {
    type Input = U4;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let (len, sig_l, sig_r, feedback) = (input[0], input[1], input[2], input[3]);
        let mut left = [0.0; BLOCK_SIZE];
        let mut right = [0.0; BLOCK_SIZE];
        for (i, (l, r)) in left.iter_mut().zip(&mut right).enumerate() {
            let len = len[i].max(0.000001) / 2.0;
            self.left.set_delay((len * self.rate) as f64);
            self.right.set_delay((len * self.rate) as f64);
            *l = self.left.current() as f32;
            *r = self.right.current() as f32;
            let feedback = feedback[i] as f64;
            self.left.tick((sig_l[i] as f64 + *r as f64) * feedback);
            self.right.tick((sig_r[i] as f64 + *l as f64) * feedback);
        }
        arr![[f32; BLOCK_SIZE]; left, right]
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

    pub fn current(&self) -> f64 {
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
        assert_eq!(l.current(), 0.0);
        for _ in 0..9 {
            assert_eq!(l.current(), 0.0);
            l.tick(0.0);
        }
        assert_eq!(l.current(), 1.0);
        l.tick(0.0);
        for _ in 0..20 {
            assert_eq!(l.current(), 0.0);
            l.tick(0.0);
        }
    }

    #[test]
    fn small_interp() {
        let mut l = DelayLine::default();
        l.set_delay(10.1);
        assert_eq!(l.current(), 0.0);
        l.tick(1.0);
        for _ in 0..9 {
            assert_eq!(l.current(), 0.0);
            l.tick(0.0);
        }
        assert!((1.0 - l.current()).abs() < 0.2);
        l.tick(0.0);
        for _ in 0..9 {
            println!("{:?}", l);
            assert!((0.0 - l.current()).abs() < 0.2);
            l.tick(0.0);
        }
        assert_eq!(l.current(), 0.0);
    }
}

#[derive(Clone, Default)]
pub struct BlockDelayLine {
    lines: Vec<f64>,
    pub width: usize,
    pub len: f64,
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

    pub fn current(&mut self) -> &[f64] {
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
        assert_eq!(l.current(), &[0.0; 10]);
        for _ in 0..9 {
            assert_eq!(l.current(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert_eq!(l.current(), &[1.0; 10]);
        l.tick();
        for _ in 0..20 {
            assert_eq!(l.current(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
    }

    #[test]
    fn small_interp() {
        let mut l = BlockDelayLine::new(10, 10.1);
        l.input_buffer().copy_from_slice(&[1.0; 10]);
        println!("{} {} {}", l.len, l.in_index, l.out_index);
        assert_eq!(l.current(), &[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert_eq!(l.current(), &[0.0; 10]);
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        println!("{:?} {} {} {}", l.lines, l.len, l.in_index, l.out_index);
        assert!(l.current().iter().all(|v| (1.0 - *v).abs() < 0.2));
        l.input_buffer().copy_from_slice(&[0.0; 10]);
        l.tick();
        for _ in 0..9 {
            assert!(l.current().iter().all(|v| (0.0 - *v).abs() < 0.2));
            l.input_buffer().copy_from_slice(&[0.0; 10]);
            l.tick();
        }
        assert_eq!(l.current(), &[0.0; 10]);
    }
}

// Based on: https://gist.github.com/geraintluff/c55f9482dce0db6e6a1f4509129e9a2a
fn hadamard(data: &mut [f64]) {
    if data.len() <= 1 {
        return
    }

    let h_size = data.len()/2;

    let (a,b) = data.split_at_mut(h_size);
    hadamard(a);
    hadamard(b);

    for i in 0..h_size {
        let a = data[i];
        let b = data[i + h_size];
        data[i] = a + b;
        data[i + h_size] = a - b;
    }
}

// Based on: https://gist.github.com/geraintluff/663e42e2519465e8b94df47793076f23
fn householder(data: &mut [f64]) {
	let factor = -2.0/data.len() as f64;

	let mut sum = data.iter().sum::<f64>();

	sum *= factor;

	data.iter_mut().for_each(|v| *v += sum);
}

#[derive(Clone)]
pub struct Diffusor {
    delays: Vec<(f64, DelayLine)>,
    shuffles: Vec<(usize, f64)>,
}

impl Diffusor {
    pub fn new<R: Rng+Sized>(len: f64, rng: &mut R) -> Self {
        let mut idxs: Vec<usize> = (0..8).collect();
        idxs.shuffle(rng);

        let delta = len / 8.0;
        Self {
            delays: (0..8).map(|i| ((i+1) as f64* delta, DelayLine::default())).collect(),
            shuffles: idxs.into_iter().map(|idx| (idx, if rng.gen::<f64>() > 0.5 { 1.0 } else { -1.0 })).collect(),
        }
    }

    fn tick(&mut self, input: &[f64; 8]) -> [f64; 8] {
        let mut r = [0.0; 8];
        for ((r, input), (_, delay)) in r.iter_mut().zip(input).zip(&mut self.delays) {
            *r = delay.current();
            delay.tick(*input);
        }

        for (a, (b, m)) in self.shuffles.iter().enumerate() {
            r.swap(a, *b);
            r[a] *= m;
        }

        hadamard(&mut r);
        let norm = (1.0/r.len() as f64).sqrt();
        r.iter_mut().for_each(|v| *v *= norm );

        r
    }
}

impl Node for Diffusor {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = <Ports<Self::Output> >::default();
        for (i, r) in input[0].iter().zip(&mut r[0]) {
            let diffused = self.tick(&[*i as f64; 8]);
            *r = diffused.iter().sum::<f64>() as f32;
            *r /= 8.0;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for (len, d) in &mut self.delays {
            d.set_delay(*len*rate as f64);
        }
    }

}

// Based on: https://signalsmith-audio.co.uk/writing/2021/lets-write-a-reverb/
#[derive(Clone)]
pub struct Reverb2 {
    diffusors: Vec<Diffusor>,
    delays: Vec<(f64, DelayLine)>,
}

impl Default for Reverb2 {
    fn default() -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        Self {
            diffusors: (0..4).map(|i| Diffusor::new(0.03 * i as f64, &mut rng)).collect(),
            delays: (0..8).map(|_| (rng.gen_range(0.1..0.2), DelayLine::default())).collect(),
        }
    }
}

impl Node for Reverb2 {
    type Input = U1;
    type Output = U2;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = < Ports<Self::Output> >::default();
        let (left, right) = r.split_at_mut(1);
        for ((i, left), right) in input[0].iter().zip(&mut left[0]).zip(&mut right[0]) {
            let mut i = [*i as f64; 8];
            for d in &mut self.diffusors {
                i = d.tick(&i);
            }

            let mut j = [0.0 as f64; 8];
            for (j, (_, d)) in j.iter_mut().zip(&self.delays) {
                let v = d.current();
                *j = v * 0.84;
            }
            householder(&mut j);

            for ((i, (_, d)), j) in i.iter().zip(&mut self.delays).zip(&j) {
                d.tick(*i + *j);
            }

            *left = j[0] as f32;
            *right = j[1] as f32;
        }
        r
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for d in &mut self.diffusors {
            d.set_sample_rate(rate);
        }
        for (d, l) in &mut self.delays {
            l.set_delay(*d * rate as f64);
        }
    }
}
