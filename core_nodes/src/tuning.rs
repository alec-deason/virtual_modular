use generic_array::{
    arr,
    sequence::{Concat, Split},
    typenum::*,
    ArrayLength,
};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;
#[derive(Clone)]
pub struct InstrumentTuner {
    rate: usize,
    buffer: [f32; 1024*10],
    corrections: Vec<(f32, f32)>,
    sweep_frequencies: Vec<(f32, f32)>,
    fill_idx: usize,
    test_idx: usize,
    sweep_idx: usize,
    sweep_clock: f32,
}

impl Default for InstrumentTuner {
    fn default() -> Self {
        let mut sweep_frequencies = vec![(160.0, 0.0)];
        while sweep_frequencies[sweep_frequencies.len()-1].0 < 2000.0 {
            let new_freq = sweep_frequencies[sweep_frequencies.len()-1].0 + 100.123;
            sweep_frequencies.push((new_freq, 0.0));
        }
        let mut corrections = vec![(0.0, 0.0), (-100.0, 0.0)];
        while corrections[corrections.len()-1].0 < 100.0 {
            let new_freq = corrections[corrections.len()-1].0 + 1.0;
            corrections.push((new_freq, 0.0));
        }
        Self {
            rate: 44100,
            buffer: [0.0; 1024*10],
            fill_idx: 0,
            sweep_frequencies,
            sweep_idx: 0,
            test_idx: 0,
            corrections,
            sweep_clock: 1.0,
        }
    }
}

impl Node for InstrumentTuner {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            self.sweep_clock -= 1.0/self.rate as f32;
            if self.sweep_clock <= 0.0 {
                if self.fill_idx == self.buffer.len() {
                    const POWER_THRESHOLD : f32 = 5.0;
                    const CLARITY_THRESHOLD : f32 = 0.7;
                    let mut detector = McLeodDetector::new(1024*10, (1024*10)/2);

                    let pitch = detector.get_pitch(&self.buffer, self.rate, POWER_THRESHOLD, CLARITY_THRESHOLD).unwrap();
                    self.fill_idx = 0;
                    let error = (pitch.frequency - self.sweep_frequencies[self.sweep_idx].0).abs();
                    self.corrections[self.test_idx].1 = error;
                    self.sweep_clock = 0.1;
                    if self.test_idx == self.corrections.len() -1 {
                        let best = self.corrections.iter().min_by_key(|(_, e)| (e * 10000.0) as i32).unwrap_or(&(0.0, 0.0)).0;
                        self.sweep_frequencies[self.sweep_idx].1 = best;
                        self.sweep_idx += 1;
                        self.test_idx = 0;
                    } else {
                        self.test_idx += 1;
                    }
                    if self.sweep_idx == self.sweep_frequencies.len() -1 && self.test_idx == self.corrections.len() - 1 {
                        println!("{:?}", self.sweep_frequencies);
                        self.sweep_idx = 0;
                    }
                } else {
                    self.buffer[self.fill_idx] = input[0][i];
                    self.fill_idx += 1;
                }
            }
            r[i] = self.sweep_frequencies[self.sweep_idx].0 + self.corrections[self.test_idx].0;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.rate = rate as usize;
    }
}
