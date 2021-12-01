use generic_array::{
    arr,
    typenum::*,
};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Default)]
pub struct TapsAndStrikes {
    current_freq: f32,
    old_freq: f32,
    current_freq_modified: f32,
    crossover:f32,
    can_roll: bool,
    triggered: bool,
    per_sample: f32,
}

impl Node for TapsAndStrikes {
    type Input = U5;
    type Output = U2;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let trigger = input[1];
        let prob = input[2];
        let roll_prob = input[3];
        let attack = input[4];

        let mut r_freq = [0.0f32; BLOCK_SIZE];
        let mut r_trigger = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let mut could_hit = false;
            let mut must_hit = false;
            let new_freq = freq[i];
            if trigger[i] > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    could_hit = true;
                    self.can_roll = true;
                    if new_freq == self.current_freq {
                        must_hit = true;
                    }
                    self.old_freq = self.current_freq;
                    self.current_freq = new_freq;
                }
            } else {
                self.triggered = false;
            }
            r_freq[i] = self.current_freq_modified;
            self.crossover = self.crossover - self.per_sample/attack[i];
            if self.crossover <= 0.0 {
                self.current_freq_modified = self.current_freq;
            }
            let mut prob = prob[i];
            if self.can_roll && self.crossover <= -1.0 {
                could_hit = true;
                prob = roll_prob[i];
                self.can_roll = false;
            }
            if must_hit || (could_hit && thread_rng().gen::<f32>() < prob) {
                if self.current_freq > self.old_freq {
                    self.current_freq_modified = self.current_freq * 1.05;
                } else {
                    self.current_freq_modified = self.current_freq * 0.95;
                }
                self.crossover = 1.0;
                r_trigger[i] = 1.0;
            }
        }
        arr![[f32; BLOCK_SIZE]; r_freq, r_trigger]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate;
    }
}
