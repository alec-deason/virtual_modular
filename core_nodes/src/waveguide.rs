use generic_array::{
    arr,
    typenum::*,
};
use rand::prelude::*;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};
use crate::{delay_and_reverb::DelayLine, filter::{OnePole, Biquad}};
#[derive(Clone)]
pub struct BowedString {
    nut_to_bow: DelayLine,
    bow_to_bridge: DelayLine,
    per_sample: f64,
    string_filter: OnePole,
    body_filters: [Biquad; 6],
}

impl Default for BowedString {
    fn default() -> Self {
        Self {
            nut_to_bow: DelayLine::default(),
            bow_to_bridge: DelayLine::default(),
            per_sample: 1.0 / 44100.0,
            string_filter: OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
            body_filters: [
                Biquad::new(1.0, 1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.988, -1.9867, 0.9923),
            ],
        }
    }
}

impl Node for BowedString {
    type Input = U5;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let bow_velocity = input[1];
        let bow_force = input[2];
        let bow_position = input[3];
        let base_freq = input[4];
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let freq = freq[i] as f64;
            let freq = 0.03065048 + 1.00002*freq + 0.00004114233*freq.powi(2);
            let bow_velocity = bow_velocity[i] as f64;
            let bow_force = bow_force[i] as f64;
            let base_freq = base_freq[i] as f64;
            let total_l = 1.0/(base_freq * self.per_sample);
            let desired_l = 1.0/(freq.max(base_freq) * self.per_sample);
            let bow_position = ((bow_position[i] as f64 + 1.0) / 2.0).max(0.01).min(0.99);

            let bow_nut_l = total_l * (1.0 - bow_position) - (total_l-desired_l);
            let bow_bridge_l = total_l * bow_position;

            self.nut_to_bow.set_delay(bow_nut_l);
            self.bow_to_bridge.set_delay(bow_bridge_l);

            let nut = -self.nut_to_bow.next();
            let bridge = -self.string_filter.tick(self.bow_to_bridge.next()).tanh();

            let dv = bow_velocity - (nut + bridge);

            let phat = ((dv + 0.001) * bow_force + 0.75)
                .powf(-4.0)
                .max(0.0)
                .min(0.98);

            self.bow_to_bridge.tick(nut + phat * dv);
            self.nut_to_bow.tick(bridge + phat * dv);

            let mut output = bridge;
            for f in self.body_filters.iter_mut() {
                output = f.tick(output);
            }

            r[i] = output as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
        self.string_filter = OnePole::new(0.75 - (0.2 * 22050.0 / rate as f64), 0.9);
    }
}

#[derive(Clone)]
pub struct ImaginaryGuitar {
    strings: Vec<(DelayLine, f64, OnePole, bool)>,
    per_sample: f64,
    body_filters: [Biquad; 6],
}

impl Default for ImaginaryGuitar {
    fn default() -> Self {
        Self {
            strings: vec![
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 329.63) * 44100.0);
                        l
                    },
                    329.63,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 246.94) * 44100.0);
                        l
                    },
                    246.94,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 196.0) * 44100.0);
                        l
                    },
                    196.0,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 146.83) * 44100.0);
                        l
                    },
                    146.83,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 110.0) * 44100.0);
                        l
                    },
                    110.0,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
                (
                    {
                        let mut l = DelayLine::default();
                        l.set_delay((1.0 / 82.40) * 44100.0);
                        l
                    },
                    82.0,
                    OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
                    false,
                ),
            ],
            body_filters: [
                Biquad::new(1.0, 1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.988, -1.9867, 0.9923),
            ],
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for ImaginaryGuitar {
    type Input = U12;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let crossover = 0.005;

            let mut diffusion = 0.0;
            for (s, _, _, _) in &mut self.strings {
                let v = s.next();
                diffusion += v;
            }
            diffusion /= self.strings.len() as f64;
            for f in self.body_filters.iter_mut() {
                diffusion = f.tick(diffusion);
            }
            r[i] = diffusion as f32;
            for (j, (s, base_freq, f, triggered)) in self.strings.iter_mut().enumerate() {
                let fret = input[j * 2][i];
                s.set_delay(((1.0 / (*base_freq)) / self.per_sample) * (1.0 - fret as f64));
                let trigger = input[j * 2 + 1][i];
                if trigger > 0.5 {
                    if !*triggered {
                        *triggered = true;
                        s.add_at(10.0, 1.0);
                    }
                } else {
                    *triggered = false;
                }
                let mut v = s.next();
                v = -f.tick(v + diffusion * crossover as f64);
                s.tick(v);
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
        for (d, freq, f, _) in &mut self.strings {
            *f = OnePole::new(0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64), 0.99);
            d.set_delay((1.0 / (*freq)) / self.per_sample);
        }
    }
}

#[derive(Clone)]
pub struct StringBodyFilter {
    filters: [Biquad; 6],
}

impl Default for StringBodyFilter {
    fn default() -> Self {
        Self {
            filters: [
                Biquad::new(1.0, 1.5667, 0.3133, -0.5509, -0.3925),
                Biquad::new(1.0, -1.9537, 0.9542, -1.6357, 0.8697),
                Biquad::new(1.0, -1.6683, 0.8852, -1.7674, 0.8735),
                Biquad::new(1.0, -1.8585, 0.9653, -1.8498, 0.9516),
                Biquad::new(1.0, -1.9299, 0.9621, -1.9354, 0.9590),
                Biquad::new(1.0, -1.9800, 0.988, -1.9867, 0.9923),
            ],
        }
    }
}

impl Node for StringBodyFilter {
    type Input = U1;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let mut v = input[0][i] as f64;
            for f in self.filters.iter_mut() {
                v = f.tick(v);
            }
            r[i] = v as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

#[derive(Clone)]
pub struct PluckedString {
    line: DelayLine,
    string_filter: OnePole,
    triggered: bool,
    per_sample: f64,
}

impl Default for PluckedString {
    fn default() -> Self {
        Self {
            line: DelayLine::default(),
            string_filter: OnePole::new(0.75 - (0.2 * 22050.0 / 44100.0), 0.9),
            triggered: false,
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for PluckedString {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let freq = input[0][i];
            let trigger = input[1][i];
            let mut slap_threshold = input[2][i];
            if slap_threshold == 0.0 {
                slap_threshold = 1.0;
            }
            self.line.set_delay((1.0 / (freq as f64)) / self.per_sample);
            let pluck = if trigger > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    trigger as f64
                } else {
                    0.0
                }
            } else {
                self.triggered = false;
                0.0
            };
            let v = -self.string_filter.tick(self.line.next());
            self.line.tick(v.min(slap_threshold as f64) + pluck);
            r[i] = v as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
        self.string_filter = OnePole::new(0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64), 0.99);
    }
}

#[derive(Clone)]
pub struct SympatheticString {
    line: DelayLine,
    per_sample: f64,
}

impl Default for SympatheticString {
    fn default() -> Self {
        Self {
            line: DelayLine::default(),
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for SympatheticString {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let freq = input[0][i];
            let driver = input[1][i];
            self.line.set_delay((1.0 / (freq as f64)) / self.per_sample);
            let v = -(self.line.next() * input[2][i] as f64).tanh(); //self.string_filter.tick(self.line.next());
            self.line.tick(v + driver as f64);
            r[i] = v as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
    }
}


#[derive(Clone)]
pub struct WaveMesh {
    lines: BlockDelayLine,
    junctions: Vec<(Option<OnePole>, Vec<usize>, Vec<usize>)>,
    lines_buffer: Vec<f64>,
    sample_rate: f64,
    gain: f64,
}

impl Default for WaveMesh {
    fn default() -> Self {
        let width = 10i32;
        let height = 10i32;

        let mut nodes = indexmap::IndexMap::new();
        let rate = 44100.0;
        for x in 0..width {
            nodes.insert(
                (x, -1),
                (
                    Some(OnePole::new(
                        0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64),
                        -0.85,
                    )),
                    vec![(x, 0)],
                ),
            );
            nodes.insert(
                (x, height),
                (
                    Some(OnePole::new(
                        0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64),
                        -0.85,
                    )),
                    vec![(x, height - 1)],
                ),
            );
            for y in 0..width {
                let inputs = vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)];
                nodes.insert((x, y), (None, inputs));
            }
        }
        for y in 0..width {
            nodes.insert(
                (-1, y),
                (
                    Some(OnePole::new(
                        0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64),
                        -0.85,
                    )),
                    vec![(0, y)],
                ),
            );
            nodes.insert(
                (width, y),
                (
                    Some(OnePole::new(
                        0.75 - (0.2 * (rate as f64 / 2.0) / rate as f64),
                        -0.85,
                    )),
                    vec![(width - 1, y)],
                ),
            );
        }
        let mut lines_map = indexmap::IndexMap::new();
        let mut lines = 0;
        let mut junctions = Vec::new();
        for (src, (reflective, ns)) in nodes {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for dst in ns {
                let input = *lines_map.entry((src, dst)).or_insert_with(|| {
                    lines += 1;
                    lines - 1
                });
                inputs.push(input);
                let output = *lines_map.entry((dst, src)).or_insert_with(|| {
                    let mut d = DelayLine::default();
                    lines += 1;
                    lines - 1
                });
                outputs.push(output);
            }
            junctions.push((reflective, inputs, outputs));
        }

        Self {
            lines_buffer: vec![0.0; lines],
            lines: BlockDelayLine::new(lines, 100.0),
            junctions,
            sample_rate: 44100.0,
            gain: 0.85,
        }
    }
}

impl Node for WaveMesh {
    type Input = U3;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let driver = input[0];
        let gain = input[1];
        let freq = input[2];

        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let gain = gain[i] as f64;
            if gain != self.gain {
                for (f, _, _) in &mut self.junctions {
                    if let Some(f) = f {
                        f.set_gain(gain);
                    }
                }
            }
            let line_len = freq[i] as f64 * self.sample_rate * (10.0 / self.lines.len);
            self.lines.set_delay(line_len);
            {
                let Self {
                    junctions,
                    lines_buffer,
                    lines,
                    ..
                } = self;
                let v = lines.next();
                for (reflective, in_edges, out_edges) in junctions {
                    let mut b = 0.0;
                    for (e, o) in in_edges.iter().zip(out_edges.iter()) {
                        let v = v[*e];
                        if reflective.is_none() {
                            lines_buffer[*o] = -v;
                            b += v;
                        } else {
                            lines_buffer[*o] = 0.0;
                            b -= v;
                        }
                    }
                    if let Some(f) = reflective {
                        b = f.tick(b);
                    } else {
                        b *= 0.5;
                    }
                    for o in out_edges {
                        lines_buffer[*o] += b;
                    }
                }
            }
            let driver = driver[i] as f64 / self.lines.width as f64;
            r[i] = self.lines_buffer[0] as f32;
            {
                let Self {
                    lines_buffer,
                    lines,
                    ..
                } = self;
                let b = lines.input_buffer();
                b.copy_from_slice(&lines_buffer);
                b.iter_mut().for_each(|b| *b += driver);
                lines.tick();
            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate as f64;
        for (f, _, _) in &mut self.junctions {
            if let Some(f) = f {
                *f = OnePole::new(0.75 - (0.2 * 22050.0 / rate as f64), 0.8);
            }
        }
    }
}

// Based on: https://ccrma.stanford.edu/software/clm/compmus/clm-tutorials/pm.html#s-f
#[derive(Clone)]
pub struct Flute {
    embouchure: DelayLine,
    body: DelayLine,
    y2: f64,
    y1: f64,
    rng: rand::rngs::StdRng,
    dc_blocker: DCBlocker,
    sample_rate: f64,
}

impl Default for Flute {
    fn default() -> Self {
        Self {
            embouchure: DelayLine::default(),
            body: DelayLine::default(),
            y1: 0.0,
            y2: 0.0,
            rng: rand::rngs::StdRng::from_rng(thread_rng()).unwrap(),
            dc_blocker: DCBlocker::default(),
            sample_rate: 44100.0,
        }
    }
}

impl Node for Flute {
    type Input = U7;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let flow = input[1];
        let noise_amount = input[2];
        let feedback_1 = input[3];
        let feedback_2 = input[4];
        let lowpass_cutoff = input[5];
        let embouchure_ratio = input[6];

        let mut r = [0.0; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let freq = (freq[i] as f64).max(20.0);
            let freq = -3.508965 + 0.9918286*freq + 0.0001488014*freq.powi(2);
            self.body.set_delay((1.0 / freq) * self.sample_rate);
            self.embouchure
                .set_delay((1.0 / freq) * self.sample_rate * embouchure_ratio[i] as f64);
            let flow = flow[i] as f64;
            let n = (self.rng.gen_range(-1.0..1.0) + self.y2) * 0.5;
            self.y2 = n;
            let excitation = n * flow * noise_amount[i] as f64 + flow;
            let body_out = self.body.next();
            let embouchure_out = self.embouchure.next();
            let embouchure_out = embouchure_out - embouchure_out.powi(3);

            self.embouchure
                .tick(body_out * feedback_1[i] as f64 + excitation);
            let body_in = embouchure_out + body_out * feedback_2[i] as f64;
            let a = lowpass_cutoff[i] as f64;
            let body_in = a * body_in + (1.0 - a) * self.y1;
            self.y1 = body_in;
            self.body.tick(self.dc_blocker.tick(body_in));

            r[i] = body_out as f32;
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate as f64;
    }
}
#[derive(Clone)]
pub struct PennyWhistle {
    delay_a: DelayLine,
    delay_b: DelayLine,
    freq: f64,
    old_freq: f64,
    a_or_b: bool,
    breath: f64,
    breathing: f64,
    crossfade: f64,
    excitation_filter: Simper,
    body_filter_a: Simper,
    body_filter_b: Simper,
    dc_blocker: DCBlocker,
    per_sample: f64,
}

impl Default for PennyWhistle {
    fn default() -> Self {
        Self {
            delay_a: DelayLine::default(),
            delay_b: DelayLine::default(),
            crossfade: 0.0,
            freq: 0.0,
            old_freq: 0.0,
            a_or_b: false,
            breath: 10.0,
            breathing: 0.0,
            excitation_filter: Simper::default(),
            body_filter_a: Simper::default(),
            body_filter_b: Simper::default(),
            dc_blocker: DCBlocker::default(),
            per_sample: 1.0 / 44100.0,
        }
    }
}

impl Node for PennyWhistle {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let freq = input[0];
        let pressure = input[1];
        let note_trigger = input[2];
        let feedback = input[3];

        let mut r = [0.0f32; BLOCK_SIZE];
        let mut rng = thread_rng();

        for i in 0..BLOCK_SIZE {
            let mut freq = freq[i].max(1.0);
            let mut pressure = pressure[i];
            if pressure == 0.0 {
                self.breath = (self.breath + self.per_sample*10.0*2.0).min(12.0);
            }
            let mut excitation_cutoff = 6000.0;
            let mut body_cutoff = 8.5;
            let mut noise_scale = 0.01;
            let mut feedback = feedback[i];
            if freq >=  1046.50 {
                // Second octave
                excitation_cutoff *= 1.5;
                body_cutoff /= 2.0;
                noise_scale *= 1.5;
                freq += -1.0658208528612741e+002
                    +  2.3408841612785913e-001 * freq
                    + -6.6071059587061859e-005 * freq.powi(2);
            } else {

                freq += -3.6423050308971732e+000
                    +  3.3458850639015048e-002 * freq;
            }
            if self.breath < 0.0 && note_trigger[i] > 0.5 {
                if self.breathing == 0.0 {
                    self.breathing = 1.0;
                    println!("breath");
                }
            }
            if self.breathing > 0.0 {
                excitation_cutoff = 1500.0 + 2000.0 * (1.0-self.breathing as f32);
                noise_scale = self.breathing as f32*0.2;
            }
            let noise = rng.gen_range(-1.0..1.0) * noise_scale;
            self.excitation_filter.set_parameters(excitation_cutoff, 0.0);
            let excitation = self.excitation_filter.low(
                noise,
            );
            if self.breathing > 0.0 {
                r[i] += excitation;
                pressure = 0.0;
                feedback -= self.breathing.powf(1.5) as f32*feedback*0.8;
                self.breath = thread_rng().gen_range(8.0..12.0);
            }

            if (self.freq-freq as f64).abs() > 0.5 {
                self.a_or_b = !self.a_or_b;
                self.old_freq = self.freq;
                self.freq=freq as f64;
                if self.a_or_b {
                    self.delay_a
                        .set_delay((1.0 / (freq as f64)) / self.per_sample as f64);
                } else {
                    self.delay_b
                        .set_delay((1.0 / (freq as f64)) / self.per_sample as f64);
                }
                self.crossfade = 1.0;
            }
            let delay_out = if self.a_or_b {
                self.body_filter_b.set_parameters(body_cutoff*self.old_freq as f32, 0.0);
                let mut r = self.body_filter_b.low(self.delay_b.next() as f32) * self.crossfade as f32;
                self.body_filter_a.set_parameters(body_cutoff*self.freq as f32, 0.0);
                r += self.body_filter_a.low(self.delay_a.next() as f32) * (1.0-self.crossfade as f32);
                r
            } else {
                self.body_filter_a.set_parameters(body_cutoff*self.old_freq as f32, 0.0);
                let mut r = self.body_filter_a.low(self.delay_a.next() as f32) * self.crossfade as f32;
                self.body_filter_b.set_parameters(body_cutoff*self.freq as f32, 0.0);
                r += self.body_filter_b.low(self.delay_b.next() as f32) * (1.0-self.crossfade as f32);
                r
            };
            self.crossfade = (self.crossfade - self.per_sample as f64*20.0).max(0.0);
            self.breath -= self.per_sample;
            self.breathing = (self.breathing - self.per_sample * 20.0).max(0.0);
            r[i] += delay_out;
            let body_feedback = self
                .dc_blocker
                .tick(((delay_out * feedback).tanh() + excitation * pressure) as f64);
            self.delay_a.tick(body_feedback as f64);
            self.delay_b.tick(body_feedback as f64);
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.per_sample = 1.0 / rate as f64;
        self.excitation_filter.set_sample_rate(rate);
        self.body_filter_a.set_sample_rate(rate);
        self.body_filter_b.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct ToneHoleFlute {
    tone_holes: Vec<ToneHole>,
    plus_lines: Vec<DelayLine>,
    plus_lines_buffer: Vec<f64>,
    minus_lines: Vec<DelayLine>,
    minus_lines_buffer: Vec<f64>,
    body_filter: Simper,
}

impl Default for ToneHoleFlute {
    fn default() -> Self {
        let mut body_filter = Simper::default();
        body_filter.set_parameters(10000.0, 0.0);
        Self {
            tone_holes: (0..4).map(|_| ToneHole::default()).collect(),
            minus_lines: (0..4).map(|_| DelayLine::default()).collect(),
            minus_lines_buffer: vec![0.0; 4],
            plus_lines: (0..4).map(|_| DelayLine::default()).collect(),
            plus_lines_buffer: vec![0.0; 4],
            body_filter,
        }
    }
}

#[derive(Clone)]
struct ToneHole {
    line: DelayLine,
    filter: Simper,
}

impl Default for ToneHole {
    fn default() -> Self {
        let mut filter = Simper::default();
        filter.set_parameters(10000.0, 0.0);
        Self {
            line: DelayLine::default(),
            filter,
        }
    }
}

impl ToneHole {
    fn tick(&mut self, pa_plus: f64, pb_minus: f64, r0: f64, hole_reflectivity: f64) -> (f64, f64) {
        let pth_minus = self.line.next();

        let middle = (pa_plus + pth_minus*-2.0 + pb_minus) * r0;
        self.line.tick(-(self.filter.low((pth_minus*-1.0+pb_minus+middle+pa_plus) as f32) as f64*hole_reflectivity).tanh());

        (
            middle+pa_plus,
            middle+pb_minus,
        )
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.line.set_delay((1.0/6880.0) * rate as f64);
    }
}

impl Node for ToneHoleFlute {
    type Input = U4;
    type Output = U1;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let noise_scale = input[0];
        let r0 = input[1];
        let hole_reflectivity = input[2];
        let feedback = input[3];

        let mut r = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            let r0 = r0[i] as f64;
            let hole_reflectivity = hole_reflectivity[i] as f64;
            let feedback = feedback[i] as f64;

            self.plus_lines_buffer.iter_mut().zip(&self.plus_lines).for_each(|(b,l)| *b = l.next());
            self.minus_lines_buffer.iter_mut().zip(&self.minus_lines).for_each(|(b,l)| *b = l.next());
            for (j, ((h, plus_in), minus_in)) in self.tone_holes.iter_mut().zip(&self.plus_lines_buffer).zip(&self.minus_lines_buffer).enumerate() {
                let reflectivity = if hole_reflectivity > i as f64 { 0.0 } else { 1.0 };
                let (positive, negative) = h.tick(*plus_in, *minus_in, r0, reflectivity);
                if j > 0 {
                    self.minus_lines[j-1].tick(negative);
                } else {
                    self.plus_lines[0].tick(-negative + thread_rng().gen_range(-1.0..1.0)*noise_scale[j] as f64);
                }
                if j < self.plus_lines.len()-1 {
                    self.plus_lines[j+1].tick(positive);
                } else {
                    self.minus_lines[j].tick(-(self.body_filter.low(positive as f32) as f64 * feedback).tanh());
                    r[i] = positive as f32;
                }

            }
        }
        arr![[f32; BLOCK_SIZE]; r]
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for h in &mut self.tone_holes {
            h.set_sample_rate(rate);
        }
        for l in self.minus_lines.iter_mut().chain(&mut self.plus_lines) {
            l.set_delay((1.0/880.0) * rate as f64);
        }
    }
}
