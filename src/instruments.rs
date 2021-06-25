use rand::prelude::*;
use packed_simd_2::f32x8;
use crate::{
    type_list::{Value, NoValue, ValueT, Combine},
    simd_graph::*,
};

macro_rules! routing {
    ($struct_name:ident, $($field_name:ident:$value_type:ty),* $(,)*) => {
        #[derive(Clone)]
        pub struct $struct_name {
            $(
                $field_name: Box<dyn Node<Input=NoValue, Output=Value<$value_type>>>,
            )*
        }

        impl $struct_name {
            $(
                pub fn $field_name(&mut self, n: impl Node<Input=NoValue, Output=Value<$value_type>> + 'static) {
                    self.$field_name = Box::new(n);
                }
            )*

            pub fn set_sample_rate(&mut self, rate: f32) {
                $(
                    self.$field_name.set_sample_rate(rate);
                )*
            }
        }
    }
}
routing! {
    BassRouting,
    gate: (f32x8,),
    pitch: (f32x8,),
}

#[derive(Clone)]
pub struct Bass {
    output: Box<dyn Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>>>,
    pitch: f32,
    pub routing: BassRouting,
    triggered: bool,
}
impl Default for Bass {
    fn default() -> Self {
        let routing = BassRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            pitch: Box::new(Constant(f32x8::splat(0.0))),
        };

        let pop = Pipe(curry(Constant(f32x8::splat(2.0)), Mul::default()), Pipe(curry(Pipe(Constant(f32x8::splat(20.0)), Pipe(WaveTable::noise(), Rescale(0.0, 1.0))), Mul::default()), WaveTable::sine()));

        let decay_freq = Pipe(
            Pipe(
                curry(Constant(f32x8::splat(0.5)), Mul::default()),
                Branch::new(
                    Pass::default(),
                    Pipe(WaveTable::positive_sine(), Rescale(0.25, 1.0))
                )
            ),
            Mul::default()
        );
        let decay = Pipe(decay_freq, WaveTable::sine());

        let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let attack = 0.025;
            let release= 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t/attack).min(1.0)
            } else if t < attack+decay {
                let t = t - attack ;
                let t = 1.0 - (t/decay).min(1.0);
                t + (1.0-t)*sustain
            } else if t < attack+decay+sustain_time {
                sustain
            } else {
                let t = t - (attack+decay+sustain_time);
                sustain - (t/release).min(1.0)*sustain
            };
            v = v*0.9 + new_v*0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let decay = Pipe(
            Stack::new(
                envelope,
                decay
            ),
            Mul::default()
        );
        let decay = Pipe(Pipe(decay, curry(Constant(f32x8::splat(0.3)), Mul::default())), Split);

        let f = move |t: f32, off_time:Option<f32>| {
            let attack = 0.025;
            let release= 0.025;
            if t < attack {
                (t/attack).min(1.0)
            } else {
                let t = t - attack ;
                1.0 - (t/release).min(1.0)
            }
        };
        let envelope = ThreshEnvelope::new(f);

        let pop = Pipe(
            Stack::new(
                envelope,
                Pipe(pop, curry(Constant(f32x8::splat(0.01)), Mul::default()))
            ),
            Mul::default()
        );
        let pop = Pipe(pop, Split);
        let output = Pipe(Branch::new(pop, decay), Mix);



        Self {
            routing,
            output: Box::new(output),
            pitch: 440.0,
            triggered: false,
        }
    }
}

impl Node for Bass {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output.process(Value((*gate.car(), f32x8::splat(self.pitch))))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

routing! {
    LeadRouting,
    gate: (f32x8,),
    pitch: (f32x8,),
}

#[derive(Clone)]
pub struct Lead {
    output: Box<dyn Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>>>,
    pitch: f32,
    pub routing: LeadRouting,
    triggered: bool,
}
impl Default for Lead {
    fn default() -> Self {
        let routing = LeadRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            pitch: Box::new(Constant(f32x8::splat(0.0))),
        };

        let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let attack = 0.025;
            let release= 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t/attack).min(1.0)
            } else if t < attack+decay {
                let t = t - attack ;
                let t = 1.0 - (t/decay).min(1.0);
                t + (1.0-t)*sustain
            } else if t < attack+decay+sustain_time {
                sustain
            } else {
                let t = t - (attack+decay+sustain_time);
                sustain - (t/release).min(1.0)*sustain
            };
            v = v*0.9 + new_v*0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let filter = curry(Branch::new(Constant(f32x8::splat(800.0)), Constant(f32x8::splat(1.0))), Biquad::lowpass());
        let output = Pipe(
            Stack::new(
                envelope,
                Pipe(
                    Branch::new(
                        Pipe(
                            Pipe(WaveTable::square(), curry(Constant(f32x8::splat(0.05)), Mul::default())),
                            filter
                        ),
                        Pipe(
                            curry(Constant(f32x8::splat(0.25)), Mul::default()),
                            Pipe(WaveTable::sine(), curry(Constant(f32x8::splat(0.05)), Mul::default())),
                        ),
                    ),
                    Add::default()
                )
            ),
            Mul::default()
        );
        let output = Pipe(output, Split);

        Self {
            routing,
            output: Box::new(output),
            pitch: 440.0,
            triggered: false,
        }

    }
}

impl Node for Lead {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output.process(Value((*gate.car(), f32x8::splat(self.pitch))))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct Lead2 {
    output: Box<dyn Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>>>,
    pitch: f32,
    pub routing: LeadRouting,
    triggered: bool,
}
impl Default for Lead2 {
    fn default() -> Self {
        let routing = LeadRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            pitch: Box::new(Constant(f32x8::splat(0.0))),
        };

        let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let attack = 0.025;
            let release= 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t/attack).min(1.0)
            } else if t < attack+decay {
                let t = t - attack ;
                let t = 1.0 - (t/decay).min(1.0);
                t + (1.0-t)*sustain
            } else if t < attack+decay+sustain_time {
                sustain
            } else {
                let t = t - (attack+decay+sustain_time);
                sustain - (t/release).min(1.0)*sustain
            };
            v = v*0.9 + new_v*0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let output = Pipe(
            Stack::new(
                envelope,
                Pipe(WaveTable::sine(), curry(Constant(f32x8::splat(0.5)), Mul::default())),
            ),
            Mul::default()
        );
        let output = Pipe(output, Split);

        Self {
            routing,
            output: Box::new(output),
            pitch: 440.0,
            triggered: false,
        }

    }
}

impl Node for Lead2 {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output.process(Value((*gate.car(), f32x8::splat(self.pitch))))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

pub fn lead() -> impl Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>> + Clone {
    let f = move |t: f32, off_time:Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release= 0.05;
            let t = t - off_time;
            1.0 - (t/release).min(1.0)
        } else {
            let attack = 0.2;
            (t/attack).min(1.0)
        };
        new_v
    };
    let filter_envelope = Pipe(ThreshEnvelope::new(f), curry(Constant(f32x8::splat(1400.0)), Mul::default()));
    let filter = curry(Stack::new(filter_envelope, Constant(f32x8::splat(1.0))), Biquad::lowpass());
    let osc1 = Pipe(
        Branch::new(
            Pass::default(),
            WaveTable::saw()
        ),
        filter
    );
    let mut v = 0.0;
    let f = move |t: f32, off_time:Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release= 0.3;
            let t = t - off_time;
            1.0 - (t/release).min(1.0)
        } else {
            let attack = 0.2;
            (t/attack).min(1.0)
        };
        v = (v * 0.9) + (new_v * 0.1);
        v
    };
    let envelope = ThreshEnvelope::new(f);
    let envelope = Pipe(Flip::default(), curry(envelope, Mul::default()));
    let voice = curry(
        osc1,
        envelope
    );
    let voice = Pipe(
        voice,
        curry(Stack::new(Constant(f32x8::splat(1000.0)), Constant(f32x8::splat(1.0))), Biquad::lowpass())
    );
    Pipe(
        Unison::new(
            voice,
            3.0, 1.0, 9
        ),
        Stack::new(
            curry(Constant(f32x8::splat(0.025)), Mul::default()),
            curry(Constant(f32x8::splat(0.025)), Mul::default()),
        )
    )
}

pub fn drum() -> impl Node<Input=Value<(f32x8,)>, Output=Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(
                Impulse::default(),
                Constant(f32x8::splat(400.0))
            ),
            HarmonicOscillator::new(2.0),
        ),
    Split
    )
}

pub fn snare() -> impl Node<Input=Value<(f32x8,)>, Output=Value<(f32x8, f32x8)>> + Clone{
    let f = move |t: f32, off_time:Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release= 0.005;
            let t = t - off_time;
            1.0 - (t.powf(0.15)/release).min(1.0)
        } else {
            let attack = 0.005;
            (t.powf(0.15)/attack).min(1.0)
        };
        new_v
    };
    let envelope = ThreshEnvelope::new(f);
    let envelope = Pipe(Flip::default(), curry(envelope, Mul::default()));
    let noise = Pipe(Constant(f32x8::splat(880.0)), WaveTable::noise());
    let noise = curry(
        Pipe(noise, curry(Constant(f32x8::splat(0.25)), Mul::default())),
        envelope
    );
    Pipe(
        Pipe(
            Pipe(
                Branch::new(
                    Pipe(
                        Stack::new(
                            Impulse::default(),
                            Constant(f32x8::splat(600.0))
                        ),
                        HarmonicOscillator::new(2.0),
                    ),
                    noise
                ),
                Add::default()
            ),
            curry(
                Stack::new(Constant(f32x8::splat(1800.0)), Constant(f32x8::splat(1.0))),
                Biquad::lowpass(),
            )
        ),
        Split
    )
}
pub fn clave() -> impl Node<Input=Value<(f32x8,)>, Output=Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(
                Impulse::default(),
                Constant(f32x8::splat(900.0))
            ),
            HarmonicOscillator::new(1.0),
        ),
    Split
    )
}
pub fn clave2() -> impl Node<Input=Value<(f32x8,)>, Output=Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(
                Impulse::default(),
                Constant(f32x8::splat(2000.0))
            ),
            HarmonicOscillator::new(1.0),
        ),
    Split
    )
}

routing! {
    ScaleRouting,
    root: (f32,),
    intervals: (Vec<f32>,),
}

#[derive(Clone)]
pub struct Scale {
    pub routing: ScaleRouting,
}
impl Scale {
    pub fn unison() -> f32 {
        1.0
    }
    pub fn minor_second() -> f32 {
        1.0594630943592953f32
    }
    pub fn major_second() -> f32 {
        1.0594630943592953f32.powi(2)
    }
    pub fn minor_third() -> f32 {
        1.0594630943592953f32.powi(3)
    }
    pub fn major_third() -> f32 {
        1.0594630943592953f32.powi(4)
    }
    pub fn perfect_fourth() -> f32 {
        1.0594630943592953f32.powi(5)
    }
    pub fn minor_fifth() -> f32 {
        1.0594630943592953f32.powi(6)
    }
    pub fn perfect_fifth() -> f32 {
        1.0594630943592953f32.powi(7)
    }
    pub fn minor_sixth() -> f32 {
        1.0594630943592953f32.powi(8)
    }
    pub fn major_sixth() -> f32 {
        1.0594630943592953f32.powi(9)
    }
    pub fn minor_seventh() -> f32 {
        1.0594630943592953f32.powi(10)
    }
    pub fn major_seventh() -> f32 {
        1.0594630943592953f32.powi(11)
    }
}
impl Default for Scale {
    fn default() -> Self {
        let routing = ScaleRouting {
            root: Box::new(Constant(440.0)),
            intervals: Box::new(Constant(vec![
                1.0,
                1.0594630943592953f32.powi(2),
                1.0594630943592953f32.powi(4),
                1.0594630943592953f32.powi(5),
                1.0594630943592953f32.powi(7),
                1.0594630943592953f32.powi(9),
                1.0594630943592953f32.powi(10),
            ])),
        };
        Self {
            routing,
        }
    }
}

impl Node for Scale {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let intervals_in = self.routing.intervals.process(NoValue);
        let intervals = intervals_in.car();
        let root = *self.routing.root.process(NoValue).car();

        let idx = (input.car() % 1.0) * intervals.len() as f32;

        Value((root * intervals[idx as usize],))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.root.set_sample_rate(rate);
        self.routing.intervals.set_sample_rate(rate);
    }
}

routing! {
    PadRouting,
    gate: (f32x8,),
    pitch: (f32x8,),
}

#[derive(Clone)]
pub struct Pad {
    output: Box<dyn Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>>>,
    pub routing: PadRouting,
}
impl Default for Pad {
    fn default() -> Self {
        let routing = PadRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            pitch: Box::new(Constant(f32x8::splat(0.0))),
        };

        let lfo1 = Pipe(Pipe(Constant(7.0), LfoSine::positive()), Rescale(0.8, 1.0));
        let lfo2 = Pipe(Pipe(Constant(8.5), LfoSine::positive()), Rescale(0.8, 1.0));
        let lfo3 = Pipe(Pipe(Constant(2.3), LfoSine::positive()), Rescale(0.5, 1.0));

        let lfo4 = Pipe(Pipe(Constant(8.2), LfoSine::positive()), Rescale(0.98, 1.02));
        let lfo5 = Pipe(Pipe(Constant(7.5), LfoSine::positive()), Rescale(0.98, 1.02));
        let lfo6 = Pipe(Pipe(Constant(1.3), LfoSine::positive()), Rescale(0.98, 1.02));

        let osc1 = Pipe(
                Pipe(
                    curry(lfo4, Mul::default()),
                    Pipe(
                        Pipe(WaveTable::square(), curry(lfo1, Mul::default())),
                        curry(Stack::new(Constant(f32x8::splat(400.0)), Constant(f32x8::splat(1.0))), Biquad::lowpass())
                    )
                ),
                Split
        );
        let osc2 = Pipe(Pipe(Pipe(
            Pipe(curry(lfo5, Mul::default()), curry(Constant(f32x8::splat(Scale::major_third())), Mul::default())),
            WaveTable::sine(),
        ), curry(lfo2, Mul::default())), Split);
        let osc3 = Pipe(Pipe(Pipe(
            Pipe(curry(lfo6, Mul::default()), curry(Constant(f32x8::splat(Scale::perfect_fifth())), Mul::default())),
            WaveTable::sine(),
        ), curry(lfo3, Mul::default())), Split);


       let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release= 1.5;
                let t = t - off_time;
                1.0 - (t/release).min(1.0)
            } else {
                let attack = 1.0;
                (t/attack).min(1.0)
            };
            v = (v * 0.9) + (new_v * 0.1);
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let output = Pipe(
            Stack::new(
                Pipe(
                    Branch::new(
                        Pipe(
                            Branch::new(
                                osc1,
                                osc2
                            ),
                            Mix
                        ),
                        osc3
                    ),
                    Mix
                ),
                Pipe(envelope, Split)
            ),
            Pipe(
                Transpose,
                Stack::new(
                    Mul::default(),
                    Mul::default(),
                )
            )
        );


        Self {
            routing,
            output: Box::new(output),
        }
    }
}

impl Node for Pad {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let pitch = self.routing.pitch.process(NoValue);
        self.output.process(Value((*pitch.car(), *gate.car())))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

routing! {
    RiserRouting,
    gate: (f32x8,),
    pitch: (f32x8,),
}

#[derive(Clone)]
pub struct Riser {
    osc: Box<dyn Node<Input=Value<(f32x8,)>, Output=Value<(f32x8, f32x8)>>>,
    output: Box<dyn Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>>>,
    pub routing: RiserRouting,
    triggered: bool,
    base_pitch: f32x8,
    time: f32,
    per_sample: f32,
}
impl Default for Riser {
    fn default() -> Self {
        let routing = RiserRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            pitch: Box::new(Constant(f32x8::splat(0.0))),
        };
        let osc1 = Pipe(WaveTable::saw(), Split);
        let osc2 = Pipe(Pipe(WaveTable::noise(), curry(Constant(f32x8::splat(0.5)), Mul::default())), Split);
        let osc = Pipe(Branch::new(
            osc1,
            osc2,
        ), Mix);
        let output = Stutter::new(10, 2.0);

        Self {
            osc: Box::new(osc),
            output: Box::new(output),
            routing,
            triggered: false,
            base_pitch: f32x8::splat(0.0),
            time: 0.0,
            per_sample: 0.0,
        }
    }
}
impl Node for Riser {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let pitch = *self.routing.pitch.process(NoValue).car();
        if !self.triggered && gate.car().max_element() > 0.5 {
            self.triggered = true;
            self.time = 0.0;
            self.base_pitch = f32x8::splat(pitch.max_element());
        }
        let input = if self.triggered {
            let t = (self.time/6.0).min(1.0);
            if t == 1.0 {
                self.triggered = false;
            }
            let ta = f32x8::splat(1.0-t);
            let t = f32x8::splat(t);
            let pitch = ((self.base_pitch / f32x8::splat(6.0)) * ta) + self.base_pitch * t;
            let gain = t;
            self.time += f32x8::lanes() as f32 * self.per_sample;
            let Value((l, r)) = self.osc.process(Value((pitch,)));
            Value((
                l * gain,
                r * gain
            ))
        } else {
            Value((f32x8::splat(0.0), f32x8::splat(0.0)))
        };
        self.output.process(input)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.osc.set_sample_rate(rate);
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
        self.per_sample = 1.0/rate;
    }
}

routing! {
    BpmClockRouting,
    bpm: (f32,),
}

#[derive(Clone)]
pub struct BpmClock {
    output: Box<dyn Node<Input=Value<(f32,)>, Output=Value<(f32x8,)>>>,
    pub routing: BpmClockRouting,
}
impl Default for BpmClock {
    fn default() -> Self {
        let routing = BpmClockRouting {
            bpm: Box::new(Constant(120.0)),
        };
        let output = Pipe(
                curry(Constant((1.0/60.0)*4.0), Mul::default()),
                LfoSine::positive()
        );
        let output = Pipe(
            output,
            Impulse::default()
        );
        Self {
            output: Box::new(output),
            routing
        }
    }
}
impl Node for BpmClock {
    type Input = NoValue;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        self.output.process(self.routing.bpm.process(NoValue))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
    }
}

routing! {
    ResamplerRouting,
    gate: (f32x8,),
    speed: (f32x8,),
    jitter: (f32x8,),
    line_in: (f32x8, f32x8),
    freeze: (bool,),
}

#[derive(Clone)]
pub struct Resampler {
    buffer: Delay,
    idx: f32,
    pub routing: ResamplerRouting,
    triggered: bool,
    sample_rate: f32,
    envelope: Box<dyn Node<Input=Value<(f32x8,)>, Output=Value<(f32x8,)>>>,
}
impl Default for Resampler {
    fn default() -> Self {
        let routing = ResamplerRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            speed: Box::new(Constant(f32x8::splat(0.0))),
            jitter: Box::new(Constant(f32x8::splat(0.0))),
            line_in: Box::new(Pipe(Constant(f32x8::splat(0.0)), Split)),
            freeze: Box::new(Constant(false)),
        };

       let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release= 0.5;
                let t = t - off_time;
                1.0 - (t/release).min(1.0)
            } else {
                let attack = 0.5;
                (t/attack).min(1.0)
            };
            v = (v * 0.9) + (new_v * 0.1);
            v
        };
        let envelope = ThreshEnvelope::new(f);


        Self {
            routing,
            buffer: Delay::new(3.0),
            envelope: Box::new(envelope),
            idx: 0.0,
            triggered: false,
            sample_rate: 0.0,
        }
    }
}

impl Node for Resampler {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let gate_max = gate.car().max_element();
        let speed = *self.routing.speed.process(NoValue).car();
        let jitter = *self.routing.jitter.process(NoValue).car();
        let line_in = self.routing.line_in.process(NoValue);
        let freeze = self.routing.freeze.process(NoValue);
        if !freeze.car() {
            self.buffer.process(line_in);
        }
        let e = *self.envelope.process(gate).car();
        let e_on = e.max_element() > 0.0;
        if e_on && !self.triggered {
            self.triggered = true;
            self.idx += jitter.max_element() * self.sample_rate * thread_rng().gen::<f32>();
        } else if !e_on {
            self.triggered = false;
        }

        let r = if self.triggered {
            let len = self.buffer.len() as f32;
            self.idx = self.idx + speed.max_element();
            while self.idx >= len {
                self.idx -= len;
            }
            while self.idx < 0.0{
                self.idx += len;
            }
            let mut r = self.buffer.get(self.idx as usize).unwrap();
            r.0 *= e;
            r.1 *= e;
            r
        } else {
            (f32x8::splat(0.0), f32x8::splat(0.0))
        };
        Value(r)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        self.buffer.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
        self.envelope.set_sample_rate(rate);
    }
}


routing! {
    GranularSamplerRouting,
    gate: (f32x8,),
    speed: (f32x8,),
    jitter: (f32x8,),
    line_in: (f32x8, f32x8),
    freeze: (bool,),
}

#[derive(Clone)]
pub struct GranularSampler {
    buffer: Delay,
    idx: f32,
    pub routing: GranularSamplerRouting,
    triggered: bool,
    sample_rate: f32,
    envelope: Box<dyn Node<Input=Value<(f32x8,)>, Output=Value<(f32x8,)>>>,
}
impl Default for GranularSampler {
    fn default() -> Self {
        let routing = GranularSamplerRouting {
            gate: Box::new(Constant(f32x8::splat(0.0))),
            speed: Box::new(Constant(f32x8::splat(0.0))),
            jitter: Box::new(Constant(f32x8::splat(0.0))),
            line_in: Box::new(Pipe(Constant(f32x8::splat(0.0)), Split)),
            freeze: Box::new(Constant(false)),
        };

       let mut v = 0.0;
        let f = move |t: f32, off_time:Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release= 0.5;
                let t = t - off_time;
                1.0 - (t/release).min(1.0)
            } else {
                let attack = 0.5;
                (t/attack).min(1.0)
            };
            v = (v * 0.9) + (new_v * 0.1);
            v
        };
        let envelope = ThreshEnvelope::new(f);


        Self {
            routing,
            buffer: Delay::new(3.0),
            envelope: Box::new(envelope),
            idx: 0.0,
            triggered: false,
            sample_rate: 0.0,
        }
    }
}

impl Node for GranularSampler {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.process(NoValue);
        let gate_max = gate.car().max_element();
        let speed = *self.routing.speed.process(NoValue).car();
        let jitter = *self.routing.jitter.process(NoValue).car();
        let line_in = self.routing.line_in.process(NoValue);
        let freeze = self.routing.freeze.process(NoValue);
        if !freeze.car() {
            self.buffer.process(line_in);
        }
        let e = *self.envelope.process(gate).car();
        let e_on = e.max_element() > 0.0;
        if e_on && !self.triggered {
            self.triggered = true;
            self.idx += jitter.max_element() * self.sample_rate * thread_rng().gen::<f32>();
        } else if !e_on {
            self.triggered = false;
        }

        let r = if self.triggered {
            self.idx = (self.idx + speed.max_element()) % self.buffer.len() as f32;
            let mut r = self.buffer.get(self.idx as usize).unwrap();
            r.0 *= e;
            r.1 *= e;
            r
        } else {
            (f32x8::splat(0.0), f32x8::splat(0.0))
        };
        Value(r)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        self.buffer.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
        self.envelope.set_sample_rate(rate);
    }
}

pub fn reverb() -> impl Node<Input=Value<(f32x8, f32x8)>, Output=Value<(f32x8, f32x8)>> {
    let mut rng = thread_rng();

    let comb_count = 300;
    fn make_reverb(i: usize, count: usize, rng: &mut ThreadRng) -> impl Node<Input=Value<(f32x8,)>, Output=Value<(f32x8,)>> + Clone {
        let len_range = 11.0+(1.0 - i as f32 / count as f32) * 1500.0;
        let comb_len = (48000.0*(rng.gen_range(10.0..len_range)/1000.0)) as usize;
        let scale_range = ((i as f32 + 1.0) / count as f32) * 3.0;
        let scale = rng.gen_range(-scale_range..scale_range);
        if i == 1 {
            curry(Constant(f32x8::splat(scale)), Comb::new(DelayLine::new(comb_len)))
        } else {
            let inner = make_reverb(i-1, count, rng);
            curry(Constant(f32x8::splat(scale)), Comb::new(DelayLine::new_nested(comb_len, inner)))
        }
    }
    let reverb = Stack::new(
        make_reverb(comb_count, comb_count, &mut rng),
        make_reverb(comb_count, comb_count, &mut rng),
    );

    reverb
}
