use std::{
    collections::HashMap,
    rc::Rc,
    cell::RefCell,
};
use concat_idents::concat_idents;
use crate::{
    simd_graph::*,
    type_list::{Combine, NoValue, Value, ValueT},
};
use inline_tweak::tweak;
use packed_simd_2::{f32x8, u32x8};
use rand::prelude::*;

macro_rules! routing {
    ($struct_name:ident, $($field_name:ident:$value_type:ty),* $(,)*) => {
        #[derive(Clone)]
        pub struct $struct_name {
            $(
                $field_name: (
                    Box<dyn Node<Input=NoValue, Output=Value<$value_type>>>,
                    RawRcConstant<Value<$value_type>>,
                ),
            )*
        }

        impl Default for $struct_name {
            fn default() -> Self {
                Self {
                    $(
                        $field_name: (
                            Box::new(RawConstant(<$value_type as Default>::default())),
                            RawRcConstant::new(<$value_type as Default>::default()).0,
                        ),
                    )*
                }
            }
        }

        impl $struct_name {
            $(
                concat_idents!(bridge_name = $field_name, _, bridge {
                    pub fn $field_name(&mut self, n: impl Node<Input=NoValue, Output=Value<$value_type>> + 'static + Clone) {
                        let b_in = RawBridge(Rc::clone(&self.$field_name.1.0));
                        self.$field_name.0 = Box::new(Pipe(n, b_in));
                    }

                    concat_idents!(fn_name = $field_name, _, bridge {
                           fn fn_name(&self) -> RawRcConstant<Value<$value_type>> {
                               self.$field_name.1.clone()
                           }
                    });
                });
            )*


            pub fn set_sample_rate(&mut self, rate: f32) {
                $(
                    self.$field_name.0.set_sample_rate(rate);
                )*
            }

            pub fn process(&mut self) {
                $(
                    self.$field_name.0.process(NoValue);
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
    output: Box<dyn Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>,
    pitch: f32,
    triggered: bool,
    pub routing: BassRouting,
}
impl Default for Bass {
    fn default() -> Self {
        let routing = BassRouting::default();

        let pop = Pipe(
            curry(Constant(f32x8::splat(2.0)), Mul::default()),
            Pipe(
                curry(
                    Pipe(
                        Constant(f32x8::splat(20.0)),
                        Pipe(WaveTable::noise(), Rescale(0.0, 1.0)),
                    ),
                    Mul::default(),
                ),
                WaveTable::sine(),
            ),
        );

        let decay_freq = Pipe(
            Pipe(
                curry(Constant(f32x8::splat(0.5)), Mul::default()),
                Branch::new(
                    Pass::default(),
                    Pipe(WaveTable::positive_sine(), Rescale(0.25, 1.0)),
                ),
            ),
            Mul::default(),
        );
        let decay = Pipe(decay_freq, WaveTable::sine());

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let attack = 0.025;
            let release = 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t / attack).min(1.0)
            } else if t < attack + decay {
                let t = t - attack;
                let t = 1.0 - (t / decay).min(1.0);
                t + (1.0 - t) * sustain
            } else if t < attack + decay + sustain_time {
                sustain
            } else {
                let t = t - (attack + decay + sustain_time);
                sustain - (t / release).min(1.0) * sustain
            };
            v = v * 0.9 + new_v * 0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let decay = Pipe(Stack::new(envelope, decay), Mul::default());
        let decay = Pipe(
            Pipe(decay, curry(Constant(f32x8::splat(0.3)), Mul::default())),
            Split,
        );

        let f = move |t: f32, off_time: Option<f32>| {
            let attack = 0.025;
            let release = 0.025;
            if t < attack {
                (t / attack).min(1.0)
            } else {
                let t = t - attack;
                1.0 - (t / release).min(1.0)
            }
        };
        let envelope = ThreshEnvelope::new(f);

        let pop = Pipe(
            Stack::new(
                envelope,
                Pipe(pop, curry(Constant(f32x8::splat(0.01)), Mul::default())),
            ),
            Mul::default(),
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
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.0.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output
            .process(Value((*gate.car(), f32x8::splat(self.pitch))))
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
    output: Box<dyn Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>,
    pitch: f32,
    pub routing: LeadRouting,
    triggered: bool,
}
impl Default for Lead {
    fn default() -> Self {
        let routing = LeadRouting::default();

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let attack = 0.025;
            let release = 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t / attack).min(1.0)
            } else if t < attack + decay {
                let t = t - attack;
                let t = 1.0 - (t / decay).min(1.0);
                t + (1.0 - t) * sustain
            } else if t < attack + decay + sustain_time {
                sustain
            } else {
                let t = t - (attack + decay + sustain_time);
                sustain - (t / release).min(1.0) * sustain
            };
            v = v * 0.9 + new_v * 0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let filter = curry(
            Branch::new(Constant(f32x8::splat(800.0)), Constant(f32x8::splat(1.0))),
            Biquad::lowpass(),
        );
        let output = Pipe(
            Stack::new(
                envelope,
                Pipe(
                    Branch::new(
                        Pipe(
                            Pipe(
                                WaveTable::square(),
                                curry(Constant(f32x8::splat(0.05)), Mul::default()),
                            ),
                            filter,
                        ),
                        Pipe(
                            curry(Constant(f32x8::splat(0.25)), Mul::default()),
                            Pipe(
                                WaveTable::sine(),
                                curry(Constant(f32x8::splat(0.05)), Mul::default()),
                            ),
                        ),
                    ),
                    Add::default(),
                ),
            ),
            Mul::default(),
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
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.0.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output
            .process(Value((*gate.car(), *pitch.car())))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct Lead2 {
    output: Box<dyn Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>,
    pitch: f32,
    pub routing: LeadRouting,
    triggered: bool,
}
impl Default for Lead2 {
    fn default() -> Self {
        let routing = LeadRouting::default();

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let attack = 0.025;
            let release = 0.08;
            let sustain = 0.25;
            let sustain_time = 8.0;
            let decay = 0.1;
            let new_v = if t < attack {
                (t / attack).min(1.0)
            } else if t < attack + decay {
                let t = t - attack;
                let t = 1.0 - (t / decay).min(1.0);
                t + (1.0 - t) * sustain
            } else if t < attack + decay + sustain_time {
                sustain
            } else {
                let t = t - (attack + decay + sustain_time);
                sustain - (t / release).min(1.0) * sustain
            };
            v = v * 0.9 + new_v * 0.1;
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let output = Pipe(
            Stack::new(
                envelope,
                Pipe(
                    WaveTable::sine(),
                    curry(Constant(f32x8::splat(0.5)), Mul::default()),
                ),
            ),
            Mul::default(),
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
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.0.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.pitch = pitch.car().max_element();
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.output
            .process(Value((*gate.car(), f32x8::splat(self.pitch))))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}

pub fn lead() -> impl Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>> + Clone {
    let f = move |t: f32, off_time: Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release = 0.05;
            let t = t - off_time;
            1.0 - (t / release).min(1.0)
        } else {
            let attack = 0.2;
            (t / attack).min(1.0)
        };
        new_v
    };
    let filter_envelope = Pipe(
        ThreshEnvelope::new(f),
        curry(Constant(f32x8::splat(1400.0)), Mul::default()),
    );
    let filter = curry(
        Stack::new(filter_envelope, Constant(f32x8::splat(1.0))),
        Biquad::lowpass(),
    );
    let osc1 = Pipe(Branch::new(Pass::default(), WaveTable::saw()), filter);
    let mut v = 0.0;
    let f = move |t: f32, off_time: Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release = 0.3;
            let t = t - off_time;
            1.0 - (t / release).min(1.0)
        } else {
            let attack = 0.2;
            (t / attack).min(1.0)
        };
        v = (v * 0.9) + (new_v * 0.1);
        v
    };
    let envelope = ThreshEnvelope::new(f);
    let envelope = Pipe(Flip::default(), curry(envelope, Mul::default()));
    let voice = curry(osc1, envelope);
    let voice = Pipe(
        voice,
        curry(
            Stack::new(Constant(f32x8::splat(1000.0)), Constant(f32x8::splat(1.0))),
            Biquad::lowpass(),
        ),
    );
    Pipe(
        Unison::new(voice, 3.0, 1.0, 9),
        Stack::new(
            curry(Constant(f32x8::splat(0.025)), Mul::default()),
            curry(Constant(f32x8::splat(0.025)), Mul::default()),
        ),
    )
}

pub fn drum() -> impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(Impulse::default(), Constant(f32x8::splat(400.0))),
            HarmonicOscillator::new(2.0),
        ),
        Split,
    )
}

pub fn snare() -> impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8, f32x8)>> + Clone {
    let f = move |t: f32, off_time: Option<f32>| {
        let new_v = if let Some(off_time) = off_time {
            let release = 0.005;
            let t = t - off_time;
            1.0 - (t.powf(0.15) / release).min(1.0)
        } else {
            let attack = 0.005;
            (t.powf(0.15) / attack).min(1.0)
        };
        new_v
    };
    let envelope = ThreshEnvelope::new(f);
    let envelope = Pipe(Flip::default(), curry(envelope, Mul::default()));
    let noise = Pipe(Constant(f32x8::splat(880.0)), WaveTable::noise());
    let noise = curry(
        Pipe(noise, curry(Constant(f32x8::splat(0.25)), Mul::default())),
        envelope,
    );
    Pipe(
        Pipe(
            Pipe(
                Branch::new(
                    Pipe(
                        Stack::new(Impulse::default(), Constant(f32x8::splat(600.0))),
                        HarmonicOscillator::new(2.0),
                    ),
                    noise,
                ),
                Add::default(),
            ),
            curry(
                Stack::new(Constant(f32x8::splat(1800.0)), Constant(f32x8::splat(1.0))),
                Biquad::lowpass(),
            ),
        ),
        Split,
    )
}
pub fn clave() -> impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(Impulse::default(), Constant(f32x8::splat(900.0))),
            HarmonicOscillator::new(1.0),
        ),
        Split,
    )
}
pub fn clave2() -> impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8, f32x8)>> + Clone {
    Pipe(
        Pipe(
            Stack::new(Impulse::default(), Constant(f32x8::splat(2000.0))),
            HarmonicOscillator::new(1.0),
        ),
        Split,
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
        let mut routing = ScaleRouting::default();
        routing.root(Constant(440.0));
        routing.intervals(Constant(vec![
            1.0,
            1.0594630943592953f32.powi(2),
            1.0594630943592953f32.powi(4),
            1.0594630943592953f32.powi(5),
            1.0594630943592953f32.powi(7),
            1.0594630943592953f32.powi(9),
            1.0594630943592953f32.powi(10),
        ]));
        Self { routing }
    }
}

impl Node for Scale {
    type Input = Value<(f32,)>;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let intervals_in = self.routing.intervals.0.process(NoValue);
        let intervals = intervals_in.car();
        let root = *self.routing.root.0.process(NoValue).car();

        let idx = (input.car() % 1.0) * intervals.len() as f32;

        Value((root * intervals[idx as usize],))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.set_sample_rate(rate);
    }
}

routing! {
    PadRouting,
    gate: (f32x8,),
    pitch: (f32x8,),
}

#[derive(Clone)]
pub struct Pad {
    output: Box<dyn Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>>>,
    pub routing: PadRouting,
}
impl Default for Pad {
    fn default() -> Self {
        let routing = PadRouting::default();

        let lfo1 = Pipe(
            Pipe(Constant(7.0), LfoSine::positive_random_phase()),
            Rescale(0.8, 1.0),
        );
        let lfo2 = Pipe(
            Pipe(Constant(8.5), LfoSine::positive_random_phase()),
            Rescale(0.8, 1.0),
        );
        let lfo3 = Pipe(
            Pipe(Constant(2.3), LfoSine::positive_random_phase()),
            Rescale(0.5, 1.0),
        );

        let lfo4 = Pipe(
            Pipe(Constant(8.2), LfoSine::positive_random_phase()),
            Rescale(0.98, 1.02),
        );
        let lfo5 = Pipe(
            Pipe(Constant(7.5), LfoSine::positive_random_phase()),
            Rescale(0.98, 1.02),
        );
        let lfo6 = Pipe(
            Pipe(Constant(1.3), LfoSine::positive_random_phase()),
            Rescale(0.98, 1.02),
        );

        let osc1 = Pipe(
            Pipe(
                curry(lfo4, Mul::default()),
                Pipe(
                    Pipe(WaveTable::square(), curry(lfo1, Mul::default())),
                    curry(
                        Stack::new(Constant(f32x8::splat(400.0)), Constant(f32x8::splat(1.0))),
                        Biquad::lowpass(),
                    ),
                ),
            ),
            Split,
        );
        let osc2 = Pipe(
            Pipe(
                Pipe(
                    Pipe(
                        curry(lfo5, Mul::default()),
                        curry(Constant(f32x8::splat(Scale::major_third())), Mul::default()),
                    ),
                    WaveTable::sine(),
                ),
                curry(lfo2, Mul::default()),
            ),
            Split,
        );
        let osc3 = Pipe(
            Pipe(
                Pipe(
                    Pipe(
                        curry(lfo6, Mul::default()),
                        curry(
                            Constant(f32x8::splat(Scale::perfect_fifth())),
                            Mul::default(),
                        ),
                    ),
                    WaveTable::sine(),
                ),
                curry(lfo3, Mul::default()),
            ),
            Split,
        );

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release = 1.5;
                let t = t - off_time;
                1.0 - (t / release).min(1.0)
            } else {
                let attack = 1.0;
                (t / attack).min(1.0)
            };
            v = (v * 0.9) + (new_v * 0.1);
            v
        };
        let envelope = ThreshEnvelope::new(f);
        let output = Pipe(
            Stack::new(
                Pipe(Branch::new(Pipe(Branch::new(osc1, osc2), Mix), osc3), Mix),
                Pipe(envelope, Split),
            ),
            Pipe(Transpose, Stack::new(Mul::default(), Mul::default())),
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
        let gate = self.routing.gate.0.process(NoValue);
        let pitch = self.routing.pitch.0.process(NoValue);
        self.output.process(Value((*pitch.car(), *gate.car())))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.output.set_sample_rate(rate);
        self.routing.set_sample_rate(rate);
    }
}



routing! {
    BpmClockRouting,
    bpm: (f32,),
}

#[derive(Clone)]
pub struct BpmClock {
    output: Box<dyn Node<Input = Value<(f32,)>, Output = Value<(f32x8,)>>>,
    pub routing: BpmClockRouting,
}
impl Default for BpmClock {
    fn default() -> Self {
        let mut routing = BpmClockRouting::default();
        routing.bpm(Constant(120.0));
        let output = Pipe(
            curry(Constant((1.0 / 60.0) * 4.0), Mul::default()),
            LfoSine::positive_random_phase(),
        );
        let output = Pipe(output, Impulse::default());
        Self {
            output: Box::new(output),
            routing,
        }
    }
}
impl Node for BpmClock {
    type Input = NoValue;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        self.output.process(self.routing.bpm.0.process(NoValue))
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
    envelope: Box<dyn Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>>>,
}
impl Default for Resampler {
    fn default() -> Self {
        let routing = ResamplerRouting::default();

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release = 0.5;
                let t = t - off_time;
                1.0 - (t / release).min(1.0)
            } else {
                let attack = 0.5;
                (t / attack).min(1.0)
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
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let speed = *self.routing.speed.0.process(NoValue).car();
        let jitter = *self.routing.jitter.0.process(NoValue).car();
        let line_in = self.routing.line_in.0.process(NoValue);
        let freeze = self.routing.freeze.0.process(NoValue);
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
            while self.idx < 0.0 {
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
    envelope: Box<dyn Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>>>,
}
impl Default for GranularSampler {
    fn default() -> Self {
        let routing = GranularSamplerRouting::default();

        let mut v = 0.0;
        let f = move |t: f32, off_time: Option<f32>| {
            let new_v = if let Some(off_time) = off_time {
                let release = 0.5;
                let t = t - off_time;
                1.0 - (t / release).min(1.0)
            } else {
                let attack = 0.5;
                (t / attack).min(1.0)
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
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let speed = *self.routing.speed.0.process(NoValue).car();
        let jitter = *self.routing.jitter.0.process(NoValue).car();
        let line_in = self.routing.line_in.0.process(NoValue);
        let freeze = self.routing.freeze.0.process(NoValue);
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

pub fn reverb() -> impl Node<Input = Value<(f32x8, f32x8)>, Output = Value<(f32x8, f32x8)>> {
    let mut rng = thread_rng();

    let comb_count = 300;
    fn make_reverb(
        i: usize,
        count: usize,
        rng: &mut ThreadRng,
    ) -> impl Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>> + Clone {
        let len_range = 11.0 + (1.0 - i as f32 / count as f32) * 1500.0;
        let comb_len = (48000.0 * (rng.gen_range(10.0..len_range) / 1000.0)) as usize;
        let scale_range = ((i as f32 + 1.0) / count as f32) * 3.0;
        let scale = rng.gen_range(-scale_range..scale_range);
        if i == 1 {
            curry(
                Constant(f32x8::splat(scale)),
                Comb::new(DelayLine::new(comb_len)),
            )
        } else {
            let inner = make_reverb(i - 1, count, rng);
            curry(
                Constant(f32x8::splat(scale)),
                Comb::new(DelayLine::new_nested(comb_len, inner)),
            )
        }
    }
    let reverb = Stack::new(
        make_reverb(comb_count, comb_count, &mut rng),
        make_reverb(comb_count, comb_count, &mut rng),
    );

    reverb
}

routing! {
    DrumRouting,
    gate: (f32x8,),
}

#[derive(Clone)]
pub struct BigDrum {
    gain: Box<dyn Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>>>,
    noise: Box<dyn Node<Input = Value<(f32x8,)>, Output = Value<(f32x8,)>>>,
    pub routing: DrumRouting,
}
impl Default for BigDrum {
    fn default() -> Self {
        let routing = DrumRouting::default();

        let f = move |t: f32, _off_time: Option<f32>| {
            let s = tweak!(5.0);
            let attack = tweak!(0.002) * s;
            let release = tweak!(0.025) * s;
            if t < attack {
                (t / attack).min(1.0)
            } else if t < attack + release {
                let t = t - attack;
                1.0 - (t / release).min(1.0)
            } else {
                0.0
            }
        };
        let gain = Pipe(
            ThreshEnvelope::new(f),
            curry(FnConstant(|| f32x8::splat(tweak!(2.0))), Mul::default()),
        );

        let f = move |t: f32, _off_time: Option<f32>| {
            let s = tweak!(2.0);
            let attack = tweak!(0.001) * s;
            let release = tweak!(0.003) * s;
            if t < attack {
                (t / attack).min(1.0)
            } else if t < attack + release {
                let t = t - attack;
                1.0 - (t / release).min(1.0)
            } else {
                0.0
            }
        };
        let cutoff = Pipe(
            ThreshEnvelope::new(f),
            FnRescale(|| (tweak!(20.0), tweak!(800.0))),
        );
        let filter = curry(
            Branch::new(
                cutoff,
                Pipe(Sink::default(), FnConstant(|| f32x8::splat(tweak!(3.0)))),
            ),
            Biquad::lowpass(),
        );

        //let r: &dyn Node<Input=Value<(f32x8, )>, Output=Value<(f32x8, f32x8)>> = &noise;
        let (b_in, b) = Bridge::new((f32x8::splat(0.0),));
        let mut noise = RetriggeringWaveTable::new(WaveTable::noise());
        noise.routing.gate(Pipe(b, Strip::default()));
        noise
            .routing
            .pitch(FnConstant(|| f32x8::splat(tweak!(880.0))));
        let noise = Pipe(
            b_in,
            Pipe(
                Pipe(
                    Concat::<
                        _,
                        Value<(f32x8,)>,
                        (NoValue, Value<(f32x8,)>),
                        (Value<(f32x8,)>, Value<(f32x8,)>),
                    >::new(noise),
                    Flip::default(),
                ),
                filter,
            ),
        );

        let low_filter = curry(
            Branch::new(
                FnConstant(|| f32x8::splat(tweak!(66.0))),
                FnConstant(|| f32x8::splat(tweak!(10.0))),
            ),
            Biquad::peaking_eq(Rc::new(RefCell::new((0.1,)))),
        );
        let noise = Pipe(noise, low_filter);

        //let noise = Pipe(noise, curry(FnConstant(|| f32x8::splat(tweak!(0.1))), Mul::default()));

        Self {
            routing,
            gain: Box::new(gain),
            noise: Box::new(noise),
        }
    }
}

impl Node for BigDrum {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.0.process(NoValue);
        let gain = *self.gain.process(gate).car();
        let noise = *self.noise.process(gate).car() * gain;
        Value((noise, noise))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.set_sample_rate(rate);
        self.gain.set_sample_rate(rate);
        self.noise.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct RetriggeringWaveTable {
    wave_table: WaveTable,
    triggered: bool,
    pub routing: BassRouting,
}
impl RetriggeringWaveTable {
    fn new(table: WaveTable) -> Self {
        let routing = BassRouting::default();
        Self {
            wave_table: table,
            triggered: false,
            routing,
        }
    }
}

impl Node for RetriggeringWaveTable {
    type Input = NoValue;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let gate = self.routing.gate.0.process(NoValue);
        let gate_max = gate.car().max_element();
        let pitch = self.routing.pitch.0.process(NoValue);
        if gate_max > 0.5 && !self.triggered {
            self.triggered = true;
            self.wave_table.idx = 0.0;
        } else if gate_max < 0.5 {
            self.triggered = false;
        }
        self.wave_table.process(pitch)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.set_sample_rate(rate);
        self.wave_table.set_sample_rate(rate);
    }
}

routing! {
    PlinkRouting,
    gate: (f32x8,),
    pitch1: (f32x8,),
    pitch2: (f32x8,),
    pitch3: (f32x8,),
    pitch4: (f32x8,),

    q1: (f32x8,),
    q2: (f32x8,),
    q3: (f32x8,),
    q4: (f32x8,),

    eq_1_gain: (f32,),
    eq_2_gain: (f32,),
}
#[derive(Clone)]
pub struct ExcitablePlink {
    pub routing: PlinkRouting,
    output: Box<dyn Node<Input=NoValue, Output=Value<(f32x8,)>>>
}

impl Default for ExcitablePlink {
    fn default() -> Self {
        let mut routing = PlinkRouting::default();

        routing.pitch1(Constant(f32x8::splat(100.0)));
        routing.pitch2(Constant(f32x8::splat(1000.0)));
        routing.pitch3(Constant(f32x8::splat(80.0)));
        routing.pitch4(Constant(f32x8::splat(60.0)));

        routing.pitch1.0.process(NoValue);
        routing.pitch2.0.process(NoValue);
        routing.pitch3.0.process(NoValue);
        routing.pitch4.0.process(NoValue);

        routing.q1(Constant(f32x8::splat(1.0)));
        routing.q2(Constant(f32x8::splat(10.0)));
        routing.q3(Constant(f32x8::splat(20.0)));
        routing.q4(Constant(f32x8::splat(10.0)));

        routing.q1.0.process(NoValue);
        routing.q2.0.process(NoValue);
        routing.q3.0.process(NoValue);
        routing.q4.0.process(NoValue);

        routing.eq_1_gain(Constant(5.0));
        routing.eq_2_gain(Constant(5.0));

        let (b_in, b) = Bridge::<Value<(f32x8,)>>::new((f32x8::splat(0.0),));
        let feedback_gain = FnConstant(|| f32x8::splat(0.0005));
        let filter_a = curry(Branch::new(routing.pitch1_bridge(), routing.q1_bridge()), Biquad::bandpass());
        let filter_b = curry(Branch::new(routing.pitch2_bridge(), routing.q2_bridge()), Biquad::bandpass());
        let filter = Pipe(Branch::new(filter_a, filter_b), Add::default());
        let plink = Pipe(Pipe(routing.gate_bridge(), curry(Pipe(Pipe(b, Strip::default()), curry(feedback_gain, Mul::default())), Add::default())), filter);
        let filter = curry(Branch::new(routing.pitch3_bridge(), routing.q3_bridge()), Biquad::peaking_eq(Rc::clone(&routing.eq_1_gain.1.0)));
        let plink = Pipe(plink, filter);
        let filter = curry(Branch::new(routing.pitch4_bridge(), routing.q4_bridge()), Biquad::peaking_eq(Rc::clone(&routing.eq_2_gain.1.0)));
        let plink = Pipe(plink, filter);
        let plink = Pipe(plink, BFunc(|v:f32x8| v.tanh()));

        let ping = Pipe(Pipe(plink, b_in), curry(FnConstant(|| f32x8::splat(0.7)), Mul::<f32x8, f32x8>::default()));
        let ping = Pipe(ping, BFunc(|v:f32x8| v.tanh()));

        Self {
            routing,
            output: Box::new(ping),
        }
    }
}

impl Node for ExcitablePlink {
    type Input = NoValue;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.routing.process();
        self.output.process(NoValue)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.set_sample_rate(rate);
        self.output.set_sample_rate(rate);
    }
}

routing! {
    SoftMachineRouting,
    gate: (f32,),
    lock: (f32,),
    len: (u32,),
}
#[derive(Clone)]
pub struct SoftTuringMachine {
    pub routing: SoftMachineRouting,
    sequence: Vec<Option<f32>>,
    pulse_outs: HashMap<usize, Rc<RefCell<f32>>>,
    idx: usize,
    triggered: bool,
}

impl Default for SoftTuringMachine {
    fn default() -> Self {
        let mut routing = SoftMachineRouting::default();
        Self {
            routing,
            sequence: vec![None],
            pulse_outs: HashMap::new(),
            idx: 0,
            triggered: false,
        }
    }
}
impl SoftTuringMachine {
   pub fn pulse(&mut self, idx: usize) -> Rc<RefCell<f32>> {
        self.pulse_outs.entry(idx).or_insert_with(|| Rc::new(RefCell::new(0.0))).clone()
    }
}
impl Node for SoftTuringMachine {
    type Input = NoValue;
    type Output = Value<(f32,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        let mut rng = thread_rng();
        let gate = *self.routing.gate.0.process(NoValue).car();
        let lock = *self.routing.lock.0.process(NoValue).car();
        let len = *self.routing.len.0.process(NoValue).car();
        for cell in self.pulse_outs.values_mut() {
            *cell.borrow_mut() = 0.0;
        }
        if gate > 0.5 {
            if !self.triggered {
                self.triggered = true;
                self.idx += 1;
                self.idx = self.idx % len as usize;
                while self.sequence.len() <= self.idx {
                    //if rng.gen::<f32>() > 0.5 {
                        self.sequence.push(Some(rng.gen()));
                    //} else {
                    //    self.sequence.push(None);
                    //}
                }
                if rng.gen::<f32>() > lock {
                    //if rng.gen::<f32>() > 0.5 {
                        self.sequence[self.idx] = Some(rng.gen());
                    //}
                }
                for (i, cell) in &mut self.pulse_outs {
                    let idx = (self.idx + i) % len as usize;
                    if idx < self.sequence.len() {//&& self.sequence[idx].is_some() {
                        *cell.borrow_mut() = 1.0;
                    }
                }
            }
        } else {
            self.triggered = false;
        }
        self.idx = self.idx % len as usize;
        while self.sequence.len() <= self.idx {
            self.sequence.push(rng.gen());
        }
        let r = self.sequence[self.idx].unwrap_or(0.0);
        Value((r,))
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.routing.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct ShiftRegister {
    pub registers: Vec<Rc<RefCell<f32>>>,
    triggered: bool,
    initialized: bool,
}
impl ShiftRegister {
    pub fn new() -> Self {
        Self {
            registers: (0..8).map(|_| Rc::new(RefCell::new(0.0))).collect(),
            triggered: false,
            initialized: false,
        }
    }
}
impl Node for ShiftRegister {
    type Input = Value<(f32x8, f32x8)>;
    type Output = Value<(f32x8,)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
let mut r = f32x8::splat(0.0);
         for i in 0..f32x8::lanes() {
             let gate = (input.0).0.extract(i);
             let signal = (input.0).1.extract(i);
             if !self.triggered && gate > 0.5 || !self.initialized {
                 for i in 1..self.registers.len() {
                     let i = self.registers.len()-i;
                     let j = i-1;
                     let v = *self.registers[j].borrow();
                     *self.registers[i].borrow_mut() = v;
                 }
                 *self.registers[0].borrow_mut() = signal;
                 self.triggered = true;
                 self.initialized = true;
             } else if gate < 0.5 {
                 self.triggered = false;
             }
             r = unsafe { r.replace_unchecked(i, *self.registers[0].borrow()) };
         }
         Value((r,))
    }
}
