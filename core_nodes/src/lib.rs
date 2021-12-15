use std::collections::HashMap;
use virtual_modular_graph::NodeTemplate;

#[cfg(feature = "abc")]
pub mod abc;
pub mod arithmetic;
pub mod clock;
pub mod computation;
pub mod delay_and_reverb;
pub mod distortion;
pub mod envelope;
pub mod filter;
pub mod misc;
pub mod oscillator;
pub mod performance;
pub mod quantizer;
pub mod randomization;
pub mod sequencer;
pub mod topological;
#[cfg(feature = "tuning")]
pub mod tuning;
pub mod waveguide;

pub use arithmetic::*;
pub use clock::*;
pub use computation::*;
pub use delay_and_reverb::*;
pub use distortion::*;
pub use envelope::*;
pub use filter::*;
pub use misc::*;
pub use oscillator::*;
pub use performance::*;
pub use quantizer::*;
pub use randomization::*;
pub use sequencer::*;
pub use topological::*;
pub use waveguide::*;

#[macro_export]
macro_rules! node_templates {
    ($loader_name:ident {$($name:ident: $constructor:expr),*}) => {
        pub fn $loader_name() -> HashMap<String, NodeTemplate> {
            let mut templates = HashMap::new();
            $(
                let template = NodeTemplate {
                    node: ($constructor).into(),
                    code: stringify!($constructor).to_string()
                };
                templates.insert(stringify!($name).to_string(), template);
            )*
            templates
        }
    }
}

node_templates! {
    std_nodes {
        ToneHoleFlute: ToneHoleFlute::default(),
        Add: Add,
        Pow: Pow,
        SoftClip:SoftClip,
        Sub: Sub,
        Mul: Mul,
        Div: Div,
        And: And,
        Or: Or,
        Xor: Xor,
        Nand: Nand,
        Imp: Impulse::default(),
        RMS: RMS::default(),
        Compressor: Compressor::default(),
        CXor: CXor,
        Comp: Comparator,
        Sine: Sine::default(),
        TanhShaper: TanhShaper::default(),
        Psine: PositiveSine::default(),
        Saw: WaveTable::saw(),
        PulseOnLoad: PulseOnLoad::default(),
        Noise: Noise::default(),
        PNoise: Noise::positive(),
        Ad: ADEnvelope::default(),
        Adsr: ADSREnvelope::default(),
        QImp: QuantizedImpulse::default(),
        AllPass: AllPass::default(),
        Sh: SampleAndHold::default(),
        Pd: PulseDivider::default(),
        Log: Log,
        LogTrigger: LogTrigger::default(),
        Acc: Accumulator::default(),
        Pan: Pan,
        MonoPan: MonoPan,
        MidSideDecoder: MidSideDecoder,
        MidSideEncoder: MidSideEncoder,
        StereoIdentity: StereoIdentity,
        Rescale: ModulatedRescale,
        Lp: SimperLowPass::default(),
        Hp: SimperHighPass::default(),
        Bp: SimperBandPass::default(),
        Toggle: Toggle::default(),
        Slew: Slew::default(),
        Reverb: Reverb::default(),
        Delay: ModableDelay::default(),
        PingPong: ModablePingPong::default(),
        Bg: BernoulliGate::default(),
        C: Identity,
        QuadSwitch: QuadSwitch::default(),
        Seq: PatternSequencer::default(),
        Burst: BurstSequencer::default(),
        TapsAndStrikes: TapsAndStrikes::default(),
        Folder: Folder,
        EuclidianPulse: EuclidianPulse::default(),
        PulseOnChange: PulseOnChange::default(),
        Brownian: Brownian::default(),
        MajorKeyMarkov: Markov::major_key_chords(),
        Quantizer: DegreeQuantizer::default(),
        BowedString: BowedString::default(),
        PluckedString: PluckedString::default(),
        ImaginaryGuitar: ImaginaryGuitar::default(),
        EvenHarmonicDistortion: EvenHarmonicDistortion,
        SympatheticString: SympatheticString::default(),
        WaveMesh: WaveMesh::default(),
        PennyWhistle: PennyWhistle::default(),
        StereoIdentity: StereoIdentity,
        StringBodyFilter: StringBodyFilter::default(),
        Constant: Constant::default()
    }
}
