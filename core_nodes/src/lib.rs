use generic_array::{arr, typenum::*};
use std::collections::HashMap;
use virtual_modular_graph::{Node, NodeTemplate, Ports, BLOCK_SIZE};

pub mod utils;
#[cfg(feature = "abc")]
pub mod abc;
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
        Add: FnNode::new(|(a, b)| a+b),
        Sub: FnNode::new(|(a, b)| a-b),
        Mul: FnNode::new(|(a, b)| a*b),
        Div: FnNode::new(|(a, b)| a/b),
        Pow: FnNode::new(|(a, b):(f32, f32)| a.powf(b)),
        And: FnNode::new(|(a, b)| if a > 0.5 && b > 0.5 { 1.0 } else { 0.0 }),
        Or: FnNode::new(|(a, b)| if a > 0.5 || b > 0.5 { 1.0 } else { 0.0 }),
        Xor: FnNode::new(|(a, b)| {
            let a = a > 0.5;
            let b = b > 0.5;
            if (a || b) && !(a && b) { 1.0 } else { 0.0 }
        }),
        Nand: FnNode::new(|(a, b)| {
            let a = a > 0.5;
            let b = b > 0.5;
            if !(a && b) { 1.0 } else { 0.0 }
        }),
        SineS: FnNode::new(|(a,):(f32,)| a.sin()),
        ASineS: FnNode::new(|(a,):(f32,)| a.asin()),
        TanhS: FnNode::new(|(a,):(f32,)| a.tanh()),
        LogS: FnNode::new(|(a,):(f32,)| a.log2()),
        Imp: Impulse::default(),
        NCube: NCube::default(),
        RMS: RMS::default(),
        Compressor: Compressor::default(),
        CXor: CXor,
        Comp: Comparator,
        Sine: SineWave::default(),
        Triangle: TriangleWave::default(),
        Square: SquareWave::default(),
        TanhShaper: TanhShaper::default(),
        Psine: PositiveSineWave::default(),
        Saw: SawWave::default(),
        PulseOnLoad: PulseOnLoad::default(),
        Noise: Noise::default(),
        PNoise: Noise::positive(),
        Ad: ADEnvelope::default(),
        Adsr: ADSREnvelope::default(),
        QImp: QuantizedImpulse::default(),
        AllPass: AllPass::default(),
        Comb: Comb::default(),
        ParallelCombs: ParallelCombs::default(),
        SerialAllPasses: SerialAllPasses::default(),
        Diffusor: Diffusor::new(0.06, &mut rand::thread_rng()),
        Reverb2: Reverb2::default(),
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
        Reverb: Reverb::poop(),
        Delay: ModableDelay::default(),
        PingPong: ModablePingPong::default(),
        Bg: BernoulliGate::default(),
        C: Identity,
        QuadSwitch: QuadSwitch::default(),
        Seq: PatternSequencer::default(),
        StepSeq: StepSequencer::default(),
        Burst: BurstSequencer::default(),
        Choice: Choice::default(),
        BurstTrigger: BurstTrigger::default(),
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

#[derive(Clone)]
pub struct FnNode<A: Clone, R: Clone>(R, std::marker::PhantomData<A>);
impl<A: Clone, R: FnMut(A) -> f32 + Clone> FnNode<A, R> {
    pub fn new(func: R) -> Self {
        Self(func, Default::default())
    }
}

impl<F: FnMut() -> f32 + Clone> Node for FnNode<(), F> {
    type Input = U0;
    type Output = U1;

    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut r = [0.0; BLOCK_SIZE];
        for r in r.iter_mut() {
            *r = self.0();
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

impl<F: FnMut((f32,)) -> f32 + Clone> Node for FnNode<(f32,), F> {
    type Input = U1;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let a = input[0];
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            *r = self.0((a[i],));
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}

impl<F: FnMut((f32, f32)) -> f32 + Clone> Node for FnNode<(f32, f32), F> {
    type Input = U2;
    type Output = U1;

    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let a = input[0];
        let b = input[1];
        let mut r = [0.0; BLOCK_SIZE];
        for (i, r) in r.iter_mut().enumerate() {
            *r = self.0((a[i], b[i]));
        }
        arr![[f32; BLOCK_SIZE]; r]
    }
}
