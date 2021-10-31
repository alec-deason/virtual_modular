use dyn_clone::DynClone;
use std::{
    sync::{Mutex,Arc},
    collections::{HashSet, HashMap},
    cell::RefCell,
    convert::{TryFrom, TryInto},
};

use packed_simd_2::f32x8;

use generic_array::{ArrayLength, typenum::{ToInt, U0, U1, U2}, arr};
use crate::{
    simd_graph::{Node, Ports},
};
use crate::{simd_graph::*, voices::*};

#[macro_export]
macro_rules! dynamic_nodes {
    ($loader_name:ident {$($name:ident: $constructor:expr),*}) => {
        #[macro_export]
        macro_rules! $loader_name {
            ($builder:expr) => {
                $(
                    println!("{}", stringify!($name));
                    $builder.templates.insert(stringify!($name).to_string(), (BoxedDynamicNode::new($constructor), stringify!($constructor).to_string()));
                )*
            }
        }
    }
}

dynamic_nodes!{
    std_nodes {
        Add: Add,
        Sub: Sub,
        Mul: Mul,
        Div: Div,
        Imp: Impulse::default(),
        CXor: CXor,
        Comp: Comparator,
        Sine: WaveTable::sine(),
        Psine: WaveTable::positive_sine(),
        Saw: WaveTable::saw(),
        PulseOnLoad: PulseOnLoad::default(),
        Noise: Noise::default(),
        PNoise: Noise::positive(),
        Ad: InlineADEnvelope::default(),
        Adsr: InlineADSREnvelope::default(),
        QImp: QuantizedImpulse::default(),
        AllPass: AllPass::default(),
        Sh: SampleAndHold::default(),
        Pd: PulseDivider::default(),
        Log: Log,
        Pan: Pan,
        MidSideDecoder: MidSideDecoder,
        MidSideEncoder: MidSideEncoder,
        Rescale: ModulatedRescale,
        Svfl: SimperSvf::low_pass(),
        Svfh: SimperSvf::high_pass(),
        Svfb: SimperSvf::band_pass(),
        Svfn: SimperSvf::notch(),
        Looper: Looper::default(),
        Toggle: Toggle::default(),
        Portamento: Portamento::default(),
        Reverb: Reverb::new(),
        Delay: ModableDelay::new(),
        Bg: BernoulliGate::default(),
        C: Identity,
        QuadSwitch: QuadSwitch::default(),
        Folder: Folder,
        EuclidianPulse: EuclidianPulse::default(),
        PulseOnChange: PulseOnChange::default(),
        Brownian: Brownian::default(),
        LeadVoice: lead_voice(),
        Compressor: SidechainCompressor::new(0.3),
        MajorKeyMarkov: Markov::major_key_chords(),
        ScaleMajor: Quantizer::new(&[16.351875, 18.35375, 20.601875, 21.826875, 24.5, 27.5, 30.8675, 32.703125]),
        ScaleDegreeMajor: DegreeQuantizer::new(&[16.351875, 18.35375, 20.601875, 21.826875, 24.5, 27.5, 30.8675, 32.703125]),
        ScaleDegreeMinor: DegreeQuantizer::new(&[18.35,20.60,21.83,24.50,27.50,29.14, 32.70]),
        BowedString: BowedString::default(),
        PluckedString: PluckedString::default(),
        ImaginaryGuitar: ImaginaryGuitar::default(),
        SympatheticString: SympatheticString::default(),
        WaveMesh: WaveMesh::default(),
        Flute: Flute::default(),
        StringBodyFilter: StringBodyFilter::default()
    }
}

#[derive(Clone)]
pub struct DynamicGraphBuilder {
    pub templates: HashMap<String, (BoxedDynamicNode, String)>,
}

impl Default for DynamicGraphBuilder {
    fn default() -> Self {
        let mut b = Self {
            templates: Default::default()
        };
        std_nodes!(&mut b);
        b
    }
}

impl DynamicGraphBuilder {
    pub fn parse(&self, data: &str) -> Result<DynamicGraph, String> {
        let mut g = DynamicGraph::default();
        let l = self.parse_inner(data)?;

        g.reparse(&l)?;
        Ok(g)
    }

    pub fn parse_inner(&self, data: &str) -> Result<Vec<Line>, String> {
        use pom::parser::*;
        use pom::parser::Parser;
        use std::str::{self, FromStr};

        fn node_name<'a>() -> Parser<'a, u8, String> {
            let number = one_of(b"0123456789");
            (one_of(b"abcdefghijklmnopqrstuvwxyzxy").repeat(1) + (one_of(b"abcdefghijklmnopqrstuvwxyzxy")| number | sym(b'_')).repeat(0..)).collect().convert(str::from_utf8).map(|s| s.to_string())
        }
        fn node_constructor_name<'a>() -> Parser<'a, u8, String> {
            let lowercase = one_of(b"abcdefghijklmnopqrstuvwxyzxy");
            let uppercase = one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            let number = one_of(b"0123456789");
            (uppercase.repeat(1) + (lowercase | one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") | number | sym(b'_')).repeat(0..)).collect().convert(str::from_utf8).map(|s| s.to_string())
        }

        fn constructor_call<'a>() -> Parser<'a, u8, Expression> {
            let parameters = (sym(b'(') * whitespace() * list(call(expression), whitespace() * sym(b',') - whitespace()) - whitespace() - sym(b')')).opt();

            let r = node_constructor_name() + parameters;

            r.map(|(constructor_name, parameters)| {
                Expression::Term(Term::NodeConstructor(constructor_name, parameters))
            })
        }

        fn term<'a>() -> Parser<'a, u8, Expression> {
            let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
            let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
            let number = sym(b'-').opt() + integer + frac.opt();
            let number = number
                .collect()
                .map(|cs| String::from_utf8(cs.to_vec()).unwrap().parse::<f32>().unwrap())
                .map(|n| Expression::Term(Term::Number(n)));
            let port_number = one_of(b"0123456789").repeat(1..).collect().map(|cs| String::from_utf8(cs.to_vec()).unwrap().parse::<usize>().unwrap());
            let node_reference = (node_name() + (sym(b'|') * port_number).opt()).map(|(node, c)| Expression::Term(Term::Node(node, c.unwrap_or(0))));
            let constructor = constructor_call();
            let term = number | node_reference | constructor;
            term
        }
        fn grouped_expression<'a>() -> Parser<'a, u8, Expression> {
            call(term) | (sym(b'(') * call(expression) - sym(b')'))
        }
        fn expression<'a>() -> Parser<'a, u8, Expression> {
            let operator = one_of(b"+-*/").repeat(1).collect().convert(str::from_utf8);

            let r = (call(grouped_expression) - whitespace() + operator - whitespace() + call(grouped_expression)).convert::<_,&'static str,_>(|((a, o), b)| Ok(Expression::Operation(Box::new(a), o.try_into()?, Box::new(b)))) | call(term);
            r
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
        }

        fn interpolation<'a>() -> Parser<'a, u8, Interpolation> {
            let constant = seq(b"c(") * number() - sym(b',') + number() - sym(b')');
            let constant = constant.map(|(value, duration)| Interpolation::Constant { value , duration });
            let linear = seq(b"l(") * number() - sym(b',') + number() - sym(b',') + number() - sym(b')');
            let linear = linear.map(|((start, end), duration)| Interpolation::Linear{ start, end, duration });
            constant | linear
        }

        fn automate<'a>() -> Parser<'a, u8, Vec<Line>> {

            let name = (none_of(b"\n=[](),")).repeat(1..).convert(String::from_utf8) - sym(b'=');
            let r = name + sym(b'[') * list(interpolation(), sym(b',')) - sym(b']');
            r.map(|(k, es)| {
                vec![Line::Node(k, "automation".to_string(), Some(NodeParameters::AutomationSequence(es)))]
            })
        }

        fn sumseq<'a>() -> Parser<'a, u8, Vec<Line>> {
            let name = (none_of(b"\n=[](),")).repeat(1..).convert(String::from_utf8) - seq(b"=sumseq");
            let r = name + sym(b'[') * list(sym(b'(') * number() - sym(b',') + number() - sym(b')'), sym(b',')) - sym(b']');
            let parameter = (sym(b'(') * list(expression(), sym(b',')) - sym(b')')).opt();
            let r = r + parameter;
            r.map(|((k, ns), p)| {
                let ns:Vec<_> = ns.into_iter().map(|(c,p)| (c as u32, p)).collect();
                let ns:[(u32, f32); 4] = ns.try_into().unwrap();
                let mut edges:Vec<_> = if let Some(p) = p {
                    p.iter().enumerate().map(|(i,e)| {
                        e.as_lines(k.clone(), i)
                    }).flatten().collect()
                } else {
                    vec![]
                };
                edges.push(Line::Node(k, "sum_sequencer".to_string(), Some(NodeParameters::SumSequence(ns))));
                edges
            })
        }

        fn mn_modified_event<'a>() -> Parser<'a, u8, RawSubsequence> {
            (call(mn_raw_event) + one_of(b"*/!") + number()).convert(|((e, o), n)| {
                match o {
                    b'*' => Ok(RawSubsequence::ClockMultiplier(Box::new(e),1.0/n)),
                    b'/' => Ok(RawSubsequence::ClockMultiplier(Box::new(e),n)),
                    b'!' => Ok(RawSubsequence::ClockMultiplier(Box::new(RawSubsequence::Tuplet((0..n.max(1.0) as usize).map(|_| e.clone()).collect(), 0)),1.0/n)),
                    _ => Err(()),
                }
            })
        }

        fn mn_raw_event<'a>() -> Parser<'a, u8, RawSubsequence> {
            call(mn_bracketed_pattern) | sym(b'~').map(|_| RawSubsequence::Rest(0.0)) | number().map(|n| RawSubsequence::Item(n, 0.0))
        }

        fn mn_event<'a>() -> Parser<'a, u8, RawSubsequence> {
            call(mn_modified_event) | call(mn_raw_event)
        }

        fn mn_sequence<'a>() -> Parser<'a, u8, Vec<RawSubsequence>> {
            list(call(mn_event), sym(b' ').repeat(1..))
        }

        fn mn_repeat<'a>() -> Parser<'a, u8, RawSubsequence> {
            (sym(b'[') * whitespace() * call(mn_sequence) - whitespace() - sym(b']'))
            .convert(|s| {
                if s.is_empty() {
                    Err(())
                } else {
                    Ok(RawSubsequence::Tuplet(s, 0))
                }
            })
        }

        fn mn_cycle<'a>() -> Parser<'a, u8, RawSubsequence> {
            (sym(b'<') * whitespace() * call(mn_sequence) - whitespace() - sym(b'>'))
            .convert(|s| {
                if s.is_empty() {
                    Err(())
                } else {
                    Ok(RawSubsequence::Iter(s, 0))
                }
            })
        }

        fn mn_choice<'a>() -> Parser<'a, u8, RawSubsequence> {
            (sym(b'|') * whitespace() * call(mn_sequence) - whitespace() - sym(b'|'))
            .convert(|s| {
                if s.is_empty() {
                    Err(())
                } else {
                    Ok(RawSubsequence::Choice(s, 0))
                }
            })
        }

        fn mn_nested<'a>() -> Parser<'a, u8, RawSubsequence> {
            node_name().map(|s| {
                RawSubsequence::Nested(s)
            })
        }

        fn mn_bracketed_pattern<'a>() -> Parser<'a, u8, RawSubsequence> {
            call(mn_repeat) | call(mn_cycle) | call(mn_choice) | call(mn_nested)
        }

        fn sequencer<'a>() -> Parser<'a, u8, Vec<Line>> {
            let name = node_name() - seq(b"=seq");
            let r = name + mn_bracketed_pattern();
            let parameter = sym(b'(') * expression() - sym(b')');
            let r = r + parameter.opt();
            r.map(|((node_name, sub_sequence), clock)| {
                let mut edges = Vec::with_capacity(3);
                if let Some(clock) = clock {
                    edges.extend(clock.as_lines(node_name.clone(), 0));
                }
                edges.push(Line::Node(node_name, "pattern_sequencer".to_string(), Some(NodeParameters::PatternSequence(sub_sequence))));
                edges
            })
        }
        fn external_parameter<'a>() -> Parser<'a, u8, Vec<Line>> {
            (none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b'=') - seq(b"e{") + none_of(b"}").repeat(1..).convert(String::from_utf8) - sym(b'}')).map(|(n,k)| {
                vec![Line::ExternalParam(n, k)]
            })
        }

        fn comment<'a>() -> Parser<'a, u8, Vec<Line>> {
            let comment = (sym(b'#') * none_of(b"\n").repeat(0..)) - sym(b'\n');
            let empty_line = (sym(b'\n')).repeat(1);
            (comment | empty_line).map(|_| vec![Line::Comment])
        }
        fn bridge<'a>() -> Parser<'a, u8, Vec<Line>> {
            let n_in = seq(b"b{") * none_of(b",").repeat(1..).convert(String::from_utf8) - sym(b',');
            let n_out = none_of(b"}").repeat(1..).convert(String::from_utf8) - sym(b'}');
            (n_in+n_out).map(|(n_in,n_out)| {
                vec![
                    Line::BridgeNode(n_in, n_out)
                ]
            })
        }
        fn synth<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = (comment() | bridge() | external_parameter() | sequencer() | sumseq() | node_definition() | automate() | edge() | node()).repeat(1..);
            p.map(|ls| ls.into_iter().flatten().collect())
        }
        fn node_definition<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = (none_of(b"\n={,")).repeat(1..).convert(String::from_utf8) - sym(b'{') + call(synth) - sym(b'}') - sym(b'\n');
            p.map(|(name, lines)| vec![Line::NodeDefinition(name, lines)])
        }
        fn node<'a>() -> Parser<'a, u8, Vec<Line>> {
            (node_name() - whitespace() - sym(b'=') - whitespace() + expression() - whitespace() - sym(b'\n')).map(|(node_name, expression)| {
                match expression {
                    Expression::Term(Term::NodeConstructor(n,p)) => {
                        let mut edges:Vec<_> = if let Some(p) = p {
                            p.iter().enumerate().map(|(i,e)| {
                                e.as_lines(node_name.clone(), i)
                            }).flatten().collect()
                        } else {
                            vec![]
                        };

                        edges.push(Line::Node(node_name, n, None));
                        edges
                    }
                    _ => {
                        let mut edges = expression.as_lines(node_name.clone(), 0);
                        edges.push(Line::Node(node_name, "C".to_string(), None));
                        edges
                    }
                }
            })
        }
        //fn include<'a>() -> Parser<'a, u8, Vec<Line>> {
        //    let p = seq(b"include(") * none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b')') - whitespace() - sym(b'\n');
        //    p.convert::<_, String, _>(|k| {
        //        let data = std::fs::read_to_string(k.clone()).unwrap();
        //        Ok(self.parse_inner(&data)?.into_iter().chain(vec![Line::Include(k)]).collect())
        //    })
        //}
        fn whitespace<'a>() -> Parser<'a, u8, Vec<u8>> {
            sym(b' ').repeat(0..)
        }
        fn edge<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = sym(b'(') * whitespace() *
                none_of(b"),").repeat(1..).convert(String::from_utf8) - whitespace() - sym(b',') - whitespace() +
                one_of(b"0123456789").repeat(1..).convert(String::from_utf8).convert(|s|u32::from_str(&s)) - whitespace() - sym(b',') - whitespace() +
                expression()
                - whitespace() - sym(b')') - whitespace() - sym(b'\n');
            p.map(|((dst, i), e)| {
                e.as_lines(dst, i as usize)
            })
        }
        synth().parse(data.as_bytes()).map_err(|e| format!("{:?}", e))
    }
}

pub trait DynamicNode: DynClone {
    fn process(&mut self);
    fn post_process(&mut self);
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn get(&self, i:usize) -> f32x8;
    fn set(&mut self, i:usize, v: f32x8);
    fn add(&mut self, i:usize, v: f32x8);
    fn set_sample_rate(&mut self, rate: f32);
}
dyn_clone::clone_trait_object!(DynamicNode);

impl<A: ArrayLength<f32x8> + ToInt<usize> + Clone, B: ArrayLength<f32x8> + ToInt<usize> + Clone, N: Node<Input=A, Output=B> + Clone> DynamicNode for (N, RefCell<Ports<A>>, RefCell<Ports<B>>) {
    fn process(&mut self) {
        *self.2.borrow_mut() = self.0.process(self.1.borrow().clone());
    }

    fn post_process(&mut self) {
        self.0.post_process();
    }

    fn input_len(&self) -> usize {
        A::to_int()
    }

    fn output_len(&self) -> usize {
        B::to_int()
    }

    fn get(&self, i:usize) -> f32x8 {
        self.2.borrow()[i]
    }

    fn set(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut()[i] = v;
    }

    fn add(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut()[i] += v;
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone, Debug)]
pub enum Term {
    Node(String, usize),
    NodeConstructor(String, Option<Vec<Expression>>),
    Number(f32),
}
#[derive(Clone, Debug)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div
}
impl TryFrom<&str> for Operator {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(Operator::Add),
            "-" => Ok(Operator::Sub),
            "*" => Ok(Operator::Mul),
            "/" => Ok(Operator::Div),
            _ => Err("Unknown operator")
        }
    }
}
#[derive(Clone, Debug)]
pub enum Expression {
    Term(Term),
    Operation(Box<Expression>, Operator, Box<Expression>),
}
impl Expression {
    fn as_lines(&self, target_node: String, target_port: usize) -> Vec<Line> {
        let mut lines = vec![];
        match self {
            Expression::Term(t) => {
                match t {
                    Term::Node(n, c) => {
                        lines.push(Line::Edge(n.clone(), *c as u32, target_node, target_port as u32));
                    }
                    Term::NodeConstructor(constructor_name, parameters) => {
                        let n = uuid::Uuid::new_v4().to_string();
                        lines.push(Line::Node(n.clone(), constructor_name.clone(), None));
                        if let Some(parameters) = parameters {
                            for (i, e) in parameters.iter().enumerate() {
                                lines.extend(e.as_lines(n.clone(), i));
                            }
                        }
                        lines.push(Line::Edge(n, 0, target_node, target_port as u32));
                    }
                    Term::Number(v) => {
                        let n = uuid::Uuid::new_v4().to_string();
                        lines.push(Line::Node(n.clone(), "Constant".to_string(), Some(NodeParameters::Number(*v))));
                        lines.push(Line::Edge(n, 0, target_node, target_port as u32));
                    }
                }
            }
            Expression::Operation(a, o, b) => {
                let n = uuid::Uuid::new_v4().to_string();
                lines.extend(a.as_lines(n.clone(), 0));
                lines.extend(b.as_lines(n.clone(), 1));
                match o {
                    Operator::Add => lines.push(Line::Node(n.clone(), "Add".to_string(), None)),
                    Operator::Sub => lines.push(Line::Node(n.clone(), "Sub".to_string(), None)),
                    Operator::Mul => lines.push(Line::Node(n.clone(), "Mul".to_string(), None)),
                    Operator::Div => lines.push(Line::Node(n.clone(), "Div".to_string(), None)),
                }
                lines.push(Line::Edge(n, 0, target_node, target_port as u32));
            }
        }
        lines
    }
}

impl DynamicNode for (DynamicGraph, RefCell<[f32x8; 2]>) {
    fn process(&mut self) {
        let r = self.0.process();
        let mut o = self.1.borrow_mut();
        o[0] = r[0];
        o[1] = r[1];
    }

    fn post_process(&mut self) {
        self.0.post_process();
    }

    fn input_len(&self) -> usize {
        self.0.input.len()
    }

    fn output_len(&self) -> usize {
        2
    }

    fn get(&self, i:usize) -> f32x8 {
        self.1.borrow()[i]
    }

    fn set(&mut self, i:usize, v: f32x8) {
        self.0.input[i] = v;
    }

    fn add(&mut self, i:usize, v: f32x8) {
        self.0.input[i] += v;
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct BoxedDynamicNode(Box<dyn DynamicNode>);

impl std::fmt::Debug for BoxedDynamicNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedDynamicNode")
        .finish()
    }
}
impl BoxedDynamicNode {
        pub fn new<A: ArrayLength<f32x8> + ToInt<usize> + Clone + 'static, B: ArrayLength<f32x8> + ToInt<usize> + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(n: N) -> Self {
        let a = Ports::default();
        let b = Ports::default();
        Self(Box::new((n, RefCell::new(a), RefCell::new(b))))
    }
}
impl DynamicNode for BoxedDynamicNode {
    fn process(&mut self) {
        self.0.process();
    }

    fn post_process(&mut self) {
        self.0.post_process();
    }

    fn input_len(&self) -> usize {
        self.0.input_len()
    }

    fn output_len(&self) -> usize {
        self.0.output_len()
    }

    fn get(&self, i:usize) -> f32x8 {
        self.0.get(i)
    }

    fn set(&mut self, i:usize, v: f32x8) {
        self.0.set(i, v);
    }

    fn add(&mut self, i:usize, v: f32x8) {
        self.0.add(i, v);
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: HashMap<String, (BoxedDynamicNode, String)>,
    builder: DynamicGraphBuilder,
    dynamic_nodes: HashMap<String, Box<DynamicGraph>>,
    pub external_inputs: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    external_input_nodes: HashMap<String, (String, f32)>,
    pub watch_list: Arc<Mutex<HashSet<String>>>,
    input: Vec<f32x8>,
    output: (f32x8, f32x8),
    edges: HashMap<String, Vec<(usize, String, usize)>>,
    reset_edges: Vec<(String,usize)>,
    topo_sort: Vec<String>,
    pub reload_data: Arc<Mutex<Option<String>>>,
    sample_rate: f32,
}

impl DynamicGraph {
    pub fn process(&mut self) -> Ports<U2> {
        if let Ok(mut ei) = self.external_inputs.lock() {
            for (k, v) in self.external_input_nodes.values_mut() {
                if let Some(nv) = ei.get_mut(k).and_then(|v| v.pop()) {
                    *v = nv;
                }
            }
        }
        self.output = (f32x8::splat(0.0), f32x8::splat(0.0));
        /*
        for (n,v) in &self.reset_edges {
            if let Some(node) = self.nodes.get_mut(n) {
                node.0.set(*v, f32x8::splat(0.0));
            } else {
                self.dynamic_nodes.get_mut(n).expect(&format!("no definition for {}",n)).input[*v] = f32x8::splat(0.0);
            }
        }
        */
        for (n,_) in self.nodes.values_mut() {
            for i in 0..n.input_len() {
                n.set(i, f32x8::splat(0.0));
            }
        }
        for n in self.dynamic_nodes.values_mut() {
            for i in 0..n.input.len() {
                n.input[i] = f32x8::splat(0.0);
            }
        }

        for node in &self.topo_sort {
            if let Some(others) = self.edges.get(node) {
                for (dst_i, n, src_i) in others {
                    let v = self.get(n, *src_i).unwrap();
                    if node == "output" {
                        if *dst_i == 0 {
                            self.output.0 += v;
                        } else {
                            self.output.1 += v;
                        }
                    } else {
                        if let Some(n) = self.nodes.get_mut(node) {
                            n.0.add(*dst_i, v);
                        } else {
                            self.dynamic_nodes.get_mut(node).expect(&format!("no definition for {}",node)).input[*dst_i] += v;
                        }
                    }
                }
            }
            if node != "output" {
                if let Some(n) = self.nodes.get_mut(node) {
                    n.0.process();
                } else if !self.external_input_nodes.contains_key(node) {
                    self.dynamic_nodes.get_mut(node).unwrap().process();
                }
            }
        }
        for (node,_) in self.nodes.values_mut() {
            node.post_process();
        }
        arr![f32x8; self.output.0, self.output.1]
    }

    pub fn add_node<A: ArrayLength<f32x8> + ToInt<usize> + 'static, B:ArrayLength<f32x8> + ToInt<usize> + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(&mut self, name: String, ty: String, n: N) {
        self.add_boxed_node(name, ty, BoxedDynamicNode::new(n));
    }
    pub fn add_boxed_node(&mut self, name: String, ty: String, n: BoxedDynamicNode) {
        self.nodes.insert(name, (n, ty));
    }

    pub fn add_edge(&mut self, src_i: usize, src: String, dst_i: usize, dst: String) {
        let es = self.edges.entry(dst).or_insert_with(|| vec![]);
        es.push((dst_i, src, src_i));
    }

    pub fn update_sort(&mut self) -> Result<(), String> {
        let mut edges = HashMap::new();
        for (dst, srcs) in &self.edges {
            for (_, src, _) in srcs {
                if src == "input" {
                    continue
                }
                edges.entry(src).or_insert_with(|| HashSet::new()).insert(dst);
            }
        }
        let mut nodes:HashSet<String> = self.nodes.keys().cloned().collect();
        nodes.extend(self.dynamic_nodes.keys().cloned());
        nodes.extend(self.external_input_nodes.keys().cloned());
        let delay_nodes:HashSet<_> = self.nodes.iter().filter_map(|(k,(_,ty))| if ty == "delay" || ty == "reg" { Some(k) } else { None } ).collect();

        self.topo_sort.clear();
        while !nodes.is_empty() {
            let mut to_remove = None;
            for node in &nodes {
                if !edges.iter().any(|(src, dsts)| !delay_nodes.contains(src) && dsts.contains(node)) {
                    self.topo_sort.push(node.clone());
                    to_remove = Some(node.clone());
                    break;
                }
            }
            let node = to_remove.ok_or_else(|| format!("Graph must not contain cycles. Remaining_nodes: {:?}", nodes))?;
            edges.remove(&node);
            nodes.remove(&node);
        }
        self.topo_sort.push("output".to_string());
        Ok(())
    }

    fn reparse(&mut self, l: &[Line]) -> Result<(), String> {
        self.edges.clear();
        let mut nodes = HashMap::new();
        let mut definitions = HashMap::new();
        let mut watch_list = HashSet::new();
        let mut input_len = 0;
        let mut raw_sequencers = HashMap::new();

        for l in l {
            match l {
                Line::Node(k, ty, parameters) => {
                    nodes.insert(k, ty.clone());
                    if ty == "pattern_sequencer" {
                        if let Some(NodeParameters::PatternSequence(parameters)) = parameters {
                            raw_sequencers.insert(k.clone(), parameters.clone());
                        } else {
                            panic!();
                        }
                    }
                }
                Line::BridgeNode(in_node, out_node) => {
                    nodes.insert(in_node, "bridge_in".to_string());
                    nodes.insert(out_node, "bridge_out".to_string());
                }
                Line::NodeDefinition(k, l) => {
                    definitions.insert(k, l);
                }
                Line::Include(p) => { watch_list.insert(p.clone()); }
                _ => ()
            }
        }
        self.nodes.retain(|k,v| nodes.get(k).map(|ty| &v.1==ty && ty != "pattern_sequencer").unwrap_or(false));
        self.external_input_nodes.clear();
        self.dynamic_nodes.retain(|k,v| nodes.contains_key(k));
        self.reset_edges.clear();
        for l in l {
            match l {
                Line::Node(k, ty, parameters) => {
                    if let Some(g) = self.dynamic_nodes.get_mut(k) {
                        g.reparse(definitions.get(ty).unwrap())?;
                    } if !self.nodes.contains_key(k) {
                        match ty.as_str() {
                            "sum_sequencer" => {
                                if let Some(NodeParameters::SumSequence(parameters)) = parameters {
                                    let n = SumSequencer::new(*parameters);
                                    self.add_node(k.clone(), ty.clone(), n);
                                } else {
                                    panic!();
                                }
                            },
                            "pattern_sequencer" => {
                                if let Some(NodeParameters::PatternSequence(parameters)) = parameters {
                                    let n = PatternSequencer::new(parameters.as_subsequence(&raw_sequencers).expect("There was a cycle in the sequence definitions!"));
                                    self.add_node(k.clone(), ty.clone(), n);
                                } else {
                                    panic!();
                                }
                            },
                            "automation" => {
                                if let Some(NodeParameters::AutomationSequence(parameters)) = parameters {
                                    let n = Automation::new(parameters);
                                    self.add_node(k.clone(), ty.clone(), n);
                                } else {
                                    panic!();
                                }
                            },
                            "Constant" => {
                                if let Some(NodeParameters::Number(parameters)) = parameters {
                                    let n = Constant(*parameters);
                                    self.add_node(k.clone(), ty.clone(), n);
                                } else {
                                    panic!();
                                }
                            },
                            _ => {
                                if let Some(template) = self.builder.templates.get(ty) {
                                    let mut n = template.0.clone();
                                    n.set_sample_rate(self.sample_rate);
                                    self.add_boxed_node(k.clone(), ty.clone(), n);
                                } else if let Some(n) = definitions.get(ty) {
                                    let mut g = DynamicGraph::default();
                                    g.reparse(n)?;
                                    watch_list.extend(g.watch_list.lock().unwrap().iter().cloned());
                                    g.set_sample_rate(self.sample_rate);
                                    self.dynamic_nodes.insert(k.clone(), Box::new(g));
                                } else {
                                    return Err(format!("No definition for {}", ty));
                                }
                            }
                        }
                    }
                }
                Line::BridgeNode(in_node, out_node) => {
                    let (a, b) = Bridge::<U1>::new();
                    self.add_node(in_node.clone(), "bridge_in".to_string(), a);
                    self.add_node(out_node.clone(), "bridge_out".to_string(), b);
                }
                Line::ExternalParam(n, k) => {
                    self.external_input_nodes.insert(n.clone(),(k.clone(), 0.0));
                }
                Line::Edge(src, src_i, dst, dst_i) => {
                    if dst != "output" {
                        self.reset_edges.push((dst.clone(), *dst_i as usize));
                    }
                    if src=="input" {
                        input_len = input_len.max(src_i + 1);
                    }
                    self.add_edge(*src_i as usize, src.clone(), *dst_i as usize, dst.clone())
                }
                _ => ()
            }
        }
        self.input.resize(input_len as usize, f32x8::splat(0.0));
        *self.watch_list.lock().unwrap() = watch_list;
        let r = self.update_sort();
        r
    }

    fn get(&self, n: &str, i: usize) -> Option<f32x8> {
        if n == "input" {
            self.input.get(i).cloned()
        } else if let Some((_, v)) = self.external_input_nodes.get(n) {
            Some(f32x8::splat(*v))
        } else if let Some(n) = self.nodes.get(n) {
            Some(n.0.get(i))
        } else if let Some(n) = self.dynamic_nodes.get(n) {
            match i {
                0 => Some(n.output.0),
                1 => Some(n.output.1),
                _ => panic!(),
            }
        } else {
            panic!("Undefined node {}", n);
        }
    }


}
#[derive(Clone, Debug)]
pub enum NodeParameters {
    PatternSequence(RawSubsequence),
    Number(f32),
    AutomationSequence(Vec<Interpolation>),
    SumSequence([(u32, f32); 4]),
}
#[derive(Clone, Debug)]
pub enum RawSubsequence {
    Nested(String),
    Item(f32, f32),
    Rest(f32),
    Tuplet(Vec<RawSubsequence>, usize),
    Iter(Vec<RawSubsequence>, usize),
    Choice(Vec<RawSubsequence>, usize),
    ClockMultiplier(Box<RawSubsequence>, f32),
}

impl RawSubsequence {
    pub fn as_subsequence(&self, defs: &HashMap<String, RawSubsequence>) -> Option<Subsequence> {
        self.as_subsequence_rec(defs, HashSet::new())
    }

    fn as_subsequence_rec(&self, defs: &HashMap<String, RawSubsequence>, mut seen: HashSet<String>) -> Option<Subsequence> {
        match self {
            RawSubsequence::Nested(name) => {
                if seen.insert(name.clone()) {
                    defs.get(name).and_then(|s| {
                        s.as_subsequence_rec(defs, seen.clone()).map(|s| {
                            let len = s.len();
                            if len > 1 {
                                Subsequence::ClockMultiplier(Box::new(s), 1.0/(len as f32))
                            } else {
                                s
                            }
                        })
                    })
                } else {
                    None
                }
            },
            RawSubsequence::Item(a,b,) => Some(Subsequence::Item(*a,*b)),
            RawSubsequence::Rest(a) => Some(Subsequence::Rest(*a)),
            RawSubsequence::Tuplet(a,b) => a.iter().fold(Some(Vec::new()), |v, a| v.and_then(|mut v| a.as_subsequence_rec(defs, seen.clone()).map(|a| {v.push(a); v}))).map(|a| Subsequence::Tuplet(a, *b)),
            RawSubsequence::Iter(a,b) => a.iter().fold(Some(Vec::new()), |v, a| v.and_then(|mut v| a.as_subsequence_rec(defs, seen.clone()).map(|a| {v.push(a); v}))).map(|a| Subsequence::Iter(a, *b)),
            RawSubsequence::Choice(a,b) => a.iter().fold(Some(Vec::new()), |v, a| v.and_then(|mut v| a.as_subsequence_rec(defs, seen.clone()).map(|a| {v.push(a); v}))).map(|a| Subsequence::Choice(a, *b)),
            RawSubsequence::ClockMultiplier(a,b) => a.as_subsequence_rec(defs, seen.clone()).map(|a| Subsequence::ClockMultiplier(Box::new(a),*b)),
        }
    }
}

#[derive(Debug)]
pub enum Line {
    ExternalParam(String, String),
    Node(String, String, Option<NodeParameters>),
    BridgeNode(String, String),
    NodeDefinition(String, Vec<Line>),
    Edge(String, u32, String, u32),
    Include(String),
    Comment,
}

impl Node for DynamicGraph {
    type Input = U0;
    type Output = U2;
    #[inline]
    fn process(&mut self, _input: Ports<Self::Input>) -> Ports<Self::Output> {
        let mut reparse_data = None;
        if let Ok(mut d) = self.reload_data.lock() {
            reparse_data = d.take();
        }
        if let Some(d) = reparse_data {
            let l = self.builder.parse_inner(&d);
            match l {
                Ok(l) => {
                    let mut g = self.clone();
                    let r = g.reparse(&l);
                    if r.is_err() {
                        println!("{:?}", r);
                    } else {
                        *self = g;
                    }
                }
                Err(e) => println!("{:?}", e)
            }
            println!("reloaded...");
        }
        self.process()
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        for node in self.nodes.values_mut() {
            node.0.set_sample_rate(rate);
        }
        for node in self.dynamic_nodes.values_mut() {
            node.set_sample_rate(rate);
        }
    }
}



