use dyn_clone::DynClone;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use generic_array::{
    arr,
    typenum::{ToInt, U0, U1, U2},
    ArrayLength,
};
use virtual_modular_core_nodes::*;
use virtual_modular_definition_language::{parse, Line};
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[macro_export]
macro_rules! dynamic_nodes {
    ($loader_name:ident {$($name:ident: $constructor:expr),*}) => {
        #[macro_export]
        macro_rules! $loader_name {
            ($builder:expr) => {
                $(
                    $builder.templates.insert(stringify!($name).to_string(), (BoxedDynamicNode::new($constructor), stringify!($constructor).to_string()));
                )*
            }
        }
    }
}

dynamic_nodes! {
    std_nodes {
        ToneHoleFlute: ToneHoleFlute::default(),
        Add: Add,
        Compressor: Compressor::default(),
        SoftClip:SoftClip,
        Sub: Sub,
        Mul: Mul,
        Div: Div,
        Imp: Impulse::default(),
        CXor: CXor,
        Comp: Comparator,
        Sine: Sine::default(),
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
        Portamento: Portamento::default(),
        Reverb: Reverb::default(),
        Delay: ModableDelay::default(),
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
        ScaleMajor: Quantizer::new(&[16.351875, 18.35375, 20.601875, 21.826875, 24.5, 27.5, 30.8675, 32.703125]),
        ScaleDegreeMajor: DegreeQuantizer::new(&[16.351875, 18.35375, 20.601875, 21.826875, 24.5, 27.5, 30.8675]),
        ScaleDegreeMinor: DegreeQuantizer::new(&[18.35,20.60,21.83,24.50,27.50,29.14, 32.70]),
        ScaleDegreeChromatic: DegreeQuantizer::chromatic(),
        ScaleDegreeGoblin: TritaveDegreeQuantizer::new(&[18.35,21.468,25.116,30.8,34.3777,40.2195,47.054]),
        BowedString: BowedString::default(),
        PluckedString: PluckedString::default(),
        ImaginaryGuitar: ImaginaryGuitar::default(),
        SympatheticString: SympatheticString::default(),
        WaveMesh: WaveMesh::default(),
        PennyWhistle: PennyWhistle::default(),
        StereoIdentity: StereoIdentity,
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
            templates: Default::default(),
        };
        std_nodes!(&mut b);
        b
    }
}

impl DynamicGraphBuilder {
    pub fn parse(&self, data: &str) -> Result<DynamicGraph, String> {
        let mut g = DynamicGraph::default();
        let l = parse(data)?;

        g.reparse(&l)?;
        Ok(g)
    }
}

pub trait DynamicNode: DynClone {
    fn process(&mut self);
    fn post_process(&mut self);
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn get(&self, i: usize) -> [f32; BLOCK_SIZE];
    fn set(&mut self, i: usize, v: [f32; BLOCK_SIZE]);
    fn add(&mut self, i: usize, v: [f32; BLOCK_SIZE]);
    fn set_static_parameters(&mut self, _parameters: &str) -> Result<(), String> {
        Ok(())
    }
    fn set_sample_rate(&mut self, rate: f32);
}
dyn_clone::clone_trait_object!(DynamicNode);

impl<
        A: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + Clone,
        B: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + Clone,
        N: Node<Input = A, Output = B> + Clone,
    > DynamicNode for (N, RefCell<Ports<A>>, RefCell<Ports<B>>)
{
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

    fn get(&self, i: usize) -> [f32; BLOCK_SIZE] {
        self.2.borrow()[i]
    }

    fn set(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.1.borrow_mut()[i] = v;
    }

    fn add(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.1.borrow_mut()[i]
            .iter_mut()
            .zip(&v)
            .for_each(|(r, v)| *r += *v);
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.0.set_static_parameters(parameters)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

impl DynamicNode for (DynamicGraph, RefCell<[[f32; BLOCK_SIZE]; 2]>) {
    fn process(&mut self) {
        //TODO: Should this panic or propogate the error?
        let r = self.0.process().unwrap();
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

    fn get(&self, i: usize) -> [f32; BLOCK_SIZE] {
        self.1.borrow()[i]
    }

    fn set(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.0.input[i] = v;
    }

    fn add(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.0.input[i]
            .iter_mut()
            .zip(&v)
            .for_each(|(r, v)| *r += *v);
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct BoxedDynamicNode(Box<dyn DynamicNode>);

impl std::fmt::Debug for BoxedDynamicNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedDynamicNode").finish()
    }
}
impl BoxedDynamicNode {
    pub fn new<
        A: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + Clone + 'static,
        B: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + 'static,
        N: Node<Input = A, Output = B> + Clone + 'static,
    >(
        n: N,
    ) -> Self {
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

    fn get(&self, i: usize) -> [f32; BLOCK_SIZE] {
        self.0.get(i)
    }

    fn set(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.0.set(i, v);
    }

    fn add(&mut self, i: usize, v: [f32; BLOCK_SIZE]) {
        self.0.add(i, v);
    }

    fn set_static_parameters(&mut self, parameters: &str) -> Result<(), String> {
        self.0.set_static_parameters(parameters)
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: HashMap<String, (BoxedDynamicNode, String)>,
    builder: DynamicGraphBuilder,
    pub external_inputs: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    external_input_nodes: HashMap<String, (String, f32)>,
    pub watch_list: Arc<Mutex<HashSet<String>>>,
    input: Vec<[f32; BLOCK_SIZE]>,
    output: ([f32; BLOCK_SIZE], [f32; BLOCK_SIZE]),
    edges: HashMap<String, Vec<(usize, String, usize)>>,
    reset_edges: Vec<(String, usize)>,
    topo_sort: Vec<String>,
    pub reload_data: Arc<Mutex<Option<String>>>,
    sample_rate: f32,
}

impl DynamicGraph {
    pub fn process(&mut self) -> Result<Ports<U2>, String> {
        if let Ok(mut ei) = self.external_inputs.lock() {
            for (k, v) in self.external_input_nodes.values_mut() {
                if let Some(nv) = ei.get_mut(k).and_then(|v| {
                    if !v.is_empty() {
                        Some(v.remove(0))
                    } else {
                        None
                    }
                }) {
                    *v = nv;
                }
            }
        }
        self.output = ([0.0; BLOCK_SIZE], [0.0; BLOCK_SIZE]);
        for (n, _) in self.nodes.values_mut() {
            for i in 0..n.input_len() {
                n.set(i, [0.0; BLOCK_SIZE]);
            }
        }

        for node in &self.topo_sort {
            if let Some(others) = self.edges.get(node) {
                for (dst_i, n, src_i) in others {
                    let v = self.get(n, *src_i)?;
                    if node == "output" {
                        if *dst_i == 0 {
                            self.output.0.iter_mut().zip(&v).for_each(|(r, v)| *r += *v);
                        } else {
                            self.output.1.iter_mut().zip(&v).for_each(|(r, v)| *r += *v);
                        }
                    } else if let Some(n) = self.nodes.get_mut(node) {
                        n.0.add(*dst_i, v);
                    } else {
                        return Err(format!("Unknown node {}", node));
                    }
                }
            }
            if node != "output" {
                if let Some(n) = self.nodes.get_mut(node) {
                    n.0.process();
                } else if !self.external_input_nodes.contains_key(node) {
                    return Err(format!("Unknown node {}", node));
                }
            }
        }
        for (node, _) in self.nodes.values_mut() {
            node.post_process();
        }
        Ok(arr![[f32; BLOCK_SIZE]; self.output.0, self.output.1])
    }

    pub fn add_node<
        A: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + 'static,
        B: ArrayLength<[f32; BLOCK_SIZE]> + ToInt<usize> + 'static,
        N: Node<Input = A, Output = B> + Clone + 'static,
    >(
        &mut self,
        name: String,
        ty: String,
        n: N,
    ) {
        self.add_boxed_node(name, ty, BoxedDynamicNode::new(n));
    }
    pub fn add_boxed_node(&mut self, name: String, ty: String, n: BoxedDynamicNode) {
        self.nodes.insert(name, (n, ty));
    }

    pub fn add_edge(&mut self, src_i: usize, src: String, dst_i: usize, dst: String) {
        let es = self.edges.entry(dst).or_insert_with(Vec::new);
        es.push((dst_i, src, src_i));
    }

    pub fn update_sort(&mut self) -> Result<(), String> {
        let mut edges = HashMap::new();
        for (dst, srcs) in &self.edges {
            for (_, src, _) in srcs {
                if src == "input" {
                    continue;
                }
                edges.entry(src).or_insert_with(HashSet::new).insert(dst);
            }
        }
        let mut nodes: HashSet<String> = self.nodes.keys().cloned().collect();
        nodes.extend(self.external_input_nodes.keys().cloned());
        let delay_nodes: HashSet<_> = self
            .nodes
            .iter()
            .filter_map(|(k, (_, ty))| {
                if ty == "delay" || ty == "reg" {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();

        self.topo_sort.clear();
        while !nodes.is_empty() {
            let mut to_remove = None;
            for node in &nodes {
                if !edges
                    .iter()
                    .any(|(src, dsts)| !delay_nodes.contains(src) && dsts.contains(node))
                {
                    self.topo_sort.push(node.clone());
                    to_remove = Some(node.clone());
                    break;
                }
            }
            let node = to_remove.ok_or_else(|| {
                format!(
                    "Graph must not contain cycles. Remaining_nodes: {:?}",
                    nodes
                )
            })?;
            edges.remove(&node);
            nodes.remove(&node);
        }
        self.topo_sort.push("output".to_string());
        Ok(())
    }

    fn reparse(&mut self, l: &[Line]) -> Result<(), String> {
        self.edges.clear();
        let mut nodes = HashMap::new();
        let watch_list = HashSet::new();
        let mut input_len = 0;

        for l in l {
            match l {
                Line::Node { name, ty, .. } => {
                    nodes.insert(name, ty.clone());
                }
                Line::BridgeNode(in_node, out_node) => {
                    nodes.insert(in_node, "bridge_in".to_string());
                    nodes.insert(out_node, "bridge_out".to_string());
                }
                _ => (),
            }
        }
        self.nodes
            .retain(|k, v| nodes.get(k).map(|ty| &v.1 == ty).unwrap_or(false));
        self.external_input_nodes.clear();
        self.reset_edges.clear();
        for l in l {
            match l {
                Line::Node {
                    name,
                    ty,
                    static_parameters,
                } => {
                    if !self.nodes.contains_key(name) {
                        match ty.as_str() {
                            "Constant" => {
                                if let Some(value) = static_parameters {
                                    let mut n = Constant::default();
                                    n.set_static_parameters(value)?;
                                    self.add_node(name.clone(), ty.clone(), n);
                                } else {
                                    return Err("Missing Constant value".to_string());
                                }
                            }
                            _ => {
                                if let Some(template) = self.builder.templates.get(ty) {
                                    let mut n = template.0.clone();
                                    n.set_sample_rate(self.sample_rate);
                                    if let Some(p) = static_parameters {
                                        n.set_static_parameters(p)?;
                                    }
                                    self.add_boxed_node(name.clone(), ty.clone(), n);
                                } else {
                                    return Err(format!("No definition for {}", ty));
                                }
                            }
                        }
                    } else if let Some(p) = static_parameters {
                        if let Some(n) = self.nodes.get_mut(name) {
                            n.0.set_static_parameters(p)?;
                        }
                    }
                }
                Line::BridgeNode(in_node, out_node) => {
                    let (a, b) = Bridge::<U1>::new();
                    self.add_node(in_node.clone(), "bridge_in".to_string(), a);
                    self.add_node(out_node.clone(), "bridge_out".to_string(), b);
                }
                Line::ExternalParam(n, k) => {
                    self.external_input_nodes
                        .insert(n.clone(), (k.clone(), 0.0));
                }
                Line::Edge(src, src_i, dst, dst_i) => {
                    if dst != "output" {
                        self.reset_edges.push((dst.clone(), *dst_i as usize));
                    }
                    if src == "input" {
                        input_len = input_len.max(src_i + 1);
                    }
                    self.add_edge(*src_i as usize, src.clone(), *dst_i as usize, dst.clone())
                }
                _ => (),
            }
        }
        self.input.resize(input_len as usize, [0.0; BLOCK_SIZE]);
        *self.watch_list.lock().unwrap() = watch_list;
        self.update_sort()
    }

    fn get(&self, n: &str, i: usize) -> Result<[f32; BLOCK_SIZE], String> {
        if n == "input" {
            self.input.get(i).cloned().ok_or_else(|| {
                format!(
                    "Input index {} greater than input len {}",
                    i,
                    self.input.len()
                )
            })
        } else if let Some((_, v)) = self.external_input_nodes.get(n) {
            Ok([*v; BLOCK_SIZE])
        } else if let Some(n) = self.nodes.get(n) {
            Ok(n.0.get(i))
        } else {
            Err(format!("Undefined node {}", n))
        }
    }
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
            let l = parse(&d);
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
                Err(e) => println!("{:?}", e),
            }
            println!("reloaded...");
        }
        self.process().unwrap()
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        for node in self.nodes.values_mut() {
            node.0.set_sample_rate(rate);
        }
    }
}
