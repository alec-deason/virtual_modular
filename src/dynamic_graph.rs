use dyn_clone::DynClone;
use std::{
    sync::{Mutex,Arc},
    collections::{HashSet, HashMap},
    cell::RefCell,
};
use linkme::distributed_slice;

use packed_simd_2::f32x8;

use crate::{
    simd_graph::Node,
    type_list::{DynamicValue, Value, NoValue},
};

pub trait DynamicNode: DynClone {
    fn process(&mut self);
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn get(&self, i:usize) -> f32x8;
    fn set(&mut self, i:usize, v: f32x8);
    fn set_sample_rate(&mut self, rate: f32);
}
dyn_clone::clone_trait_object!(DynamicNode);

impl<A: DynamicValue, B:DynamicValue, N: Node<Input=A, Output=B> + Clone> DynamicNode for (N, RefCell<A>, RefCell<B>) {
    fn process(&mut self) {
        *self.2.borrow_mut() = self.0.process(self.1.borrow().clone());
    }

    fn input_len(&self) -> usize {
        self.1.borrow().len()
    }

    fn output_len(&self) -> usize {
        self.2.borrow().len()
    }

    fn get(&self, i:usize) -> f32x8 {
        self.2.borrow().get(i)
    }

    fn set(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut().set(i, v);
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Copy, Clone)]
pub struct FakeConstant;
impl<A: DynamicValue> DynamicNode for (FakeConstant, RefCell<A>) {
    fn process(&mut self) {
    }

    fn input_len(&self) -> usize {
        1
    }

    fn output_len(&self) -> usize {
        1
    }

    fn get(&self, i:usize) -> f32x8 {
        self.1.borrow().get(i)
    }

    fn set(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut().set(i, v)
    }

    fn set_sample_rate(&mut self, rate: f32) {
    }
}

impl DynamicNode for (DynamicGraph, RefCell<[f32x8; 2]>) {
    fn process(&mut self) {
        let r = self.0.process();
        let mut o = self.1.borrow_mut();
        o[0] = (r.0).0;
        o[1] = (r.0).1;
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

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone)]
pub struct BoxedDynamicNode(Box<dyn DynamicNode>);
impl BoxedDynamicNode {
        pub fn new<A: DynamicValue + Default + 'static, B:DynamicValue + Default + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(n: N) -> Self {
        let a = A::default();
        let b = B::default();
        Self(Box::new((n, RefCell::new(a), RefCell::new(b))))
    }
}
impl DynamicNode for BoxedDynamicNode {
    fn process(&mut self) {
        self.0.process();
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

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[distributed_slice]
pub static MODULES: [fn() -> (String, BoxedDynamicNode)] = [..];

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: HashMap<String, (BoxedDynamicNode, String)>,
    output_node: Option<String>,
    input: Vec<f32x8>,
    edges: HashMap<String, Vec<(usize, String, usize)>>,
    topo_sort: Vec<String>,
    pub reload_data: Arc<Mutex<Option<String>>>
}

impl DynamicGraph {
    pub fn process(&mut self) -> Value<(f32x8, f32x8)> {
        if let Some(others) = self.edges.get("input") {
            for (v1, other_node, v2) in others {
                let v = self.input[*v1];
                self.nodes.get_mut(other_node).expect(&format!("no definition for {}",other_node)).0.set(*v2, v);
            }
        }

        for node in &self.topo_sort {
            self.nodes.get_mut(node).unwrap().0.process();
            if let Some(others) = self.edges.get(node) {
                for (v1, other_node, v2) in others {
                    let v = self.nodes[node].0.get(*v1);
                    self.nodes.get_mut(other_node).expect(&format!("no definition for {}",other_node)).0.set(*v2, v);
                }
            }
        }
        let n = self.output_node.as_ref().expect("no output node set");
        let n = &self.nodes.get(n).expect(&format!("Output node {} not defined", n));
        if n.0.output_len() == 1 {
            Value((n.0.get(0), n.0.get(0)))
        } else {
            Value((n.0.get(0), n.0.get(1)))
        }
    }

    pub fn add_node<A: DynamicValue + Default + 'static, B:DynamicValue + Default + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(&mut self, name: String, ty: String, n: N) {
        self.add_boxed_node(name, ty, BoxedDynamicNode::new(n));
    }
    pub fn add_boxed_node(&mut self, name: String, ty: String, n: BoxedDynamicNode) {
        self.nodes.insert(name, (n, ty));
    }

    pub fn add_edge(&mut self, v1: usize, a: String, v2: usize, b: String) {
        let e = self.edges.entry(a).or_insert_with(|| vec![]);
        e.push((v1, b, v2));
    }

    pub fn set_output_node(&mut self, n: String) {
        self.output_node = Some(n);
    }

    pub fn update_sort(&mut self) {
        let mut edges = self.edges.clone();
        let mut nodes:HashSet<String> = self.nodes.keys().cloned().collect();

        self.topo_sort.clear();
        while !nodes.is_empty() {
            let mut to_remove = None;
            for node in &nodes {
                if !edges.iter().any(|(k, a)| k!="input" && a.iter().any(|(_, k, _)| k==node)) {
                    self.topo_sort.push(node.clone());
                    to_remove = Some(node.clone());
                    break;
                }
            }
            let node = to_remove.expect("Graph must not contain cycles");
            edges.remove(&node);
            nodes.remove(&node);
        }
    }

    fn reparse(&mut self, l: &[Line]) {
        let modules: HashMap<_,_> = MODULES.iter().map(|f| f()).collect();
        self.edges.clear();
        self.output_node = None;
        let mut nodes = HashMap::new();
        let mut definitions = HashMap::new();
        for l in l {
            match l {
                Line::Node(k, ty, _) => {
                    let k = k.trim().to_string();
                    nodes.insert(k, ty);
                }
                Line::NodeDefinition(k, l) => {
                    let mut g = DynamicGraph::default();
                    g.reparse(l);
                    definitions.insert(k, g);
                }
                Line::Input(len) => {
                    self.input.resize(*len as usize, f32x8::splat(0.0));
                }
                _ => ()
            }
        }
        self.nodes.retain(|k,v| nodes.get(k).map(|ty| &&v.1==ty).unwrap_or(false));
        for l in l {
            match l {
                Line::Node(k, ty, p) => {
                    let k = k.trim().to_string();
                    if !self.nodes.contains_key(&k) {
                        if let Some(n) = modules.get(ty) {
                            self.add_boxed_node(k.clone(), ty.clone(), n.clone());
                        } else if let Some(n) = definitions.get(ty) {
                            let n = BoxedDynamicNode(Box::new((n.clone(), RefCell::new([f32x8::splat(0.0), f32x8::splat(0.0)]))));
                            self.add_boxed_node(k.clone(), ty.clone(), n);
                        } else {
                            panic!("No definition for {}", ty);
                        }
                    }
                    let n = &mut self.nodes.get_mut(&k).unwrap().0;
                    if let Some(ps) = p {
                        for (i,p) in ps.iter().enumerate() {
                            n.set(i, f32x8::splat(*p));
                        }
                    }
                }
                Line::Edge(c1, a, c, b) => self.add_edge(*c1 as usize, a.clone(), *c as usize, b.clone()),
                Line::OutputNode(a) => self.set_output_node(a.clone()),
                _ => ()
            }
        }
        self.update_sort();
    }

    pub fn parse(data: &str) -> Self {
        let mut g = DynamicGraph::default();
        let l = Self::parse_inner(data);

        g.reparse(&l);
        g
    }

    fn parse_inner(data: &str) -> Vec<Line> {
        use pom::parser::*;
        use pom::parser::Parser;
        use std::str::{self, FromStr};
        fn comment<'a>() -> Parser<'a, u8, Line> {
            let comment = (sym(b'#') * none_of(b"\n").repeat(0..)) - sym(b'\n');
            let empty_line = (sym(b'\n')).repeat(1);
            (comment | empty_line).map(|_| Line::Comment)
        }
        fn outer_synth<'a>() -> Parser<'a, u8, Vec<Line>> {
            (node_definition() | output() | edge() | node() | comment()).repeat(1..)
        }
        fn inner_synth<'a>() -> Parser<'a, u8, Vec<Line>> {
            (input() | output() | edge() | node() | comment()).repeat(1..)
        }
        fn node_definition<'a>() -> Parser<'a, u8, Line> {
            let p = (none_of(b"\n={,")).repeat(1..).convert(String::from_utf8) - sym(b'{') + inner_synth() - sym(b'}') - sym(b'\n');
            p.map(|(name, lines)| Line::NodeDefinition(name, lines))
        }
        fn node<'a>() -> Parser<'a, u8, Line> {
            let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
            let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
            let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
            let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
            let number = number
                .collect()
                .convert(str::from_utf8)
                .convert(f32::from_str);

            let parameter = (sym(b'(') * list(number, sym(b',')) - sym(b')')).opt();
            ((none_of(b"\n=(),")).repeat(1..).convert(String::from_utf8) - sym(b'=') + (none_of(b"(),\n")).repeat(1..).convert(String::from_utf8) + parameter - sym(b'\n')).map(|((k, n), p)| Line::Node(k, n, p))
        }
        fn output<'a>() -> Parser<'a, u8, Line> {
            let p = seq(b"output(") * none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b')') - sym(b'\n');
            p.map(|k| Line::OutputNode(k))
        }
        fn input<'a>() -> Parser<'a, u8, Line> {
            let integer = one_of(b"0123456789").repeat(1..);
            let p = seq(b"input(") * integer.collect().convert(str::from_utf8).convert(u32::from_str) - sym(b')') - sym(b'\n');
            p.map(|k| Line::Input(k))
        }
        fn edge<'a>() -> Parser<'a, u8, Line> {
            let p = sym(
                b'(') *
                one_of(b"0123456789").repeat(1..).convert(String::from_utf8).convert(|s|u32::from_str(&s)) - sym(b',') +
                (none_of(b"),")).repeat(1..).convert(String::from_utf8) - sym(b',') +
                one_of(b"0123456789").repeat(1..).convert(String::from_utf8).convert(|s|u32::from_str(&s)) - sym(b',') +
                (none_of(b")")).repeat(1..).convert(String::from_utf8)
                - sym(b')') - sym(b'\n');
            p.map(|(((c1, a), c),b)| Line::Edge(c1, a, c, b))
        }
        outer_synth().parse(data.as_bytes()).unwrap()
    }
}
#[derive(Debug)]
enum Line {
    Node(String, String, Option<Vec<f32>>),
    NodeDefinition(String, Vec<Line>),
    Edge(u32, String, u32, String),
    OutputNode(String),
    Input(u32),
    Comment,
}

impl Node for DynamicGraph {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, _input: Self::Input) -> Self::Output {
        let mut reparse_data = None;
        if let Ok(mut d) = self.reload_data.lock() {
            reparse_data = d.take();
        }
        if let Some(d) = reparse_data {
            let l = Self::parse_inner(&d);
            self.reparse(&l);
        }
        self.process()
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for node in self.nodes.values_mut() {
            node.0.set_sample_rate(rate);
        }
    }
}


use crate::simd_graph::*;
#[distributed_slice(MODULES)]
fn dynamic_sine() -> (String, BoxedDynamicNode) {
    ("sine".to_string(), BoxedDynamicNode::new(WaveTable::sine()))
}
#[distributed_slice(MODULES)]
fn dynamic_saw() -> (String, BoxedDynamicNode) {
    ("saw".to_string(), BoxedDynamicNode::new(WaveTable::saw()))
}
#[distributed_slice(MODULES)]
fn dynamic_square() -> (String, BoxedDynamicNode) {
    ("square".to_string(), BoxedDynamicNode::new(WaveTable::square()))
}
