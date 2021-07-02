use dyn_clone::DynClone;
use std::{
    collections::{HashSet, HashMap},
    cell::RefCell,
};

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

use linkme::distributed_slice;
#[distributed_slice]
pub static MODULES: [fn() -> (String, BoxedDynamicNode)] = [..];

#[distributed_slice(MODULES)]
fn c() -> (String, BoxedDynamicNode) {
    let n = BoxedDynamicNode(Box::new((FakeConstant, RefCell::new(Value((f32x8::splat(440.0),))))));
    ("c".to_string(), n)
}
#[distributed_slice(MODULES)]
fn sine() -> (String, BoxedDynamicNode) {
    ("sine".to_string(), BoxedDynamicNode::new(crate::simd_graph::pipe(crate::simd_graph::WaveTable::sine(), crate::simd_graph::Split)))
}
#[distributed_slice(MODULES)]
fn saw() -> (String, BoxedDynamicNode) {
    ("saw".to_string(), BoxedDynamicNode::new(crate::simd_graph::pipe(crate::simd_graph::WaveTable::saw(), crate::simd_graph::Split)))
}

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: Vec<BoxedDynamicNode>,
    output_node: usize,
    edges: HashMap<usize, (Vec<usize>, Vec<usize>)>,
    topo_sort: Vec<usize>,
}

impl DynamicGraph {
    pub fn process(&mut self) -> Value<(f32x8, f32x8)> {
        for node in &self.topo_sort {
            self.nodes[*node].process();
            if let Some((others_a, others_b)) = self.edges.get(node) {
                for other_node in others_a {
                    let v = self.nodes[*node].get(0);
                    self.nodes[*other_node].set(0, v);
                }
                for other_node in others_b {
                    let v = self.nodes[*node].get(1);
                    self.nodes[*other_node].set(1, v);
                }
            }
        }
        let n = &self.nodes[self.output_node];
        Value((n.get(0), n.get(1)))
    }

    pub fn add_node<A: DynamicValue + Default + 'static, B:DynamicValue + Default + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(&mut self, n: N) -> usize {
        self.add_boxed_node(BoxedDynamicNode::new(n))
    }
    pub fn add_boxed_node(&mut self, n: BoxedDynamicNode) -> usize {
        let k = self.nodes.len();
        self.nodes.push(n);
        k
    }

    pub fn add_edge(&mut self, a: usize, v: usize, b: usize) {
        let e = self.edges.entry(a).or_insert_with(|| (vec![], vec![]));
        if v == 0 {
            e.0.push(b);
        } else {
            e.1.push(b);
        }
    }

    pub fn set_output_node(&mut self, n: usize) {
        self.output_node = n;
    }

    pub fn update_sort(&mut self) {
        let mut edges = self.edges.clone();
        let mut nodes:HashSet<usize> = (0..self.nodes.len()).collect();

        self.topo_sort.clear();
        while !nodes.is_empty() {
            let mut to_remove = None;
            for node in &nodes {
                if !edges.values().any(|(a, b)| a.iter().any(|k| k==node) || b.iter().any(|k| k==node)) {
                    self.topo_sort.push(*node);
                    to_remove = Some(*node);
                    break;
                }
            }
            let node = to_remove.expect("Graph must not contain cycles");
            edges.remove(&node);
            nodes.remove(&node);
        }
    }

    pub fn parse(data: &str) -> Self {
        use pom::parser::*;
        use pom::parser::Parser;
        use std::str::{self, FromStr};
        #[derive(Debug)]
        enum Line {
            Node(String, String, Option<f32>),
            Edge(String, u32, String),
            OutputNode(String),
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

            let parameter = (sym(b'(') * number - sym(b')')).opt();
            ((none_of(b"\n=(),")).repeat(1..).convert(String::from_utf8) - sym(b'=') + (none_of(b"(),\n")).repeat(1..).convert(String::from_utf8) + parameter).map(|((k, n), p)| Line::Node(k, n, p))
        }
        fn output<'a>() -> Parser<'a, u8, Line> {
            let p = seq(b"output(") * none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b')');
            p.map(|k| Line::OutputNode(k))
        }
        fn edge<'a>() -> Parser<'a, u8, Line> {
            let p = sym(
                b'(') *
                (none_of(b"),")).repeat(1..).convert(String::from_utf8) - sym(b',') +
                one_of(b"0123456789").repeat(1..).convert(String::from_utf8).convert(|s|u32::from_str(&s)) - sym(b',') +
                (none_of(b")")).repeat(1..).convert(String::from_utf8)
                - sym(b')');
            p.map(|((a, c),b)| Line::Edge(a, c, b))
        }
        (one_of(b" \n").repeat(0..) * list(output() | edge() | node(), sym(b'\n')) - one_of(b" \n").repeat(0..)).map(|l| {
            println!("{:?}", l);
            let modules: HashMap<_,_> = MODULES.iter().map(|f| f()).collect();
            let mut g = DynamicGraph::default();
            let mut nodes = HashMap::new();
            for l in &l {
                if let Line::Node(k, n, p) = l {
                    let k = k.trim().to_string();
                    let mut n = modules[n].clone();
                    if let Some(p) = p {
                        n.set(0, f32x8::splat(*p));
                    }
                    nodes.insert(k, g.add_boxed_node(n));
                }
            }
            for l in &l {
                match l {
                    Line::Edge(a, c, b) => g.add_edge(nodes[a], *c as usize, nodes[b]),
                    Line::OutputNode(a) => g.set_output_node(nodes[a]),
                    _ => ()
                }
            }
            g.update_sort();
            g
        }).parse(data.as_bytes()).unwrap()
    }
}

impl Node for DynamicGraph {
    type Input = NoValue;
    type Output = Value<(f32x8, f32x8)>;
    #[inline]
    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.process()
    }

    fn set_sample_rate(&mut self, rate: f32) {
        for node in &mut self.nodes {
            node.set_sample_rate(rate);
        }
    }
}
