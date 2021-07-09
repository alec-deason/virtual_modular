use dyn_clone::DynClone;
use std::{
    sync::{Mutex,Arc},
    collections::{HashSet, HashMap},
    cell::RefCell,
    convert::{TryFrom, TryInto},
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
    fn add(&mut self, i:usize, v: f32x8);
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

    fn add(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut().add(i, v);
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.0.set_sample_rate(rate);
    }
}

#[derive(Clone, Debug)]
pub enum Term {
    Node(String, usize),
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
    fn evaluate(&self, get_node: &mut impl FnMut(String, usize) -> f32x8) -> f32x8 {
        match self {
            Expression::Term(t) => {
                match t {
                    Term::Node(n, c) => get_node(n.clone(), *c),
                    Term::Number(n) => f32x8::splat(*n)
                }
            }
            Expression::Operation(a, o, b) => {
                let a = a.evaluate(get_node);
                let b = b.evaluate(get_node);
                match o {
                    Operator::Add => a + b,
                    Operator::Sub => a - b,
                    Operator::Mul => a * b,
                    Operator::Div => a / b,
                }
            }
        }
    }

    fn nodes(&self) -> Vec<(String, usize)> {
        match self {
            Expression::Term(Term::Node(n, i)) => vec![(n.clone(), *i)],
            Expression::Operation(a, _, b) => [a.nodes(), b.nodes()].concat(),
            _ => vec![]
        }
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

    fn add(&mut self, i:usize, v: f32x8) {
        self.1.borrow_mut().add(i, v)
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

    fn add(&mut self, i:usize, v: f32x8) {
        self.0.input[i] += v;
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

    fn add(&mut self, i:usize, v: f32x8) {
        self.0.add(i, v);
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
    dynamic_nodes: HashMap<String, Box<DynamicGraph>>,
    pub watch_list: Arc<Mutex<HashSet<String>>>,
    input: Vec<f32x8>,
    output: (f32x8, f32x8),
    edges: HashMap<String, Vec<(usize, Expression)>>,
    reset_edges: Vec<(String,usize)>,
    topo_sort: Vec<String>,
    pub reload_data: Arc<Mutex<Option<String>>>,
    sample_rate: f32,
}

impl DynamicGraph {
    pub fn process(&mut self) -> Value<(f32x8, f32x8)> {
        self.output = (f32x8::splat(0.0), f32x8::splat(0.0));
        for (n,v) in &self.reset_edges {
            if let Some(node) = self.nodes.get_mut(n) {
                node.0.set(*v, f32x8::splat(0.0));
            } else {
                self.dynamic_nodes.get_mut(n).expect(&format!("no definition for {}",n)).input[*v] = f32x8::splat(0.0);
            }
        }
        let DynamicGraph {
            edges,
            nodes,
            input,
            dynamic_nodes,
            ..
        } = self;

        let DynamicGraph {
            topo_sort,
            nodes,
            output,
            dynamic_nodes,
            edges,
            ..
        } = self;
        for node in topo_sort {
            if let Some(others) = edges.get(node) {
                for (i, e) in others {
                    let mut f = |k, c| {
                        if let Some(n) = nodes.get_mut(&k) {
                            n.0.get(c)
                        } else {
                            let g = dynamic_nodes.get_mut(&k).expect(&format!("no definition for {}",&k));
                            g.input[c]
                        }
                    };
                    let v = e.evaluate(&mut f);
                    if node == "output" {
                        if *i == 0 {
                            output.0 += v;
                        } else {
                            output.1 += v;
                        }
                    } else {
                        if let Some(n) = nodes.get_mut(node) {
                            n.0.add(*i, v);
                        } else {
                            dynamic_nodes.get_mut(node).expect(&format!("no definition for {}",node)).input[*i] += v;
                        }
                    }
                }
            }
            if node != "output" {
                if let Some(n) = nodes.get_mut(node) {
                    n.0.process();
                } else {
                    dynamic_nodes.get_mut(node).unwrap().process();
                }
            }
        }
        Value((self.output.0, self.output.1))
    }

    pub fn add_node<A: DynamicValue + Default + 'static, B:DynamicValue + Default + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(&mut self, name: String, ty: String, n: N) {
        self.add_boxed_node(name, ty, BoxedDynamicNode::new(n));
    }
    pub fn add_boxed_node(&mut self, name: String, ty: String, n: BoxedDynamicNode) {
        self.nodes.insert(name, (n, ty));
    }

    pub fn add_edge(&mut self, i: usize, a: String, e: Expression) {
        let es = self.edges.entry(a).or_insert_with(|| vec![]);
        es.push((i, e));
    }

    pub fn update_sort(&mut self) -> Result<(), String> {
        let mut edges = HashMap::new();
        for (dst, srcs) in &self.edges {
            for (_, e) in srcs {
                for (n,_) in e.nodes() {
                    edges.entry(n).or_insert_with(|| vec![]).push(dst);
                }
            }
        }
        let mut nodes:HashSet<String> = self.nodes.keys().cloned().collect();
        nodes.extend(self.dynamic_nodes.keys().cloned());

        self.topo_sort.clear();
        while !nodes.is_empty() {
            let mut to_remove = None;
            for node in &nodes {
                if !edges.iter().any(|(k, a)| k!="input" && a.iter().any(|k| k==&node)) {
                    self.topo_sort.push(node.clone());
                    to_remove = Some(node.clone());
                    break;
                }
            }
            let node = to_remove.ok_or("Graph must not contain cycles".to_string())?;
            edges.remove(&node);
            nodes.remove(&node);
        }
        self.topo_sort.push("output".to_string());
        Ok(())
    }

    fn reparse(&mut self, l: &[Line]) -> Result<(), String> {
        let modules: HashMap<_,_> = MODULES.iter().map(|f| f()).collect();
        self.edges.clear();
        let mut nodes = HashMap::new();
        let mut definitions = HashMap::new();
        let mut watch_list = HashSet::new();
        let mut input_len = 0;
        for l in l {
            match l {
                Line::Node(k, ty, _) => {
                    let k = k.trim().to_string();
                    nodes.insert(k, ty);
                }
                Line::NodeDefinition(k, l) => {
                    definitions.insert(k, l);
                }
                Line::Include(p) => { watch_list.insert(p.clone()); }
                _ => ()
            }
        }
        self.nodes.retain(|k,v| nodes.get(k).map(|ty| &&v.1==ty).unwrap_or(false));
        self.dynamic_nodes.retain(|k,v| nodes.contains_key(k));
        self.reset_edges.clear();
        for l in l {
            match l {
                Line::Node(k, ty, o) => {
                    let k = k.trim().to_string();
                    if let Some(o) = o {
                        let mut n = o.clone();
                        n.set_sample_rate(self.sample_rate);
                        self.add_boxed_node(k.clone(), ty.clone(), n);
                    } else {
                        if let Some(g) = self.dynamic_nodes.get_mut(&k) {
                            g.reparse(definitions.get(ty).unwrap())?;
                        } if !self.nodes.contains_key(&k) {
                            if let Some(n) = modules.get(ty) {
                                let mut n = n.clone();
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
                Line::Edge(dst, i, e) => {
                    if dst != "output" {
                        self.reset_edges.push((dst.clone(), *i as usize));
                    }
                    for (n,i) in e.nodes() {
                        if n=="input" {
                            input_len = input_len.max(i + 1);
                        }
                    }
                    self.add_edge(*i as usize, dst.clone(), e.clone())
                }
                _ => ()
            }
        }
        self.input.resize(input_len, f32x8::splat(0.0));
        *self.watch_list.lock().unwrap() = watch_list;
        let r = self.update_sort();
        r
    }

    pub fn parse(data: &str) -> Result<Self, String> {
        let mut g = DynamicGraph::default();
        let l = Self::parse_inner(data)?;

        g.reparse(&l)?;
        Ok(g)
    }


    fn parse_inner(data: &str) -> Result<Vec<Line>, String> {
        use pom::parser::*;
        use pom::parser::Parser;
        use std::str::{self, FromStr};
        fn term<'a>() -> Parser<'a, u8, Expression> {
            let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
            let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);            let number = sym(b'-').opt() + integer + frac.opt();
            let number = number
                .collect()
                .map(|cs| String::from_utf8(cs.to_vec()).unwrap().parse::<f32>().unwrap())
                .map(|n| Expression::Term(Term::Number(n)));
            let node_name = none_of(b",=+-*/ ()|").repeat(1..).convert(String::from_utf8) + (sym(b'|')*one_of(b"0123456789").repeat(1..).collect().map(|cs| String::from_utf8(cs.to_vec()).unwrap().parse::<usize>().unwrap())).opt();
            let node_name = node_name.map(|(node, c)| Expression::Term(Term::Node(node, c.unwrap_or(0))));
            let term = number | node_name;
            term
        }
        fn grouped_expression<'a>() -> Parser<'a, u8, Expression> {
            call(term) | (sym(b'(') * call(expression) - sym(b')'))
        }
        fn expression<'a>() -> Parser<'a, u8, Expression> {
            let operator = one_of(b"+-*/").repeat(1).collect().convert(str::from_utf8);

            (call(grouped_expression) + operator + call(grouped_expression)).convert::<_,&'static str,_>(|((a, o), b)| Ok(Expression::Operation(Box::new(a), o.try_into()?, Box::new(b)))) | call(term)
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
                let n = Automation::new(&es);
                let n = BoxedDynamicNode::new(n);
                vec![Line::Node(k, "automation".to_string(), Some(n))]
            })
        }

        fn comment<'a>() -> Parser<'a, u8, Vec<Line>> {
            let comment = (sym(b'#') * none_of(b"\n").repeat(0..)) - sym(b'\n');
            let empty_line = (sym(b'\n')).repeat(1);
            (comment | empty_line).map(|_| vec![Line::Comment])
        }
        fn synth<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = (comment() | node_definition() | automate() | include() | edge() | node()).repeat(1..);
            p.map(|ls| ls.into_iter().flatten().collect())
        }
        fn node_definition<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = (none_of(b"\n={,")).repeat(1..).convert(String::from_utf8) - sym(b'{') + call(synth) - sym(b'}') - sym(b'\n');
            p.map(|(name, lines)| vec![Line::NodeDefinition(name, lines)])
        }
        fn node<'a>() -> Parser<'a, u8, Vec<Line>> {
            let integer = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
            let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
            let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
            let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
            let number = number
                .collect()
                .convert(str::from_utf8)
                .convert(f32::from_str);

            let parameter = (sym(b'(') * list(expression(), sym(b',')) - sym(b')')).opt();
            ((none_of(b"\n=(),")).repeat(1..).convert(String::from_utf8) - sym(b'=') + (none_of(b"[](),\n")).repeat(1..).convert(String::from_utf8) + parameter - sym(b'\n')).map(|((k, n), p)| {
                let mut edges:Vec<_> = if let Some(p) = p {
                    p.iter().enumerate().map(|(i,e)| {
                        Line::Edge(k.clone(), i as u32, e.clone())
                    }).collect()
                } else {
                    vec![]
                };
                edges.push(Line::Node(k, n, None));
                edges
            })
        }
        fn include<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = seq(b"include(") * none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b')') - sym(b'\n');
            p.convert::<_, String, _>(|k| {
                let data = std::fs::read_to_string(k.clone()).unwrap();
                Ok(DynamicGraph::parse_inner(&data)?.into_iter().chain(vec![Line::Include(k)]).collect())
            })
        }
        fn edge<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = sym(b'(') *
                none_of(b"),").repeat(1..).convert(String::from_utf8) - sym(b',') +
                one_of(b"0123456789").repeat(1..).convert(String::from_utf8).convert(|s|u32::from_str(&s)) - sym(b',') +
                expression()
                - sym(b')') - sym(b'\n');
            p.map(|((dst, i), e)| {
                vec![Line::Edge(dst, i, e)]
            })
        }
        synth().parse(data.as_bytes()).map_err(|e| format!("{:?}", e))
    }
}
enum Line {
    Node(String, String, Option<BoxedDynamicNode>),
    NodeDefinition(String, Vec<Line>),
    Edge(String, u32, Expression),
    Include(String),
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


use crate::simd_graph::*;
#[distributed_slice(MODULES)]
fn dynamic_sine() -> (String, BoxedDynamicNode) {
    ("sine".to_string(), BoxedDynamicNode::new(WaveTable::sine()))
}
#[distributed_slice(MODULES)]
fn dynamic_psine() -> (String, BoxedDynamicNode) {
    ("psine".to_string(), BoxedDynamicNode::new(WaveTable::positive_sine()))
}
#[distributed_slice(MODULES)]
fn dynamic_saw() -> (String, BoxedDynamicNode) {
    ("saw".to_string(), BoxedDynamicNode::new(WaveTable::saw()))
}
#[distributed_slice(MODULES)]
fn dynamic_square() -> (String, BoxedDynamicNode) {
    ("square".to_string(), BoxedDynamicNode::new(WaveTable::square()))
}
