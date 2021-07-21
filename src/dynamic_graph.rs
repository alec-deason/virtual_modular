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
    fn post_process(&mut self);
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

    fn post_process(&mut self) {
        self.0.post_process();
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
    fn as_lines(&self, target_node: String, target_port: usize) -> Vec<Line> {
        let mut lines = vec![];
        match self {
            Expression::Term(t) => {
                match t {
                    Term::Node(n, c) => {
                        lines.push(Line::Edge(n.clone(), *c as u32, target_node, target_port as u32));
                    }
                    Term::Number(v) => {
                        let n = uuid::Uuid::new_v4().to_string();
                        lines.push(Line::Node(n.clone(), "c".to_string(), Some(BoxedDynamicNode::new(Constant(f32x8::splat(*v))))));
                        lines.push(Line::Edge(n, 0, target_node, target_port as u32));
                    }
                }
            }
            Expression::Operation(a, o, b) => {
                let n = uuid::Uuid::new_v4().to_string();
                lines.extend(a.as_lines(n.clone(), 0));
                lines.extend(b.as_lines(n.clone(), 1));
                match o {
                    Operator::Add => lines.push(Line::Node(n.clone(), "add".to_string(), None)),
                    Operator::Sub => lines.push(Line::Node(n.clone(), "sub".to_string(), None)),
                    Operator::Mul => lines.push(Line::Node(n.clone(), "mul".to_string(), None)),
                    Operator::Div => lines.push(Line::Node(n.clone(), "div".to_string(), None)),
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
        o[0] = (r.0).0;
        o[1] = (r.0).1;
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

#[distributed_slice]
pub static MODULES: [(&'static str, fn() -> BoxedDynamicNode)] = [..];

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: HashMap<String, (BoxedDynamicNode, String)>,
    dynamic_nodes: HashMap<String, Box<DynamicGraph>>,
    pub external_inputs: Arc<Mutex<HashMap<String, f32>>>,
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
    pub fn process(&mut self) -> Value<(f32x8, f32x8)> {
        if let Ok(ei) = self.external_inputs.lock() {
            for (k, v) in self.external_input_nodes.values_mut() {
                *v = *ei.get(k).unwrap_or(&0.0);
            }
        }
        self.output = (f32x8::splat(0.0), f32x8::splat(0.0));
        for (n,v) in &self.reset_edges {
            if let Some(node) = self.nodes.get_mut(n) {
                node.0.set(*v, f32x8::splat(0.0));
            } else {
                self.dynamic_nodes.get_mut(n).expect(&format!("no definition for {}",n)).input[*v] = f32x8::splat(0.0);
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
        Value((self.output.0, self.output.1))
    }

    pub fn add_node<A: DynamicValue + Default + 'static, B:DynamicValue + Default + 'static, N: Node<Input=A, Output=B> + Clone + 'static>(&mut self, name: String, ty: String, n: N) {
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
        let module_constructors: HashMap<_,_> = MODULES.iter().enumerate().map(|(i, (n, _f))| (n.to_string(), i)).collect();
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
        self.external_input_nodes.clear();
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
                            if let Some(i) = module_constructors.get(ty) {
                                let mut n = MODULES[*i].1();
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

    pub fn parse(data: &str) -> Result<Self, String> {
        let mut g = DynamicGraph::default();
        let l = Self::parse_inner(data)?;

        g.reparse(&l)?;
        Ok(g)
    }


    pub fn parse_inner(data: &str) -> Result<Vec<Line>, String> {
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

        fn sumseq<'a>() -> Parser<'a, u8, Vec<Line>> {
            let name = (none_of(b"\n=[](),")).repeat(1..).convert(String::from_utf8) - seq(b"=sumseq");
            let r = name + sym(b'[') * list(sym(b'(') * number() - sym(b',') + number() - sym(b')'), sym(b',')) - sym(b']');
            let parameter = (sym(b'(') * list(expression(), sym(b',')) - sym(b')')).opt();
            let r = r + parameter;
            r.map(|((k, ns), p)| {
                let ns:Vec<_> = ns.into_iter().map(|(c,p)| (c as u32, p)).collect();
                let ns:[(u32, f32); 4] = ns.try_into().unwrap();
                let n = SumSequencer::new(ns);
                let n = BoxedDynamicNode::new(n);
                let mut edges:Vec<_> = if let Some(p) = p {
                    p.iter().enumerate().map(|(i,e)| {
                        e.as_lines(k.clone(), i)
                    }).flatten().collect()
                } else {
                    vec![]
                };
                edges.push(Line::Node(k, "sum_sequencer".to_string(), Some(n)));
                edges
            })
        }
        fn sequencer<'a>() -> Parser<'a, u8, Vec<Line>> {
            let name = (none_of(b"\n=[](),")).repeat(1..).convert(String::from_utf8) - seq(b"=seq");
            let r = name + sym(b'[') * list(number(), sym(b',')) - sym(b']');
            let parameter = (sym(b'(') * list(expression(), sym(b',')) - sym(b')')).opt();
            let r = r + parameter;
            r.map(|((k, ns), p)| {
                let n = Sequencer::new(ns);
                let n = BoxedDynamicNode::new(n);
                let mut edges:Vec<_> = if let Some(p) = p {
                    p.iter().enumerate().map(|(i,e)| {
                        e.as_lines(k.clone(), i)
                    }).flatten().collect()
                } else {
                    vec![]
                };
                edges.push(Line::Node(k, "sequencer".to_string(), Some(n)));
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
                let (a,b) = Bridge::<Value<(f32,)>>::new((0.0f32,));
                vec![
                    Line::Node(n_in, "bridge_in".to_string(), Some(BoxedDynamicNode::new(a))),
                    Line::Node(n_out, "bridge_out".to_string(), Some(BoxedDynamicNode::new(Pipe(b, Strip::<Value<((f32,),)>, Value<(f32,)>>::default())))),
                ]
            })
        }
        fn synth<'a>() -> Parser<'a, u8, Vec<Line>> {
            let p = (comment() | bridge() | external_parameter() | sequencer() | sumseq() | node_definition() | automate() | include() | edge() | node()).repeat(1..);
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
                        e.as_lines(k.clone(), i)
                    }).flatten().collect()
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
                e.as_lines(dst, i as usize)
            })
        }
        synth().parse(data.as_bytes()).map_err(|e| format!("{:?}", e))
    }
}
#[derive(Debug)]
pub enum Line {
    ExternalParam(String, String),
    Node(String, String, Option<BoxedDynamicNode>),
    NodeDefinition(String, Vec<Line>),
    Edge(String, u32, String, u32),
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


use crate::simd_graph::*;

macro_rules! dynamic_node {
    ($name:expr, $static_name:ident, $constructor:expr) => {
            concat_idents::concat_idents!(fn_name = "__MODULE_fn_", $name {
                fn fn_name() -> BoxedDynamicNode {
                    BoxedDynamicNode::new($constructor)
                }

                #[distributed_slice(MODULES)]
                static $static_name: (&'static str, fn() -> BoxedDynamicNode) = ($name, fn_name);
        });
    }
}
dynamic_node!("sine", __MODULE_sine, WaveTable::sine());
dynamic_node!("psine", __MODULE_psine, WaveTable::positive_sine());
dynamic_node!("saw", __MODULE_saw, WaveTable::saw());
dynamic_node!("square", __MODULE_square, WaveTable::square());
dynamic_node!("noise", __MODULE_noise, Pipe(Constant(f32x8::splat(440.0)), WaveTable::noise()));
dynamic_node!("rnoise", __MODULE_rnoise, curry(Constant(f32x8::splat(440.0)), crate::instruments::RetriggeringWaveTable::new(WaveTable::noise())));
dynamic_node!("ad", __MODULE_ad, InlineADEnvelope::default());
dynamic_node!("asd", __MODULE_asd, InlineASDEnvelope::default());
dynamic_node!("lpf", __MODULE_lpf, Biquad::lowpass());
dynamic_node!("hpf", __MODULE_hpf, Biquad::highpass());
dynamic_node!("bpf", __MODULE_bpf, Biquad::bandpass());
dynamic_node!("eq", __MODULE_eq, Biquad::peaking_eq());
dynamic_node!("sh", __MODULE_sh, SampleAndHold::default());
dynamic_node!("ap", __MODULE_ap, AllPass::default());
dynamic_node!("comb", __MODULE_comb, Comb::new(DelayLine::new(2)));
dynamic_node!("pd", __MODULE_pd, PulseDivider::default());
dynamic_node!("compressor", __MODULE_compressor, Compressor::default());
dynamic_node!("limiter", __MODULE_limiter, Limiter::new(1.0));
dynamic_node!("log", __MODULE_log, Log::<Value<(f32x8,)>>::default());
dynamic_node!("stutter_reverb", __MODULE_stutter_reverb, Stutter::rand_pan(50, 0.35));
dynamic_node!("soft_clip", __MODULE_soft_clip, SoftClip);
dynamic_node!("looper", __MODULE_looper, Looper::default());
dynamic_node!("turing_machine", __MODULE_turing_machine, crate::instruments::InlineSoftTuringMachine::default());
dynamic_node!("smooth_leader", __MODULE_smooth_leader, crate::instruments::SmoothLeader::new());
dynamic_node!("add", __MODULE_add, Add::<f32x8, f32x8>::default());
dynamic_node!("sub", __MODULE_sub, Sub::<f32x8, f32x8>::default());
dynamic_node!("mul", __MODULE_mul, Mul::<f32x8, f32x8>::default());
dynamic_node!("div", __MODULE_div, Div::<f32x8, f32x8>::default());
dynamic_node!("split", __MODULE_split, Split);
dynamic_node!("rescale", __MODULE_rescale, ModulatedRescale);
dynamic_node!("toggle", __MODULE_toggle, Toggle::default());
dynamic_node!("imp", __MODULE_impulse, Impulse::default());
dynamic_node!("pimp", __MODULE_pimpulse, ProbImpulse::default());
dynamic_node!("aimp", __MODULE_aimpulse, AccumulatorImpulse::default());
dynamic_node!("pow", __MODULE_pow, ToPower);
dynamic_node!("reg", __MODULE_reg, Register::default());
dynamic_node!("comp", __MODULE_comp, Comparator);
dynamic_node!("svfl", __MODULE_svf_low, SimperSvf::low_pass());
dynamic_node!("svfh", __MODULE_svf_high, SimperSvf::high_pass());
dynamic_node!("svfb", __MODULE_svf_band, SimperSvf::band_pass());
dynamic_node!("svfn", __MODULE_svf_notch, SimperSvf::notch());
dynamic_node!("portamento", __MODULE_portamento, Portamento::default());
dynamic_node!("reverb", __MODULE_reverb, Reverb::new());
dynamic_node!("fold", __MODULE_folder, Folder);
dynamic_node!("delay", __MODULE_modable_delay, ModableDelay::new());
dynamic_node!("pulse_on_load", __MODULE_pulse_on_load, PulseOnLoad::default());
