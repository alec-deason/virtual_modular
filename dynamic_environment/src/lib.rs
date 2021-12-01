use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use generic_array::{
    arr,
    typenum::{U0, U1, U2},
};
use virtual_modular_core_nodes::*;
use virtual_modular_definition_language::{parse, Line};
use virtual_modular_graph::{Node, NodeTemplate, PortCount, Ports, WrappedNode, BLOCK_SIZE};

#[derive(Clone)]
pub struct DynamicGraphBuilder {
    pub templates: HashMap<String, NodeTemplate>,
}

impl Default for DynamicGraphBuilder {
    fn default() -> Self {
        Self {
            templates: std_nodes(),
        }
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

#[derive(Clone, Default)]
pub struct DynamicGraph {
    nodes: HashMap<
        String,
        (
            Box<dyn WrappedNode>,
            Vec<[f32; BLOCK_SIZE]>,
            Vec<[f32; BLOCK_SIZE]>,
            String,
        ),
    >,
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
        for (_, input_buffer, _, _) in self.nodes.values_mut() {
            input_buffer
                .iter_mut()
                .for_each(|v| v.iter_mut().for_each(|v| *v = 0.0));
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
                        n.1[*dst_i]
                            .iter_mut()
                            .zip(&v)
                            .for_each(|(lhs, rhs)| *lhs += rhs);
                    } else {
                        return Err(format!("Unknown node {}", node));
                    }
                }
            }
            if node != "output" {
                if let Some((n, input, output, _)) = self.nodes.get_mut(node) {
                    n.process(input, output);
                } else if !self.external_input_nodes.contains_key(node) {
                    return Err(format!("Unknown node {}", node));
                }
            }
        }
        Ok(arr![[f32; BLOCK_SIZE]; self.output.0, self.output.1])
    }

    pub fn add_node<
        A: PortCount + 'static,
        B: PortCount + 'static,
        N: Node<Input = A, Output = B> + Clone + 'static,
    >(
        &mut self,
        name: String,
        ty: String,
        n: N,
    ) {
        self.add_boxed_node(name, ty, n.into());
    }
    pub fn add_boxed_node(&mut self, name: String, ty: String, n: Box<dyn WrappedNode>) {
        let input = vec![[0.0; BLOCK_SIZE]; n.input_len()];
        let output = vec![[0.0; BLOCK_SIZE]; n.output_len()];
        self.nodes.insert(name, (n, input, output, ty));
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
            .filter_map(|(k, (_, _, _, ty))| {
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
            .retain(|k, v| nodes.get(k).map(|ty| &v.3 == ty).unwrap_or(false));
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
                                    let mut n = template.node.clone();
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
            Ok(n.2[i])
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
