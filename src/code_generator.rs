use std::{
    collections::{HashMap, HashSet},
};
use indexmap::IndexMap;

use crate::{
    dynamic_graph::{Line, NodeParameters, BoxedDynamicNode, DynamicNode, DynamicGraphBuilder},
};

#[derive(Clone, Debug)]
enum Edge {
    Unresolved(Vec<UnresolvedEdge>),
    Resolved(ResolvedEdge),
}
#[derive(Clone, Debug)]
enum BridgeUse {
    Input,
    Output,
    NA
}
#[derive(Clone, Debug)]
struct UnresolvedEdge {
    bridge_use: BridgeUse,
    dst_port: u32,
    src_node: String,
    src_port: u32
}
#[derive(Clone, Debug)]
struct ResolvedEdge {
    dst_port: u32,
    code: String
}
pub fn to_rust(builder: &DynamicGraphBuilder, input_code: &[Line]) -> (String, u32) {
    let mut nodes = HashMap::new();
    nodes.insert("output".to_string(), ("Output()".to_string(), 2, 0));



    let mut edges = HashMap::new();
    let mut consumed_nodes = HashSet::new();
    let mut bridge_map = HashMap::new();
    let mut input_count:HashMap<String, i32> = HashMap::new();
    for line in input_code {
        if let Line::Node(node_name, node_type, parameters) = line {
            nodes.insert(node_name.to_string(), code_for_node(node_type.to_string(), parameters.clone(), builder));
            input_count.insert(node_name.clone(), 0);
            println!("init_node: {}", node_name);
        }
    }
    let mut output_code = String::new();

    let mut global_input_ports = HashSet::new();

    for line in input_code {
        match line {
            Line::BridgeNode(input_name, output_name) => {
                println!("let ({}, {}) = Bridge::<U1>::new();\n", input_name, output_name);
                output_code.push_str(&format!("let ({}, {}) = Bridge::<U1>::new();\n", input_name, output_name));
                nodes.insert(output_name.clone(), (format!("{}.clone()", output_name), 0, 1));
                input_count.insert(output_name.clone(), 0);
                nodes.insert(input_name.clone(), (format!("{}.clone()", input_name), 0, 1));
                input_count.insert(input_name.clone(), 0);
            }
            Line::Edge(src_node, src_port, dst_node, dst_port) => {
                if src_node == "input" {
                    global_input_ports.insert(*src_port);
                } else {
                    *input_count.entry(dst_node.clone()).or_insert(0) += 1;
                }
                if src_node == "input" {
                    edges.entry(dst_node).or_insert_with(Vec::new).push((src_node.clone(), *src_port, dst_port));
                } else if consumed_nodes.contains(src_node) {
                    let bridge_name = uuid::Uuid::new_v4().to_string();
                    bridge_map.entry(src_node).or_insert_with(HashSet::new).insert(src_port);
                    edges.entry(dst_node).or_insert_with(Vec::new).push((bridge_name.clone(), 0, dst_port));
                    input_count.insert(bridge_name.clone(), 0);
                    nodes.insert(bridge_name, (format!("bridge_out___{}_{}.clone()", src_node, src_port), 0, 1));
                } else {
                    consumed_nodes.insert(src_node);
                    edges.entry(dst_node).or_insert_with(Vec::new).push((src_node.clone(), *src_port, dst_port));
                }
            }
            _ => ()
        }
    }

    let global_input_count = global_input_ports.iter().max().map(|i| i+1).unwrap_or(0);
    nodes.insert("input".to_string(), ("input.clone()".to_string(), 0, global_input_count));

    if global_input_count > 0 {
        output_code.push_str(&format!("let (input_input, input) = Bridge::<U{}>::new();\n", global_input_count));
    }

    for (node_name, ports) in &bridge_map {
        for port in ports {
            output_code.push_str(&format!("let (bridge_in___{}_{}, bridge_out___{}_{}) = Bridge::<U1>::new();\n", node_name, port, node_name, port));
        }
    }

    let mut processed = vec![];

    while !edges.is_empty() {
        let to_process:Vec<_> = input_count.drain_filter(|node, count| {
            *count <= 0
        }).map(|(node, count)| node).collect();
        if to_process.is_empty() {
            let processed: HashSet<_> = processed.into_iter().collect();
            let all: HashSet<_> = nodes.keys().cloned().collect();
            panic!("{:?}", all.difference(&processed));
        }
        for node_name in to_process {
            let in_edges = edges.remove(&node_name).unwrap_or_else(Vec::new);
            let (code, in_ports, out_ports) = &nodes.get(&node_name).expect(&format!("Node not found: {}", node_name));

            let mut needed_ports: HashSet<_> = (0..*in_ports).collect();
            let mut in_code: Vec<_> = in_edges.into_iter().map(|(src_node, src_port, dst_port)| {
                needed_ports.remove(dst_port);
                let (code, in_ports, out_ports) = &nodes[&src_node];
                let mut code = code.clone();
                if *out_ports > 1 {
                    code = format!("Pipe({}, IndexPort::new({}))", code, src_port);
                }
                (*dst_port, code)
            }).collect();
            for port in needed_ports {
                in_code.push((port, "Constant(0.0)".to_string()));
            }
            in_code.sort_by_key(|(p, _)| *p);
            let mut i = 0;
            while i < in_code.len() {
                if in_code[i].0 < i as u32 {
                    let new_code = format!("Pipe(Branch({}, {}), Add)", in_code[i-1].1, in_code[i].1);
                    in_code[i-1].1 = new_code;
                    in_code.remove(i);
                } else {
                    i += 1;
                }
            }
            let mut in_code:Vec<_> = in_code.into_iter().map(|(_, c)| c).collect();
            while in_code.len() > 1 {
                let mut new_in_code = vec![];
                for es in in_code.chunks(2) {
                    if es.len() == 1 {
                        new_in_code.push(es[0].clone());
                    } else {
                        new_in_code.push(format!("Branch({}, {})", es[0], es[1]));
                    }
                }
                in_code = new_in_code;
            }

            if node_name == "output" {
                if global_input_count > 0 {
                    return (format!("{}\nPipe(Pipe(input_input, Sink::default()), {})", output_code, in_code.remove(0)), global_input_count);
                } else {
                    return (format!("{}\n{}", output_code, in_code.remove(0)), global_input_count);
                }
            }

            let mut code = if in_code.is_empty() {
                code.clone()
            } else {
                format!("Pipe({}, {})", in_code[0], code)
            };

            if let Some(bridge_ports) = bridge_map.get(&node_name) {
                let mut outputs:Vec<_> = (0..*out_ports).map(|i| {
                    if bridge_ports.contains(&&i) {
                        (format!("bridge_in___{}_{}", node_name, i), 1)
                    } else {
                        ("Identity".to_string(), 1)
                    }
                }).collect();
                while outputs.len() > 1 {
                    let mut new_outputs = vec![];
                    for es in outputs.chunks(2) {
                        if es.len() == 1 {
                            new_outputs.push(es[0].clone());
                        } else {
                            new_outputs.push((format!("Stack::new({}, {})", es[0].0, es[1].0), es[0].1+es[1].1));
                        }
                    }
                    outputs = new_outputs;
                }
                code = format!("Pipe({}, {})", code, outputs[0].0);
            }
            nodes.get_mut(&node_name).unwrap().0 = code;
            for (dst_node, in_edges) in &edges {
                for (src_node, _, _) in in_edges {
                    if src_node == &node_name {
                        if let Some(c) = input_count.get_mut(*dst_node) {
                            *c -= 1;
                        } else {
                            panic!("{}: {}", dst_node, &nodes[*dst_node].0);
                        }
                    }
                }
            }
            processed.push(node_name.clone());
        }
    }
    unreachable!()
}

fn code_for_node(node_type: String, parameters: Option<NodeParameters>, builder: &DynamicGraphBuilder) -> (String, u32, u32) {
    let (input_port_count, output_port_count) = {
        match node_type.as_str() {
          "Output" => (2, 0),
          "pattern_sequencer" => (1, 4),
          "sum_sequencer" => (1, 1),
          "automation" => (0, 1),
          "Constant" => (0, 1),
          _ => {
            let n = builder.templates.get(node_type.as_str()).expect(node_type.as_str()).0.clone();
            (n.input_len() as u32, n.output_len() as u32)
          }
        }
    };

    let src_node_code = match node_type.as_str() {
        "Constant" => {
            if let Some(NodeParameters::Number(n)) = parameters {
                format!("Constant({:.4})", n)
            } else {
                panic!();
            }
        }
        "pattern_sequencer" => {
            if let Some(NodeParameters::PatternSequence(p)) = parameters {
                format!("PatternSequencer::new({})", p.to_rust())
            } else {
                panic!();
            }
        }
        "sum_sequencer" => {
            if let Some(NodeParameters::SumSequence(p)) = parameters {
                format!("SumSequencer::new({:?})", p)
            } else {
                panic!();
            }
        }
        "automation" => {
            if let Some(NodeParameters::AutomationSequence(p)) = parameters {
                let p: Vec<_> = p.iter().map(|p| p.to_rust()).collect();
                format!("Automation::new(&[{}])", p.join(","))
            } else {
                panic!();
            }
        }
        "Output" => {
            "Output()".to_string()
        }
        _ => builder.templates[node_type.as_str()].1.to_string()
    };

    (src_node_code, input_port_count, output_port_count)
}
