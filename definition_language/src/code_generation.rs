use std::collections::{HashMap, HashSet};
use virtual_modular_graph::NodeTemplate;

use crate::Line;

pub fn to_rust(
    node_templates: &HashMap<String, NodeTemplate>,
    input_code: &[Line],
) -> (String, u32) {
    let mut nodes = HashMap::new();
    nodes.insert("output".to_string(), ("Output()".to_string(), 2, 0));

    let mut edges = HashMap::new();
    let mut consumed_nodes = HashSet::new();
    let mut bridge_map = HashMap::new();
    let mut input_count: HashMap<String, i32> = HashMap::new();
    for line in input_code {
        if let Line::Node {
            name,
            ty,
            static_parameters,
        } = line
        {
            nodes.insert(
                name.clone(),
                code_for_node(ty.clone(), static_parameters, node_templates),
            );
            input_count.insert(name.clone(), 0);
        }
    }
    let mut output_code = String::new();

    let mut global_input_ports = HashSet::new();

    for line in input_code {
        match line {
            Line::BridgeNode(input_name, output_name) => {
                output_code.push_str(&format!(
                    "let ({}, {}) = Bridge::<U1>::new();\n",
                    input_name, output_name
                ));
                nodes.insert(
                    output_name.clone(),
                    (format!("{}.clone()", output_name), 0, 1),
                );
                input_count.insert(output_name.clone(), 0);
                nodes.insert(
                    input_name.clone(),
                    (format!("{}.clone()", input_name), 0, 1),
                );
                input_count.insert(input_name.clone(), 0);
            }
            Line::Edge(src_node, src_port, dst_node, dst_port) => {
                if src_node == "input" {
                    global_input_ports.insert(*src_port);
                } else {
                    *input_count.entry(dst_node.clone()).or_insert(0) += 1;
                }
                if src_node == "input" {
                    edges.entry(dst_node).or_insert_with(Vec::new).push((
                        src_node.clone(),
                        *src_port,
                        dst_port,
                    ));
                } else if consumed_nodes.contains(src_node) {
                    let bridge_name = uuid::Uuid::new_v4().to_string();
                    bridge_map
                        .entry(src_node)
                        .or_insert_with(HashSet::new)
                        .insert(src_port);
                    input_count.insert(bridge_name.clone(), 0);
                    edges.entry(dst_node).or_insert_with(Vec::new).push((
                        bridge_name.clone(),
                        *src_port,
                        dst_port,
                    ));
                    let out_ports = nodes
                        .get(src_node)
                        .unwrap_or_else(|| panic!("Node not found: {}", src_node))
                        .2;
                    nodes.insert(
                        bridge_name,
                        (format!("bridge_out___{}.clone()", src_node), 0, out_ports),
                    );
                } else {
                    consumed_nodes.insert(src_node);
                    edges.entry(dst_node).or_insert_with(Vec::new).push((
                        src_node.clone(),
                        *src_port,
                        dst_port,
                    ));
                }
            }
            _ => (),
        }
    }

    let global_input_count = global_input_ports.iter().max().map(|i| i + 1).unwrap_or(0);
    nodes.insert(
        "input".to_string(),
        ("input.clone()".to_string(), 0, global_input_count),
    );

    if global_input_count > 0 {
        output_code.push_str(&format!(
            "let (input_input, input) = Bridge::<U{}>::new();\n",
            global_input_count
        ));
    }

    for node_name in bridge_map.keys() {
        let (_, _, out_ports) = &nodes
            .get(node_name.as_str())
            .unwrap_or_else(|| panic!("Node not found: {}", node_name));
        output_code.push_str(&format!(
            "let (bridge_in___{}, bridge_out___{}) = Bridge::<U{}>::new();\n",
            node_name, node_name, out_ports
        ));
    }

    let mut processed = vec![];

    while !edges.is_empty() {
        let to_process: Vec<_> = input_count
            .drain_filter(|_node, count| *count <= 0)
            .map(|(node, _count)| node)
            .collect();
        if to_process.is_empty() {
            let processed: HashSet<_> = processed.into_iter().collect();
            let all: HashSet<_> = nodes.keys().cloned().collect();
            panic!("{:?}", all.difference(&processed));
        }
        for node_name in to_process {
            let in_edges = edges.remove(&node_name).unwrap_or_else(Vec::new);
            let (code, in_ports, out_ports) = &nodes
                .get(&node_name)
                .unwrap_or_else(|| panic!("Node not found: {}", node_name));

            let mut needed_ports: HashSet<_> = (0..*in_ports).collect();
            let mut in_code: Vec<_> = in_edges
                .into_iter()
                .map(|(src_node, src_port, dst_port)| {
                    needed_ports.remove(dst_port);
                    let (code, _in_ports, _out_ports) = &nodes[&src_node];
                    let mut code = code.clone();
                    if bridge_map.contains_key(&src_node) {
                        code = format!("Pipe({}, bridge_in___{})", code, src_node);
                    }
                    if *out_ports > 1 {
                        code = format!("Pipe({}, IndexPort::new({}))", code, src_port);
                    }
                    (*dst_port, code)
                })
                .collect();
            for port in needed_ports {
                in_code.push((port, "Constant(0.0)".to_string()));
            }
            in_code.sort_by_key(|(p, _)| *p);
            let mut i = 0;
            while i < in_code.len() {
                if in_code[i].0 < i as u32 {
                    let new_code =
                        format!("Pipe(Branch({}, {}), Add)", in_code[i - 1].1, in_code[i].1);
                    in_code[i - 1].1 = new_code;
                    in_code.remove(i);
                } else {
                    i += 1;
                }
            }
            let mut in_code: Vec<_> = in_code.into_iter().map(|(_, c)| c).collect();
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
                    return (
                        format!(
                            "{}\nPipe(Pipe(input_input, Sink::default()), {})",
                            output_code,
                            in_code.remove(0)
                        ),
                        global_input_count,
                    );
                } else {
                    return (
                        format!("{}\n{}", output_code, in_code.remove(0)),
                        global_input_count,
                    );
                }
            }

            let code = if in_code.is_empty() {
                code.clone()
            } else {
                format!("Pipe({}, {})", in_code[0], code)
            };

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

fn code_for_node(
    node_type: String,
    static_parameters: &Option<String>,
    node_templates: &HashMap<String, NodeTemplate>,
) -> (String, u32, u32) {
    let (input_port_count, output_port_count) = {
        match node_type.as_str() {
            "Output" => (2, 0),
            _ => {
                let template = node_templates
                    .get(node_type.as_str())
                    .unwrap_or_else(|| panic!("{}", node_type))
                    .clone();
                (
                    template.node.input_len() as u32,
                    template.node.output_len() as u32,
                )
            }
        }
    };

    let mut src_node_code = match node_type.as_str() {
        "Constant" => format!("Constant({:?})", static_parameters.as_ref().unwrap().parse::<f32>().unwrap()),
        "Output" => "Output()".to_string(),
        _ => node_templates[node_type.as_str()].code.clone(),
    };

    if node_type != "Constant" {
        if let Some(parameters) = static_parameters {
            src_node_code = format!(
                r##"
                    {{
                        let mut n = {};
                        n.set_static_parameters("{}").unwrap();
                        n
                    }}
            "##,
                src_node_code, parameters
            );
        }
    }

    (src_node_code, input_port_count, output_port_count)
}
