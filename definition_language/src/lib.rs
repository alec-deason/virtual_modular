#![feature(hash_drain_filter)]

#[cfg(feature = "code_generation")]
pub mod code_generation;

use std::convert::{TryFrom, TryInto};

pub fn parse(data: &str) -> Result<Vec<Line>, String> {
    use pom::parser::Parser;
    use pom::parser::*;
    use std::str::{self, FromStr};

    fn node_name<'a>() -> Parser<'a, u8, String> {
        let number = one_of(b"0123456789");
        (one_of(b"abcdefghijklmnopqrstuvwxyzxy").repeat(1)
            + (one_of(b"abcdefghijklmnopqrstuvwxyzxy") | number | sym(b'_')).repeat(0..))
        .collect()
        .convert(str::from_utf8)
        .map(|s| s.to_string())
        .name("node_name")
    }
    fn node_constructor_name<'a>() -> Parser<'a, u8, String> {
        let lowercase = one_of(b"abcdefghijklmnopqrstuvwxyzxy");
        let uppercase = one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        let number = one_of(b"0123456789");
        (uppercase.repeat(1)
            + (lowercase | one_of(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") | number | sym(b'_')).repeat(0..))
        .collect()
        .convert(str::from_utf8)
        .map(|s| s.to_string())
        .name("node_constructor_name")
    }

    fn constructor_call<'a>() -> Parser<'a, u8, Expression> {
        let static_parameters = (sym(b'{')
            * none_of(b"}")
                .repeat(0..)
                .collect()
                .convert(str::from_utf8)
                .map(|s| s.to_string())
            - sym(b'}'))
        .opt();
        let inputs = (sym(b'(')
            * whitespace()
            * list(call(expression), whitespace() * sym(b',') - whitespace())
            - whitespace()
            - sym(b')'))
        .opt();

        let r = node_constructor_name() + static_parameters + inputs;

        r.map(|((constructor_name, static_parameters), inputs)| {
            Expression::Term(Term::NodeConstructor {
                name: constructor_name,
                inputs,
                static_parameters,
            })
        })
        .name("constructor_call")
    }

    fn term<'a>() -> Parser<'a, u8, Expression> {
        let port_number = one_of(b"0123456789")
            .repeat(1..)
            .collect()
            .convert(str::from_utf8)
            .convert(|v| v.parse::<usize>());
        let node_reference = (node_name() + (sym(b'|') * port_number).opt())
            .map(|(node, c)| Expression::Term(Term::Node(node, c.unwrap_or(0))));
        let constructor = constructor_call();
        let grouped_expression = sym(b'(') * call(expression) - sym(b')');
        let term = number().map(|n| Expression::Term(Term::Number(n)))
            | node_reference
            | constructor
            | grouped_expression;
        term.name("term")
    }

    fn operator_expression<'a>(
        operations: &'a [u8],
        mut next_level: impl FnMut() -> Parser<'a, u8, Expression>,
    ) -> Parser<'a, u8, Expression> {
        (next_level()
            + (whitespace()
                * one_of(operations)
                    .repeat(1)
                    .collect()
                    .convert(str::from_utf8)
                - whitespace()
                + next_level())
            .repeat(0..))
        .convert::<_, &'static str, _>(|(mut a, tail)| {
            if tail.is_empty() {
                Ok(a)
            } else {
                for (o, b) in tail {
                    a = Expression::Operation(Box::new(a), o.try_into()?, Box::new(b));
                }
                Ok(a)
            }
        })
    }
    fn expression<'a>() -> Parser<'a, u8, Expression> {
        operator_expression(b"+-", || {
            operator_expression(b"*/", || operator_expression(b"^", || call(term)))
        })
    }

    fn number<'a>() -> Parser<'a, u8, f32> {
        let integer = one_of(b"0123456789").repeat(0..);
        let frac = sym(b'.') + one_of(b"0123456789").repeat(1..);
        let exp = one_of(b"eE") + one_of(b"+-").opt() + one_of(b"0123456789").repeat(1..);
        let number = sym(b'-').opt() + integer + frac.opt() + exp.opt();
        number
            .collect()
            .convert(str::from_utf8)
            .convert(f32::from_str)
            .name("number")
    }

    fn external_parameter<'a>() -> Parser<'a, u8, Vec<Line>> {
        (none_of(b"\n=(),").repeat(1..).convert(String::from_utf8) - sym(b'=') - seq(b"e{")
            + none_of(b"}").repeat(1..).convert(String::from_utf8)
            - sym(b'}'))
        .map(|(n, k)| vec![Line::ExternalParam(n, k)])
    }

    fn comment<'a>() -> Parser<'a, u8, Vec<Line>> {
        let comment = (sym(b'#') * none_of(b"\n").repeat(0..)) - sym(b'\n');
        let empty_line = (sym(b'\n')).repeat(1);
        (comment | empty_line).map(|_| vec![Line::Comment])
    }

    fn bridge<'a>() -> Parser<'a, u8, Vec<Line>> {
        let n_in = seq(b"b{") * whitespace() * none_of(b",").repeat(1..).convert(String::from_utf8) - whitespace() - sym(b',');
        let n_out = whitespace() * none_of(b"}").repeat(1..).convert(String::from_utf8) - whitespace() - sym(b'}');
        (n_in + n_out).map(|(n_in, n_out)| vec![Line::BridgeNode(n_in, n_out)])
    }

    fn synth<'a>() -> Parser<'a, u8, Vec<Line>> {
        let p = (comment() | bridge() | external_parameter() | edge() | node()).repeat(1..) - end();
        p.map(|ls| ls.into_iter().flatten().collect())
    }

    fn node<'a>() -> Parser<'a, u8, Vec<Line>> {
        (node_name() - whitespace() - sym(b'=') - whitespace() + expression()
            - whitespace()
            - sym(b'\n'))
        .map(|(node_name, expression)| match expression {
            Expression::Term(Term::NodeConstructor {
                name: ty,
                inputs,
                static_parameters,
            }) => {
                let mut edges: Vec<_> = if let Some(inputs) = inputs {
                    inputs
                        .iter()
                        .enumerate()
                        .map(|(i, e)| e.as_lines(node_name.clone(), i))
                        .flatten()
                        .collect()
                } else {
                    vec![]
                };

                edges.push(Line::Node {
                    name: node_name,
                    ty,
                    static_parameters,
                });
                edges
            }
            _ => {
                let mut edges = expression.as_lines(node_name.clone(), 0);
                edges.push(Line::Node {
                    name: node_name,
                    ty: "C".to_string(),
                    static_parameters: None,
                });
                edges
            }
        })
        .name("node")
    }

    fn whitespace<'a>() -> Parser<'a, u8, Vec<u8>> {
        sym(b' ').repeat(0..).name("whitespace")
    }

    fn edge<'a>() -> Parser<'a, u8, Vec<Line>> {
        let p = sym(b'(') * whitespace() * node_name() - whitespace() - sym(b',') - whitespace()
            + one_of(b"0123456789")
                .repeat(1..)
                .convert(String::from_utf8)
                .convert(|s| u32::from_str(&s))
            - whitespace()
            - sym(b',')
            - whitespace()
            + expression()
            - whitespace()
            - sym(b')')
            - whitespace()
            - sym(b'\n');
        p.map(|((dst, i), e)| e.as_lines(dst, i as usize))
            .name("edge")
    }

    let parsed = synth()
        .parse(data.as_bytes())
        .map_err(|e| format!("{:?}", e));
    parsed
}

#[derive(Clone, Debug)]
pub enum Term {
    Node(String, usize),
    NodeConstructor {
        name: String,
        inputs: Option<Vec<Expression>>,
        static_parameters: Option<String>,
    },
    Number(f32),
}
#[derive(Clone, Debug)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}
impl TryFrom<&str> for Operator {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(Operator::Add),
            "-" => Ok(Operator::Sub),
            "*" => Ok(Operator::Mul),
            "/" => Ok(Operator::Div),
            "^" => Ok(Operator::Pow),
            _ => Err("Unknown operator"),
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
            Expression::Term(t) => match t {
                Term::Node(n, c) => {
                    lines.push(Line::Edge(
                        n.clone(),
                        *c as u32,
                        target_node,
                        target_port as u32,
                    ));
                }
                Term::NodeConstructor {
                    name,
                    inputs,
                    static_parameters,
                } => {
                    let n = uuid::Uuid::new_v4().to_string();
                    lines.push(Line::Node {
                        name: n.clone(),
                        ty: name.clone(),
                        static_parameters: static_parameters.clone(),
                    });
                    if let Some(inputs) = inputs {
                        for (i, e) in inputs.iter().enumerate() {
                            lines.extend(e.as_lines(n.clone(), i));
                        }
                    }
                    lines.push(Line::Edge(n, 0, target_node, target_port as u32));
                }
                Term::Number(v) => {
                    let n = uuid::Uuid::new_v4().to_string();
                    lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Constant".to_string(),
                        static_parameters: Some(format!("{}", v)),
                    });
                    lines.push(Line::Edge(n, 0, target_node, target_port as u32));
                }
            },
            Expression::Operation(a, o, b) => {
                let n = uuid::Uuid::new_v4().to_string();
                lines.extend(a.as_lines(n.clone(), 0));
                lines.extend(b.as_lines(n.clone(), 1));
                match o {
                    Operator::Add => lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Add".to_string(),
                        static_parameters: None,
                    }),
                    Operator::Sub => lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Sub".to_string(),
                        static_parameters: None,
                    }),
                    Operator::Mul => lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Mul".to_string(),
                        static_parameters: None,
                    }),
                    Operator::Div => lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Div".to_string(),
                        static_parameters: None,
                    }),
                    Operator::Pow => lines.push(Line::Node {
                        name: n.clone(),
                        ty: "Pow".to_string(),
                        static_parameters: None,
                    }),
                }
                lines.push(Line::Edge(n, 0, target_node, target_port as u32));
            }
        }
        lines
    }
}

#[derive(Debug)]
pub enum Line {
    ExternalParam(String, String),
    Node {
        name: String,
        ty: String,
        static_parameters: Option<String>,
    },
    BridgeNode(String, String),
    Edge(String, u32, String, u32),
    Comment,
}
