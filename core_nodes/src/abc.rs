use generic_array::{arr, typenum::*};
use std::collections::HashMap;
use virtual_modular_graph::{Node, Ports, BLOCK_SIZE};

#[derive(Clone, Debug)]
pub struct ABCSequence {
    line: Vec<abc_parser::datatypes::MusicSymbol>,
    key: HashMap<char, abc_parser::datatypes::Accidental>,
    idx: usize,
    clock: u32,
    sounding: Option<f32>,
    current_duration: f32,
    triggered: bool,
}

impl ABCSequence {
    pub fn new(tune: &str) -> Option<Self> {
        let parsed = abc_parser::abc::tune(tune).ok()?;
        let key = parsed
            .header
            .info
            .iter()
            .find(|f| f.0 == 'K')
            .map(|f| f.1.clone())
            .unwrap_or("C".to_string());
        let key: HashMap<_, _> = match key.as_str() {
            "C" => vec![],
            "G" => vec![('F', abc_parser::datatypes::Accidental::Sharp)],
            _ => panic!(),
        }
        .into_iter()
        .collect();
        let mut line: Vec<_> = parsed
            .body
            .unwrap()
            .music
            .into_iter()
            .map(|l| l.symbols.clone())
            .flatten()
            .collect();
        line.retain(|s| match s {
            abc_parser::datatypes::MusicSymbol::Rest(abc_parser::datatypes::Rest::Note(..)) => true,
            abc_parser::datatypes::MusicSymbol::Note { .. } => true,
            _ => false,
        });
        let mut r = Self {
            line,
            key,
            idx: 0,
            clock: 0,
            current_duration: 0.0,
            sounding: None,
            triggered: false,
        };
        let dur = r.duration(0);
        r.clock = dur;
        r.current_duration = dur as f32 / 24.0;
        r.sounding = r.freq(0);
        Some(r)
    }
}

fn accidental_to_freq_multiplier(accidental: &abc_parser::datatypes::Accidental) -> f32 {
    let semitones = match accidental {
        abc_parser::datatypes::Accidental::Sharp => 1,
        abc_parser::datatypes::Accidental::Flat => -1,
        abc_parser::datatypes::Accidental::Natural => 0,
        abc_parser::datatypes::Accidental::DoubleSharp => 2,
        abc_parser::datatypes::Accidental::DoubleFlat => -2,
    };
    2.0f32.powf((semitones * 100) as f32 / 1200.0)
}

impl ABCSequence {
    fn freq(&self, idx: usize) -> Option<f32> {
        if let abc_parser::datatypes::MusicSymbol::Note {
            note,
            octave,
            accidental,
            ..
        } = self.line[idx]
        {
            if accidental.is_some() {
                todo!()
            }
            let mut base = match note {
                abc_parser::datatypes::Note::C => 16.35,
                abc_parser::datatypes::Note::D => 18.35,
                abc_parser::datatypes::Note::E => 20.60,
                abc_parser::datatypes::Note::F => 21.83,
                abc_parser::datatypes::Note::G => 24.50,
                abc_parser::datatypes::Note::A => 27.50,
                abc_parser::datatypes::Note::B => 30.87,
            };
            let accidental = match note {
                abc_parser::datatypes::Note::C => self.key.get(&'C'),
                abc_parser::datatypes::Note::D => self.key.get(&'D'),
                abc_parser::datatypes::Note::E => self.key.get(&'E'),
                abc_parser::datatypes::Note::F => self.key.get(&'F'),
                abc_parser::datatypes::Note::G => self.key.get(&'G'),
                abc_parser::datatypes::Note::A => self.key.get(&'A'),
                abc_parser::datatypes::Note::B => self.key.get(&'B'),
            };
            if let Some(accidental) = accidental {
                base *= accidental_to_freq_multiplier(accidental);
            }
            Some(base * 2.0f32.powi(octave as i32 + 2))
        } else {
            panic!()
        }
    }

    fn duration(&self, idx: usize) -> u32 {
        match self.line[idx] {
            abc_parser::datatypes::MusicSymbol::Rest(abc_parser::datatypes::Rest::Note(
                _length,
            )) => {
                unimplemented!()
            }
            abc_parser::datatypes::MusicSymbol::Note { length, .. } => (length * 24.0) as u32,
            _ => panic!("{:?}", self.line[idx]),
        }
    }
}

impl Node for ABCSequence {
    type Input = U1;
    type Output = U4;
    #[inline]
    fn process(&mut self, input: Ports<Self::Input>) -> Ports<Self::Output> {
        let trigger = input[0];

        let mut r_freq = [0.0f32; BLOCK_SIZE];
        let mut r_gate = [0.0f32; BLOCK_SIZE];
        let mut r_eoc = [0.0f32; BLOCK_SIZE];
        let mut r_dur = [0.0f32; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            if trigger[i] > 0.5 {
                if !self.triggered {
                    self.triggered = true;
                    self.clock -= 1;
                }
            } else {
                self.triggered = false;
            }
            if self.clock == 0 {
                self.idx = self.idx + 1;
                if self.idx >= self.line.len() {
                    self.idx = 0;
                    r_eoc[i] = 1.0;
                }
                self.clock = self.duration(self.idx);
                self.current_duration = self.clock as f32 / 24.0;
                self.sounding = self.freq(self.idx);
                r_gate[i] = 0.0;
            } else {
                r_gate[i] = if self.sounding.is_some() { 1.0 } else { 0.0 };
            }
            r_freq[i] = self.sounding.unwrap_or(0.0);
            r_dur[i] = self.current_duration;
        }

        arr![[f32; BLOCK_SIZE]; r_freq, r_gate, r_eoc, r_dur]
    }
}
