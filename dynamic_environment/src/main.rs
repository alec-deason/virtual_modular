#![feature(exact_size_is_empty)]
use ringbuf::{Consumer, RingBuffer};
#[cfg(feauture = "midi")]
use std::convert::TryFrom;
use std::sync::Arc;

use virtual_modular::InstrumentSynth;
use virtual_modular_dynamic_environment::DynamicGraphBuilder;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() {
    let synth_path = std::env::args().nth(1);

    let builder = InstrumentSynth::builder();

    let graph = if let Some(synth_data) = synth_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
    {
        DynamicGraphBuilder::default().parse(&synth_data).unwrap()
    } else {
        panic!("Could not read synth definition file {:?}", synth_path)
    };

    #[cfg(feauture = "midi")]
    launch_midi_listener(&graph);

    let reload_data = Arc::clone(&graph.reload_data);
    let watch_list = Arc::clone(&graph.watch_list);

    let mut synth = builder.build_with_synth(graph);

    let cpal_host = cpal::default_host();

    let device = cpal_host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();

    let sample_rate = config.sample_rate().0 as f32;
    synth.set_sample_rate(sample_rate);

    let rb = RingBuffer::<(f32, f32)>::new(4048);
    let (mut prod, cons) = rb.split();

    std::thread::spawn(move || {
        let mut left = vec![0.0; 32];
        let mut right = vec![0.0; 32];
        loop {
            synth.process(&mut left, &mut right);
            let mut to_push = left.iter().zip(&right).map(|(l, r)| (*l, *r));
            loop {
                prod.push_iter(&mut to_push);
                if to_push.is_empty() {
                    break;
                } else {
                    std::thread::sleep(std::time::Duration::from_secs_f32(10.0 / 44000.0));
                }
            }
        }
    });

    let _cpal_stream = match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), cons),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), cons),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), cons),
    };

    let mut last_change = std::time::SystemTime::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(300));
        if let Some(synth_path) = &synth_path {
            let mut ps: Vec<_> = watch_list.lock().unwrap().iter().cloned().collect();
            ps.push(synth_path.to_string());
            let mut needs_reload = false;
            for p in ps {
                if let Ok(metadata) = std::fs::metadata(&p) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > last_change {
                            needs_reload = true;
                            last_change = modified;
                            break;
                        }
                    }
                }
            }
            if needs_reload {
                reload_data
                    .lock()
                    .unwrap()
                    .replace(std::fs::read_to_string(synth_path).unwrap());
            }
        }
    }
}

#[cfg(feauture = "midi")]
fn launch_midi_listener(graph: &DynamicGraph) {
    let inputs = Arc::clone(graph.external_inputs);
    std::thread::spawn(move || {
        let mut voices = std::collections::HashMap::new();
        for c in 0..10 {
            voices.insert(c, (0..4).map(|i| (i, None)).collect::<Vec<_>>());
        }
        let mut current_voice = 0;

        let (client, _status) =
            jack::Client::new("virtual_modular_midi", jack::ClientOptions::NO_START_SERVER)
                .unwrap();
        let shower = client
            .register_port("midi_in", jack::MidiIn::default())
            .unwrap();
        let cback = move |_: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
            {
                let mut inputs = inputs.lock().unwrap();
                for e in shower.iter(ps) {
                    let message = wmidi::MidiMessage::try_from(e.bytes).unwrap();
                    match message {
                        wmidi::MidiMessage::NoteOn(c, n, v) => {
                            if let Some(voices) = voices.get_mut(&c.index()) {
                                let mut consumed = None;
                                let mut aval = None;
                                for (i, (j, f)) in voices.iter_mut().enumerate() {
                                    if Some(n) == *f {
                                        consumed = Some(i);
                                        break;
                                    } else if f.is_none() && aval.is_none() {
                                        aval = Some((i, *j));
                                    }
                                }
                                let velocity = u8::try_from(v).unwrap();
                                if let Some(i) = consumed {
                                    inputs
                                        .entry(format!("midi_{}_voice_{}_velocity", c.index(), i))
                                        .or_insert_with(Vec::new)
                                        .push(u8::try_from(v).unwrap() as f32 / 127.0);
                                } else {
                                    let (aval, voice) = if let Some((aval, voice)) = aval {
                                        (aval, voice)
                                    } else {
                                        (0, voices[0].0)
                                    };
                                    inputs
                                        .entry(format!("midi_{}_voice_{}_freq", c.index(), voice))
                                        .or_insert_with(Vec::new)
                                        .push(n.to_freq_f32());
                                    inputs
                                        .entry(format!(
                                            "midi_{}_voice_{}_velocity",
                                            c.index(),
                                            voice
                                        ))
                                        .or_insert_with(Vec::new)
                                        .push(velocity as f32 / 127.0);
                                    if velocity > 0 {
                                        voices.remove(aval);
                                        voices.insert(0, (voice, Some(n)));
                                        voices.rotate_left(1);
                                    } else {
                                        for (j, f) in voices.iter_mut() {
                                            if Some(n) == *f {
                                                *f = None;
                                                inputs
                                                    .entry(format!(
                                                        "midi_{}_voice_{}_velocity",
                                                        c.index(),
                                                        j
                                                    ))
                                                    .or_insert_with(Vec::new)
                                                    .push(0.0);
                                            }
                                        }
                                    }
                                }
                            }
                            if u8::try_from(n).unwrap() != current_voice {
                                current_voice = u8::try_from(n).unwrap();
                                inputs
                                    .entry(format!("midi_mono_{}_freq", c.index()))
                                    .or_insert_with(Vec::new)
                                    .push(n.to_freq_f32());
                                inputs
                                    .entry(format!("midi_mono_{}_velocity", c.index()))
                                    .or_insert_with(Vec::new)
                                    .push(0.0);
                                inputs
                                    .entry(format!("midi_mono_{}_velocity", c.index()))
                                    .or_insert_with(Vec::new)
                                    .push(u8::try_from(v).unwrap() as f32 / 127.0);
                            }
                            inputs
                                .entry(format!("midi_{}_freq", c.index()))
                                .or_insert_with(Vec::new)
                                .push(n.to_freq_f32());
                            inputs
                                .entry(format!("midi_{}_velocity", c.index()))
                                .or_insert_with(Vec::new)
                                .push(u8::try_from(v).unwrap() as f32 / 127.0);
                        }
                        wmidi::MidiMessage::NoteOff(c, n, ..) => {
                            if let Some(voices) = voices.get_mut(&c.index()) {
                                for (j, f) in voices.iter_mut() {
                                    if Some(n) == *f {
                                        *f = None;
                                        inputs
                                            .entry(format!(
                                                "midi_{}_voice_{}_velocity",
                                                c.index(),
                                                j
                                            ))
                                            .or_insert_with(Vec::new)
                                            .push(0.0);
                                    }
                                }
                            }
                            if u8::try_from(n).unwrap() == current_voice {
                                current_voice = 0;
                                inputs
                                    .entry(format!("midi_mono_{}_freq", c.index()))
                                    .or_insert_with(Vec::new)
                                    .push(n.to_freq_f32());
                                inputs
                                    .entry(format!("midi_mono_{}_velocity", c.index()))
                                    .or_insert_with(Vec::new)
                                    .push(0.0);
                            }
                            inputs
                                .entry(format!("midi_{}_freq", c.index()))
                                .or_insert_with(Vec::new)
                                .push(n.to_freq_f32());
                            inputs
                                .entry(format!("midi_{}_velocity", c.index()))
                                .or_insert_with(Vec::new)
                                .push(0.0);
                        }
                        wmidi::MidiMessage::PitchBendChange(c, b) => {
                            inputs
                                .entry(format!("midi_{}_pitch_bend", c.index()))
                                .or_insert_with(Vec::new)
                                .push(
                                    (u16::try_from(b).unwrap() as f32 / 2.0f32.powi(14) - 0.5)
                                        * 2.0,
                                );
                        }
                        wmidi::MidiMessage::ControlChange(
                            c,
                            wmidi::ControlFunction::MODULATION_WHEEL,
                            v,
                        ) => {
                            let v = u8::try_from(v).unwrap() as f32 / 127.0;
                            inputs
                                .entry(format!("midi_{}_mod_wheel", c.index()))
                                .or_insert_with(Vec::new)
                                .push(v);
                        }
                        wmidi::MidiMessage::ControlChange(
                            c,
                            wmidi::ControlFunction::DAMPER_PEDAL,
                            v,
                        ) => {
                            let v = u8::try_from(v).unwrap() as f32 / 127.0;
                            inputs
                                .entry(format!("midi_{}_pedal", c.index()))
                                .or_insert_with(Vec::new)
                                .push(v);
                        }
                        _ => (),
                    }
                }
            }
            jack::Control::Continue
        };
        let _active_client = client
            .activate_async((), jack::ClosureProcessHandler::new(cback))
            .unwrap();
        loop {
            std::thread::sleep(std::time::Duration::new(10, 0));
        }
    });
}

fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut ring_buffer: Consumer<(f32, f32)>,
) -> cpal::Stream
where
    T: cpal::Sample,
{
    let channels = config.channels as usize;

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                write_data(data, channels, &mut ring_buffer)
            },
            err_fn,
        )
        .unwrap();
    stream.play().unwrap();
    stream
}

fn write_data<T>(output: &mut [T], channels: usize, ring_buffer: &mut Consumer<(f32, f32)>)
where
    T: cpal::Sample,
{
    let mut underran = false;
    for frame in output.chunks_mut(channels) {
        let (value_left, value_right) = ring_buffer.pop().unwrap_or_else(|| {
            underran = true;
            (0.0, 0.0)
        });
        frame[0] = cpal::Sample::from::<f32>(&(value_left * 0.5));
        frame[1] = cpal::Sample::from::<f32>(&(value_right * 0.5));
    }
    if underran {
        println!("buffer underrun");
    }
}
