#![feature(exact_size_is_empty)]
use generic_array::typenum::*;
use gilrs::{Event, EventType, Gilrs};
use portmidi as pm;
use ringbuf::{Consumer, Producer, RingBuffer};
use std::{convert::TryFrom, sync::Arc};

use ::virtual_modular::{
    dynamic_graph::{BoxedDynamicNode, DynamicGraph, DynamicGraphBuilder},
    InstrumentSynth,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() {
    let synth_path = std::env::args().nth(1);

    let mut builder = InstrumentSynth::builder();

    let mut graph = if let Some(synth_data) = synth_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
    {
        DynamicGraphBuilder::default().parse(&synth_data).unwrap()
    } else {
        panic!()
    };

    let inputs = Arc::clone(&graph.external_inputs);
    std::thread::spawn(move || {
        let mut gilrs = Gilrs::new().unwrap();

        let context = pm::PortMidi::new().unwrap();
        use std::time::Duration;
        let timeout = Duration::from_millis(10);

        let mut voices = std::collections::HashMap::new();
        for c in 0..10 {
            voices.insert(c, (0..4).map(|i| (i, None)).collect::<Vec<_>>());
        }
        let mut current_voice = 0;

        //let info = context.device(1).unwrap();
        /*
        let in_ports = context
             .devices()
             .unwrap()
             .into_iter()
             .filter_map(|dev| context.input_port(dev, 1024).ok())
             .collect::<Vec<_>>();
         */
        //println!("{:?}", context.devices());
        //let mut in_ports = vec![context.default_input_port(1024).unwrap()];
        let (client, _status) =
            jack::Client::new("rust_jack_show_midi", jack::ClientOptions::NO_START_SERVER).unwrap();
        let shower = client
            .register_port("rust_midi_shower", jack::MidiIn::default())
            .unwrap();
        let cback = move |_: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
            {
                let mut inputs = inputs.lock().unwrap();
                //while let Some(Event { id, event, time }) = gilrs.next_event() {
                //    match event {
                //        EventType::AxisChanged(a, v, ..) => {
                //            inputs.entry(format!("pad_{:?}", a)).or_insert_with(Vec::new).push(v);
                //        }
                //        EventType::ButtonPressed(b, ..) => {
                //            inputs.entry(format!("pad_{:?}", b)).or_insert_with(Vec::new).push(1.0);
                //        }
                //        EventType::ButtonReleased(b, ..) => {
                //            inputs.entry(format!("pad_{:?}", b)).or_insert_with(Vec::new).push(0.0);
                //        }
                //        _ => ()
                //    }
                //}
                //for port in &mut in_ports {
                ///if let Ok(_) = port.poll() {
                //port.poll();
                // while let Ok(Some(event)) = port.read() {
                for e in shower.iter(ps) {
                    //let data = [event.message.status, event.message.data1, event.message.data2, event.message.data3];
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
                                    let mut smash = false;
                                    let (aval, voice) = if let Some((aval, voice)) = aval {
                                        (aval, voice)
                                    } else {
                                        smash = true;
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
                                        for (i, (j, f)) in voices.iter_mut().enumerate() {
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
                        wmidi::MidiMessage::NoteOff(c, n, v) => {
                            if let Some(voices) = voices.get_mut(&c.index()) {
                                for (i, (j, f)) in voices.iter_mut().enumerate() {
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
        let active_client = client
            .activate_async((), jack::ClosureProcessHandler::new(cback))
            .unwrap();
        loop {
            std::thread::sleep(std::time::Duration::new(10, 0));
        }
    });

    let reload_data = Arc::clone(&graph.reload_data);
    let watch_list = Arc::clone(&graph.watch_list);

    let mut synth = builder.build_with_synth(graph);

    let cpal_host = cpal::default_host();

    let device = cpal_host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();

    let sample_rate = config.sample_rate().0 as f32;
    synth.set_sample_rate(sample_rate);

    let rb = RingBuffer::<(f32, f32)>::new(4048);
    let (mut prod, mut cons) = rb.split();

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

    let cpal_stream = match config.sample_format() {
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
    for (i, frame) in output.chunks_mut(channels).enumerate() {
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
