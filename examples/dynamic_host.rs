#![feature(exact_size_is_empty)]
use std::{
    sync::Arc,
    convert::TryFrom
};
use ringbuf::{RingBuffer, Producer, Consumer};
use portmidi as pm;
use gilrs::{Gilrs, EventType, Event};

use ::instruments::{instruments::*, simd_graph::*, type_list::Value, InstrumentSynth, dynamic_graph::{DynamicGraph, BoxedDynamicNode}};
use packed_simd_2::f32x8;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const DEFAULT_SYNTH: &'static str = "# Sequence
rhythm_gate=sine(16.0)
rhythm_rnd=turing_machine(rhythm_gate,0.9,8)

rnd=turing_machine(rhythm_rnd,0.9,8)

# Pitch mapping
pitch=rescale(440.0,880.0,rnd)

# Oscillator
top_tone=saw(pitch)
sub_pitch=mul(0.25,pitch)
sub_tone=sine(sub_pitch)
tone=add(top_tone,sub_tone)
envelope=ad(0.01,0.02,rhythm_rnd)
enveloped_tone=mul(envelope,tone)

# Final
filter=lpf(440,0.7,enveloped_tone)

(output,0,filter)
(output,1,filter)
";


fn main() {
    let synth_path = std::env::args().nth(1);

    let mut builder = InstrumentSynth::builder();

    let mut graph = if let Some(synth_data) = synth_path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
        DynamicGraph::parse(&synth_data).unwrap()
    } else {
        DynamicGraph::parse(DEFAULT_SYNTH).unwrap()
    };

    let inputs = Arc::clone(&graph.external_inputs);
    std::thread::spawn(move || {
        let mut gilrs = Gilrs::new().unwrap();

        let context = pm::PortMidi::new().unwrap();
        use std::time::Duration;
        let timeout = Duration::from_millis(10);

        let info = context.device(1).unwrap();
        let in_ports = context
             .devices()
             .unwrap()
             .into_iter()
             .filter_map(|dev| context.input_port(dev, 1024).ok())
             .collect::<Vec<_>>();
         loop {
            {
                let mut inputs = inputs.lock().unwrap();
                while let Some(Event { id, event, time }) = gilrs.next_event() {
                    match event {
                        EventType::AxisChanged(a, v, ..) => {
                            inputs.insert(format!("pad_{:?}", a), v);
                        }
                        EventType::ButtonPressed(b, ..) => {
                            inputs.insert(format!("pad_{:?}", b), 1.0);
                        }
                        EventType::ButtonReleased(b, ..) => {
                            inputs.insert(format!("pad_{:?}", b), 0.0);
                        }
                        _ => ()
                    }
                }
                 for port in &in_ports {
                     if let Ok(Some(events)) = port.read_n(1024) {
                         for event in events {
                            let data = [event.message.status, event.message.data1, event.message.data2, event.message.data3];
                            let message = wmidi::MidiMessage::try_from(&data[..]).unwrap();
                            match message {
                                wmidi::MidiMessage::NoteOn(c,n,v) => {
                                    inputs.insert(format!("midi_{}_freq", c.index()), n.to_freq_f32());
                                    inputs.insert(format!("midi_{}_velocity", c.index()), u8::try_from(v).unwrap() as f32 / 127.0);
                                }
                                wmidi::MidiMessage::NoteOff(c,n,v) => {
                                    inputs.insert(format!("midi_{}_freq", c.index()), n.to_freq_f32());
                                    inputs.insert(format!("midi_{}_velocity", c.index()), 0.0);
                                }
                                wmidi::MidiMessage::PitchBendChange(c,b) => {
                                    inputs.insert(format!("midi_{}_pitch_bend", c.index()), (u16::try_from(b).unwrap() as f32 / 2.0f32.powi(14) - 0.5) * 2.0);
                                }
                                wmidi::MidiMessage::ControlChange(c,wmidi::ControlFunction::MODULATION_WHEEL,v) => {
                                    let v = u8::try_from(v).unwrap() as f32 / 127.0;
                                    inputs.insert(format!("midi_{}_mod_wheel", c.index()), v);
                                } 
                                wmidi::MidiMessage::ControlChange(c,wmidi::ControlFunction::DAMPER_PEDAL,v) =>{
                                    let v = u8::try_from(v).unwrap() as f32 / 127.0;
                                    inputs.insert(format!("midi_{}_pedal", c.index()), v);
                                }
                                _ => ()
                            }
                         }
                     }
                 }
            }
              std::thread::sleep(timeout);
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

    let rb = RingBuffer::<(f32,f32)>::new(4048);
    let (mut prod, mut cons) = rb.split();

    std::thread::spawn(move || {
        let mut left = vec![0.0; 32];
        let mut right = vec![0.0; 32];
        loop {
            synth.process(&mut left, &mut right);
            let mut to_push = left.iter().zip(&right).map(|(l,r)| (*l,*r));
            loop {
                prod.push_iter(&mut to_push);
                if to_push.is_empty() {
                    break
                } else {
                    std::thread::sleep(std::time::Duration::from_secs_f32(10.0/44000.0));
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
            let mut ps:Vec<_> = watch_list.lock().unwrap().iter().cloned().collect();
            ps.push(synth_path.to_string());
            let mut needs_reload = false;
            for p in ps {
                if let Ok(metadata) = std::fs::metadata(&p) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > last_change {
                            needs_reload = true;
                            last_change = modified;
                            break
                        }
                    }
                }
            }
            if needs_reload {
                reload_data.lock().unwrap().replace(std::fs::read_to_string(synth_path).unwrap());
            }
        }
    }
}

fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut ring_buffer: Consumer<(f32,f32)>,
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

fn write_data<T>(
    output: &mut [T],
    channels: usize,
    ring_buffer: &mut Consumer<(f32,f32)>,
) where
    T: cpal::Sample,
{
    let mut underran = false;
    for (i, frame) in output.chunks_mut(channels).enumerate() {
        let (value_left, value_right) = ring_buffer.pop().unwrap_or_else(|| { underran=true; (0.0, 0.0) });
        frame[0] = cpal::Sample::from::<f32>(&(value_left * 0.5));
        frame[1] = cpal::Sample::from::<f32>(&(value_right * 0.5));
    }
    if underran {
        println!("buffer underrun");
    }
}
