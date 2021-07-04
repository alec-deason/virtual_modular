use std::sync::Arc;
use ::instruments::{instruments::*, simd_graph::*, type_list::Value, InstrumentSynth, dynamic_graph::{DynamicGraph, BoxedDynamicNode}};
use packed_simd_2::f32x8;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const DEFAULT_SYNTH: &'static str = "# Sequence
rhythm_gate=sine(16.0)
rhythm_rnd=turing_machine(0.0,0.9,8)
(0,rhythm_gate,0,rhythm_rnd)

rnd=turing_machine(0,0.9,8)
(0,rhythm_rnd,0,rnd)

# Pitch mapping
pitch=rescale(440.0,880.0)
(0,rnd,2,pitch)

# Oscillator
top_tone=saw
(0,pitch,0,top_tone)
sub_pitch=mul(0.25)
(0,pitch,1,sub_pitch)
sub_tone=sine
(0,sub_pitch,0,sub_tone)
tone=add
(0,top_tone,0,tone)
(0,sub_tone,1,tone)
envelope=ad
(0,rhythm_rnd,0,envelope)
enveloped_tone=mul
(0,envelope,0,enveloped_tone)
(0,tone,1,enveloped_tone)


# Final
filter=lpf(440,0.7)
(0,enveloped_tone,2,filter)

(0,filter,0,output)
(0,filter,1,output)
";


fn main() {
    let synth_path = std::env::args().nth(1);

    let mut builder = InstrumentSynth::builder();

    let mut graph = if let Some(synth_data) = synth_path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
        DynamicGraph::parse(&synth_data).unwrap()
    } else {
        DynamicGraph::parse(DEFAULT_SYNTH).unwrap()
    };

    let reload_data = Arc::clone(&graph.reload_data);
    let watch_list = Arc::clone(&graph.watch_list);

    let mut synth = builder.build_with_synth(graph);

    let cpal_host = cpal::default_host();

    let device = cpal_host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();

    let sample_rate = config.sample_rate().0 as f32;
    synth.set_sample_rate(sample_rate);
    let cpal_stream = match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), synth),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), synth),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), synth),
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
    mut synth: InstrumentSynth,
) -> cpal::Stream
where
    T: cpal::Sample,
{
    let channels = config.channels as usize;

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let mut outputs = vec![vec![0.0; 128]; 2];


    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                write_data(data, channels, &mut synth, &mut outputs)
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
    synth: &mut InstrumentSynth,
    outputs: &mut Vec<Vec<f32>>,
) where
    T: cpal::Sample,
{
    outputs[0].resize(output.len() / 2, 0.0);
    outputs[0].fill(0.0);
    outputs[1].resize(output.len() / 2, 0.0);
    outputs[1].fill(0.0);

    let (left, tail) = outputs.split_at_mut(1);
    synth.process(&mut left[0], &mut tail[0]);

    for (i, frame) in output.chunks_mut(channels).enumerate() {
        let value_left = outputs[0][i];
        let value_right = outputs[1][i];

        frame[0] = cpal::Sample::from::<f32>(&(value_left * 0.5));
        frame[1] = cpal::Sample::from::<f32>(&(value_right * 0.5));
    }
}
