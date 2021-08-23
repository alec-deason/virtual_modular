include!(concat!(env!("OUT_DIR"), "/synth.rs"));

use ::instruments::{simd_graph::*, InstrumentSynth};
use packed_simd_2::f32x8;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() {
    let mut builder = InstrumentSynth::builder();


    let dsp_graph = build_synth();

    let mut synth = builder.build_with_synth(dsp_graph);

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

    loop {
        std::thread::sleep(std::time::Duration::from_millis(10));
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
