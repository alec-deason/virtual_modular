use ::instruments::{instruments::*, simd_graph::*, type_list::{Value, NoValue}, InstrumentSynth};
use packed_simd_2::f32x8;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() {
    let mut builder = InstrumentSynth::builder();
    let mut mixer = Mixer::default();

    let pulse = Pipe(
        Pipe(Pipe(Constant(11.0), LfoSine::new(0.0)), Impulse::default()),
        curry(Constant(f32x8::splat(8.0)), Mul::default()),
    );

    let pulses = Pipe(
        Pipe(
            Pipe(Constant(0.1), LfoSine::positive(0.1)),
            Rescale(2.0, 8.0),
        ),
        Max,
    );
    let len = Pipe(
        Pipe(
            Pipe(Constant(0.3), LfoSine::positive(0.2)),
            Rescale(9.0, 11.0),
        ),
        Max,
    );

    let pulse = Pipe(
        pulse,
        curry(Stack::new(pulses, len), EuclidianPulse::default()),
    );

    let pitch1 = Pipe(
        Pipe(
            Pipe(Constant(0.1 * 1.0), LfoSine::positive(0.3)),
            Rescale(60.0, 1601.0),
        ),
        curry(pulse.clone(), SampleAndHold::default()),
    );
    let pitch2 = Pipe(
        Pipe(
            Pipe(Constant(0.1 * 3.0), LfoSine::positive(0.4)),
            Rescale(600.0, 2060.0),
        ),
        curry(pulse.clone(), SampleAndHold::default()),
    );

    let q1 = Pipe(
        Pipe(
            Pipe(Constant(0.1 * 11.0), LfoSine::positive(1.5)),
            Rescale(0.1, 40.0),
        ),
        curry(pulse.clone(), SampleAndHold::default()),
    );
    let q2 = Pipe(
        Pipe(
            Pipe(Constant(0.1 * 13.0), LfoSine::positive(1.51)),
            Rescale(3.0, 40.0),
        ),
        curry(pulse.clone(), SampleAndHold::default()),
    );

    let (b_in, b) = Bridge::<Value<(f32x8,)>>::new((f32x8::splat(0.0),));
    let feedback_gain = Constant(f32x8::splat(0.0005));
    let filter_a = curry(Branch::new(pitch1, q1), Biquad::bandpass());
    let filter_b = curry(Branch::new(pitch2, q2), Biquad::bandpass());
    let filter = Pipe(Branch::new(filter_a, filter_b), Add::default());
    let plink = Pipe(
        Pipe(
            pulse.clone(),
            curry(
                Pipe(
                    Pipe(b, Strip::default()),
                    curry(feedback_gain, Mul::default()),
                ),
                Add::default(),
            ),
        ),
        filter,
    );
    let ping = Pipe(
        Pipe(plink, b_in),
        curry(Constant(f32x8::splat(0.7)), Mul::<f32x8, f32x8>::default()),
    );
    let ping = Pipe(ping, BFunc(|v: f32x8| v.tanh()));

    let ping = Pipe(ping, Split);
    let ping = Pipe(ping, curry2(Constant(f32x8::splat(1.0)), Stutter::rand_pan(50, 0.15)));

    mixer.add_track(ping);

    let mut synth = builder.build_with_synth(mixer);

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
