extern crate vst;

use std::env;
use std::path::Path;
use std::process;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use vst::{
    api::Event,
    host::{Host, PluginLoader, PluginInstance, HostBuffer},
    plugin::Plugin,
};

#[allow(dead_code)]
struct SampleHost;

impl Host for SampleHost {
    fn automate(&self, index: i32, value: f32) {
        println!("Parameter {} had its value changed to {}", index, value);
    }
}

fn main() {

    let cpal_host = cpal::default_host();
    let device = cpal_host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();
    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
    }
}

pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig)
where
        T: cpal::Sample,
{
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("usage: simple_host path/to/vst");
        process::exit(1);
    }

    let path = Path::new(&args[1]);

    // Create the host
    let host = Arc::new(Mutex::new(SampleHost));

    println!("Loading {}...", path.to_str().unwrap());

    // Load the plugin
    let mut loader =
        PluginLoader::load(path, Arc::clone(&host)).unwrap_or_else(|e| panic!("Failed to load plugin: {}", e));

    // Create an instance of the plugin
    let mut instance = loader.instance().unwrap();

    // Initialize the instance
    instance.init();
    instance.resume();
    let sample_rate = config.sample_rate.0 as f32;
    instance.set_sample_rate(sample_rate);

    let mut event = vst::api::MidiEvent {
        event_type: vst::api::EventType::Midi,
        byte_size: std::mem::size_of::<vst::api::MidiEvent>() as i32,
        delta_frames: 0,
        flags: 0,
        note_length: 0,
        note_offset: 0,
        detune: 0,
        note_off_velocity: 0,
        midi_data: [128, 69, 0],
        _midi_reserved: 0,
        _reserved1: 0,
        _reserved2: 0,
    };
    let p_ref = &mut event;
    let e: *mut Event = unsafe { std::mem::transmute(p_ref) };
    let events = vst::api::Events {
        num_events: 1,
        _reserved: 0,
        events: [
            e,
            e
        ]
    };
    instance.start_process();
    instance.process_events(&events);
    instance.stop_process();

    let channels = config.channels as usize;

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut instance)
        },
        err_fn,
    ).unwrap();
    stream.play().unwrap();

    std::thread::sleep(std::time::Duration::from_millis(1000));
}

fn write_data<T>(output: &mut [T], channels: usize, instance: &mut impl Plugin)
where
    T: cpal::Sample,
{

    instance.start_process();

    let mut host_buffer: HostBuffer<f32> = HostBuffer::new(2, 2);

    let mut idx = 0;
    for frame in output.chunks_mut(channels) {
        let inputs = vec![vec![0.0; frame.len()]; 2];
        let mut outputs = vec![vec![0.0; frame.len()]; 2];
        let mut audio_buffer = host_buffer.bind(&inputs, &mut outputs);

        instance.process(&mut audio_buffer);
        for sample in frame.iter_mut() {
            let value: T = cpal::Sample::from::<f32>(&outputs[0][idx]);
            idx = (idx + 1) % outputs[0].len();
            *sample = value;
        }
    }
    instance.stop_process();
}
