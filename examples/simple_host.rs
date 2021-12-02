use hound;
use virtual_modular::InstrumentSynth;
use virtual_modular_core_nodes::*;

fn main() {
    let output_path = std::env::args()
        .nth(1)
        .expect("Must supply a path for the output WAV file");

    // Make a sine oscillator running at 220 hz
    let osc = Pipe(Branch(Constant(220.0), Constant(0.0)), Sine::default());

    // Duplicate it across two channels to make a stereo signal
    let sound = Pipe(osc, Stereo);

    // Construct the synthesizer from that stereo generator
    let builder = InstrumentSynth::builder();
    let mut synth = builder.build_with_synth(sound);

    synth.set_sample_rate(44100.0);

    // Render one second of sound to buffers
    let mut left_buffer = vec![0.0; 44100];
    let mut right_buffer = vec![0.0; 44100];
    synth.process(&mut left_buffer, &mut right_buffer);

    // Write out a WAV file. The system is designed to be real time but
    // for simplicity I'm using hound in this example rather than cpal
    // or jack. For a complete realtime example look in the dynamic_environment directory
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec).unwrap();
    let amplitude = i16::MAX as f32;
    for (l, r) in left_buffer.iter().zip(&right_buffer) {
        writer
            .write_sample((l.min(1.0) * amplitude) as i16)
            .unwrap();
        writer
            .write_sample((r.min(1.0) * amplitude) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}
