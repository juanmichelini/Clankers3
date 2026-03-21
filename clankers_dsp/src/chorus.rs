/// Stereo chorus effect
/// Ported from HybridSynth/Source/ChorusEffect.h
///
/// LFO-modulated delay line.
/// Centre delay: 7 ms. Modulation depth: 0–15 ms.
/// Stereo: L and R use opposite LFO phases for width.

const MAX_DELAY_SAMPLES: usize = 4096; // ~93 ms at 44100 Hz

pub struct Chorus {
    buffer_l: Vec<f32>,
    buffer_r: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    sr: f32,
}

impl Chorus {
    pub fn new(sample_rate: f32) -> Self {
        Chorus {
            buffer_l: vec![0.0; MAX_DELAY_SAMPLES],
            buffer_r: vec![0.0; MAX_DELAY_SAMPLES],
            write_pos: 0,
            lfo_phase: 0.0,
            sr: sample_rate,
        }
    }

    /// Process one stereo sample in place.
    /// rate  : LFO Hz (0.1–8 Hz)
    /// depth : 0..1 → 0–15 ms modulation
    /// mix   : 0..1 wet/dry
    pub fn process(&mut self, left: f32, right: f32, rate: f32, depth: f32, mix: f32) -> (f32, f32) {
        let centre_samples = (0.007 * self.sr) as f32;   // 7 ms
        let max_mod        = 0.015 * self.sr;              // 15 ms max mod

        let lfo = (self.lfo_phase * std::f32::consts::TAU).sin();
        self.lfo_phase += rate / self.sr;
        if self.lfo_phase >= 1.0 { self.lfo_phase -= 1.0; }

        let delay_l = centre_samples + lfo * depth * max_mod;
        let delay_r = centre_samples - lfo * depth * max_mod; // opposite phase → stereo width

        let wet_l = self.read_interp(&self.buffer_l, self.write_pos, delay_l);
        let wet_r = self.read_interp(&self.buffer_r, self.write_pos, delay_r);

        self.buffer_l[self.write_pos] = left;
        self.buffer_r[self.write_pos] = right;
        self.write_pos = (self.write_pos + 1) % MAX_DELAY_SAMPLES;

        let out_l = left  * (1.0 - mix) + wet_l * mix;
        let out_r = right * (1.0 - mix) + wet_r * mix;
        (out_l, out_r)
    }

    fn read_interp(&self, buf: &[f32], write: usize, delay: f32) -> f32 {
        let delay = delay.clamp(1.0, (MAX_DELAY_SAMPLES - 2) as f32);
        let n = MAX_DELAY_SAMPLES;
        let read_f = (write + n) as f32 - delay;
        let idx0   = read_f as usize % n;
        let idx1   = (idx0 + 1) % n;
        let frac   = read_f.fract();
        buf[idx0] * (1.0 - frac) + buf[idx1] * frac
    }
}
