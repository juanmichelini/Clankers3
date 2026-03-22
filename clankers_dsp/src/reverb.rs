/// Simple Schroeder reverb
/// HybridSynth uses JUCE's built-in reverb which is a standard Freeverb variant.
/// This is a clean Freeverb-style implementation: 8 parallel comb filters + 4 allpass.
///
/// Parameters map to HybridSynth CC88 (room size) and CC91 (wet mix).

const NUM_COMBS:   usize = 8;
const NUM_ALLPASS: usize = 4;

// Comb filter delay lengths (samples at 44100 Hz) — standard Freeverb values
const COMB_SIZES:    [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_SIZES: [usize; 4] = [556, 441, 341, 225];

pub struct Reverb {
    combs:    [CombFilter; NUM_COMBS],
    allpass:  [AllpassFilter; NUM_ALLPASS],
    sr_scale: f32,
}

impl Reverb {
    pub fn new(sample_rate: f32) -> Self {
        let scale = sample_rate / 44100.0;
        Reverb {
            combs:   std::array::from_fn(|i| CombFilter::new((COMB_SIZES[i] as f32 * scale) as usize + 1)),
            allpass: std::array::from_fn(|i| AllpassFilter::new((ALLPASS_SIZES[i] as f32 * scale) as usize + 1)),
            sr_scale: scale,
        }
    }

    /// Process one mono sample → returns (wet_l, wet_r).
    /// room_size : 0..1 → feedback 0.5..0.85
    /// damp      : 0..1 → high-frequency damping
    pub fn process_mono(&mut self, x: f32, room_size: f32, damp: f32) -> f32 {
        let feedback = 0.5 + room_size * 0.35;
        let damp     = damp.clamp(0.0, 1.0);

        let input = x * 0.15; // scale to prevent overload

        // 8 parallel comb filters
        let mut out = 0.0f32;
        for c in self.combs.iter_mut() {
            out += c.process(input, feedback, damp);
        }

        // 4 series allpass filters
        for a in self.allpass.iter_mut() {
            out = a.process(out);
        }

        out
    }
}

// ── Comb filter with damping ──────────────────────────────────────────────────

struct CombFilter {
    buf:      Vec<f32>,
    pos:      usize,
    filtered: f32,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        CombFilter { buf: vec![0.0; size.max(1)], pos: 0, filtered: 0.0 }
    }

    fn process(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let output = self.buf[self.pos];
        // Low-pass damp on feedback signal
        self.filtered = output * (1.0 - damp) + self.filtered * damp;
        self.buf[self.pos] = x + self.filtered * feedback;
        self.pos = (self.pos + 1) % self.buf.len();
        output
    }
}

// ── Allpass filter ────────────────────────────────────────────────────────────

struct AllpassFilter {
    buf: Vec<f32>,
    pos: usize,
}

impl AllpassFilter {
    fn new(size: usize) -> Self {
        AllpassFilter { buf: vec![0.0; size.max(1)], pos: 0 }
    }

    fn process(&mut self, x: f32) -> f32 {
        let delayed = self.buf[self.pos];
        let output  = -x + delayed;
        self.buf[self.pos] = x + delayed * 0.5;
        self.pos = (self.pos + 1) % self.buf.len();
        output
    }
}
