/// Moog ladder filter — 4-pole lowpass
/// Ported from HybridSynth/Source/Filter.cpp
///
/// Simplified Moog: 4 cascaded 1-pole stages with tanh feedback.
/// g = 1 - exp(-2π·fc/sr)  (stable exponential mapping)
/// Resonance 0..1 → feedback k 0..3.8 (clamped below self-oscillation)

pub struct MoogLadder {
    y: [f64; 5],   // y[0..3] = stage states, y[4] = feedback delay
    sr: f32,
}

impl MoogLadder {
    pub fn new(sample_rate: f32) -> Self {
        MoogLadder { y: [0.0; 5], sr: sample_rate }
    }

    pub fn reset(&mut self) { self.y = [0.0; 5]; }

    /// Process one sample.
    /// cutoff_hz : 20–20 000 Hz
    /// resonance : 0..1
    /// drive     : ≥1.0
    pub fn process(&mut self, x: f32, cutoff_hz: f32, resonance: f32, drive: f32) -> f32 {
        let cutoff = cutoff_hz.clamp(20.0, self.sr * 0.499) as f64;
        let res    = (resonance as f64).clamp(0.0, 0.95);
        let drive  = drive as f64;

        let wc = std::f64::consts::TAU * cutoff / self.sr as f64;
        let g  = 1.0 - (-wc).exp();
        let k  = res * 3.8; // 0..3.8 feedback gain

        // Feedback with tanh soft-clip (matches Filter.cpp)
        let x0 = (x as f64 * drive - k * self.y[4]).tanh();

        // 4 cascaded 1-pole passes
        self.y[0] += g * (x0       - self.y[0]);
        self.y[1] += g * (self.y[0] - self.y[1]);
        self.y[2] += g * (self.y[1] - self.y[2]);
        self.y[3] += g * (self.y[2] - self.y[3]);
        self.y[4]  = self.y[3];

        // Sanitize
        for s in self.y.iter_mut() {
            if !s.is_finite() { *s = 0.0; }
        }

        self.y[3] as f32
    }
}
