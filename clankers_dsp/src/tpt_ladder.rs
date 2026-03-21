/// TPT (Topology-Preserving Transform) 4-pole ladder filter
/// Ported from bassYnth/Source/Filter.cpp — Zavalishin design
///
/// 24 dB/oct low-pass.  Resonance 0..1 → feedback k 0..3.98
/// Drive: pre-tanh saturation before the pole chain.

pub struct TptLadder {
    s: [f32; 4],   // integrator states
    sr: f32,
}

impl TptLadder {
    pub fn new(sample_rate: f32) -> Self {
        TptLadder { s: [0.0; 4], sr: sample_rate }
    }

    pub fn reset(&mut self) {
        self.s = [0.0; 4];
    }

    /// Process one sample.
    /// cutoff_hz : filter frequency (20–20 000 Hz)
    /// resonance : 0..1
    /// drive     : ≥1.0 (1.0 = no drive)
    pub fn process(&mut self, x: f32, cutoff_hz: f32, resonance: f32, drive: f32) -> f32 {
        let cutoff_hz = cutoff_hz.clamp(20.0, self.sr * 0.49);
        let resonance  = resonance.clamp(0.0, 1.0);

        let wd = std::f32::consts::PI * cutoff_hz / self.sr;
        let g  = fast_tan(wd);
        let g1 = g / (1.0 + g);          // one-pole normalised coefficient
        let k  = 3.98 * resonance;

        // Cumulative G for 4 poles
        let g2 = g1 * g1;
        let g3 = g2 * g1;
        let g4 = g3 * g1;

        // S: running state contribution from each pole
        let sg = |s: f32| -> f32 { g1 * s };
        let s0 = sg(self.s[0]);
        let s1 = sg(s0 + self.s[1]);
        let s2 = sg(s1 + self.s[2]);
        let s3 = sg(s2 + self.s[3]);

        // Feedback sum (implicit resolve)
        let s_total = s0 + g1 * (s1 + g1 * (s2 + g1 * s3));
        // actually use the standard resolved form:
        let s_fb = self.s[0] * g3 + self.s[1] * g2 + self.s[2] * g1 + self.s[3];
        let _ = (s0, s1, s2, s3, s_total); // suppress unused warnings

        // Pre-drive tanh
        let xd = (x * drive).tanh();

        // Resolved input (suppress resonance feedback from all 4 states)
        let denom = 1.0 + k * g4;
        let y0 = (xd - k * g4 * s_fb / denom.max(1e-9)) / denom.max(1e-9);

        // Cascade 4 one-pole stages
        let tick = |input: f32, state: &mut f32| -> f32 {
            let v = g1 * (input - *state);
            let out = v + *state;
            *state = out + v;
            sanitize(*state);
            out
        };

        // We need mutable access sequentially
        let u1 = tick(y0,         &mut self.s[0]);
        let u2 = tick(u1,         &mut self.s[1]);
        let u3 = tick(u2,         &mut self.s[2]);
        let u4 = tick(u3,         &mut self.s[3]);

        sanitize(u4);
        u4
    }
}

/// Polynomial approximation of tan — matches bassYnth fastTan()
#[inline]
fn fast_tan(x: f32) -> f32 {
    let x2 = x * x;
    let num = x * (1.0 + x2 / 3.0 + 2.0 * x2 * x2 / 15.0);
    let den = 1.0 - x2 / 3.0;
    num / den.abs().max(1e-9)
}

#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}
