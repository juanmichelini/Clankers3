/// Low Pass Gate — Buchla 292
/// Ported from Buchlasystem/Source/DSP/LowpassGate.cpp
///
/// Cytomic trapezoidal SVF (2-pole state-variable filter) in combo mode:
/// vactrol CV simultaneously modulates cutoff frequency and amplitude.
///
/// Mode: Combo (LPF + VCA coupled by same vactrol CV) — matches hardware default.

pub struct Lpg {
    // SVF state
    ic1eq: f32,
    ic2eq: f32,
    sr: f32,
}

impl Lpg {
    pub fn new(sample_rate: f32) -> Self {
        Lpg { ic1eq: 0.0, ic2eq: 0.0, sr: sample_rate }
    }

    pub fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    /// Process one sample.
    /// cv       : vactrol output 0..1 (controls cutoff + amplitude together)
    /// base_cutoff : normalised base cutoff 0..1 (from CC74)
    /// resonance   : 0..1
    pub fn process(&mut self, x: f32, cv: f32, base_cutoff: f32, resonance: f32) -> f32 {
        // Cutoff: CV opens the filter up to base_cutoff ceiling
        // Low base_cutoff = darker sound even when CV is high
        let cutoff_norm = (cv * base_cutoff).clamp(0.0, 0.999);
        // Cap at sr*0.45 to prevent tan() blow-up near Nyquist
        let cutoff_hz   = (20.0 * 20000.0_f32.powf(cutoff_norm)).min(self.sr * 0.45);

        // Damping — light resonance, matches Buchla 292 hardware character
        let resonance  = resonance.clamp(0.0, 0.85);
        let k = 2.0 - 2.0 * resonance;

        let g  = (std::f32::consts::PI * cutoff_hz / self.sr).tan();
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let v3 = x  - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;

        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        sanitize(&mut self.ic1eq);
        sanitize(&mut self.ic2eq);

        // LP output × VCA (cv gates amplitude in combo mode)
        v2 * cv
    }
}

#[inline]
fn sanitize(x: &mut f32) {
    if !x.is_finite() { *x = 0.0; }
}
