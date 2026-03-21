/// Wavefolding — Buchla 259 timbre circuit
/// Ported from Buchlasystem/Source/DSP/Waveshaper.cpp
///
/// Amount 0..1:
///   0.0–0.3  → subtle Chebyshev enrichment
///   0.3–0.5  → sin() primary fold
///   0.5–1.0  → double fold (sin of sin) for aggressive harmonics
/// Soft-clipped with tanh() at output.
/// No oversampling (acceptable for arp role at typical pitches).

pub struct Wavefolder;

impl Wavefolder {
    #[inline]
    pub fn process(x: f32, amount: f32) -> f32 {
        let amount = amount.clamp(0.0, 1.0);
        let gain   = 1.0 + amount * 7.0;   // 1× .. 8× input gain
        let driven = x * gain;

        let folded = if amount < 0.3 {
            // Subtle: blend Chebyshev T3 (4x³ - 3x) for harmonic enrichment
            let cheby = 4.0 * driven * driven * driven - 3.0 * driven;
            let t = amount / 0.3;
            driven * (1.0 - t) + cheby * t
        } else if amount < 0.5 {
            // Primary fold: sin(x)
            driven.sin()
        } else {
            // Double fold
            let extra_fold = (amount - 0.5) * 2.0; // 0..1
            let first  = driven.sin();
            let second = (first * (1.0 + extra_fold * 3.0)).sin();
            first * (1.0 - extra_fold * 0.5) + second * extra_fold * 0.5
        };

        folded.tanh()
    }
}
