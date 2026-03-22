/// Vactrol model — Buchla 292 LPG gate CV simulation
/// Ported from Buchlasystem/Source/DSP/VactrolModel.cpp
///
/// Asymmetric one-pole smoothing:
///   Rising  → fast (attack_coeff, ~1–10 ms)
///   Falling → slow + nonlinear droop (decay_coeff with quadratic slowdown)
/// Output is passed through smoothstep for the characteristic nonlinear response.

pub struct Vactrol {
    state:       f32,
    droop_state: f32,
    attack_coeff:  f32,
    decay_coeff:   f32,
    sr: f32,
}

impl Vactrol {
    pub fn new(sample_rate: f32) -> Self {
        let mut v = Vactrol { state: 0.0, droop_state: 0.0, attack_coeff: 0.0, decay_coeff: 0.0, sr: sample_rate };
        v.set_times(0.002, 0.12); // default: 2 ms attack, 120 ms decay
        v
    }

    /// attack_s / decay_s in seconds
    pub fn set_times(&mut self, attack_s: f32, decay_s: f32) {
        self.attack_coeff = 1.0 - (-1.0 / (self.sr * attack_s.max(0.0001))).exp();
        self.decay_coeff  = 1.0 - (-1.0 / (self.sr * decay_s.max(0.001))).exp();
    }

    /// Process one sample. input: 0.0 (gate off) or 1.0 (gate on).
    /// Returns 0..1 vactrol CV with nonlinear droop on decay.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        if input > self.state {
            // Rising — fast
            self.state      += self.attack_coeff * (input - self.state);
            self.droop_state = 0.0; // reset droop on re-trigger
        } else {
            // Falling — slow with droop
            let droop_factor = (1.0 + self.droop_state * self.droop_state) * 0.5;
            let coeff = (self.decay_coeff * droop_factor).min(0.5);
            self.state       += coeff * (input - self.state);
            self.droop_state += self.decay_coeff * 0.1; // accumulate droop over time
            self.droop_state  = self.droop_state.min(2.0);
        }

        self.state = self.state.clamp(0.0, 1.0);
        smoothstep(self.state)
    }

    /// Instantly set vactrol to full open — bypass the one-pole attack.
    /// Use this on trigger so the LPG fires immediately instead of
    /// ramping from 0 over many samples.
    pub fn fire(&mut self) {
        self.state       = 1.0;
        self.droop_state = 0.0;
    }

    /// Fire at a specific level (0..1) — allows cutoff to shape the initial tone.
    pub fn fire_at(&mut self, level: f32) {
        self.state       = level.clamp(0.0, 1.0);
        self.droop_state = 0.0;
    }

    pub fn reset(&mut self) {
        self.state       = 0.0;
        self.droop_state = 0.0;
    }

    pub fn is_idle(&self) -> bool {
        self.state < 1e-4
    }
}

/// Nonlinear output response: state² × (3 − 2×state)
#[inline]
fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
