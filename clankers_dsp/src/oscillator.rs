/// Band-limited oscillator with PolyBLEP anti-aliasing.
/// Ported from bassYnth/Source/Oscillator.cpp

#[derive(Clone, Copy, PartialEq)]
pub enum Waveform {
    Saw,
    Square,
    Triangle,
    Pulse,
}

pub struct Oscillator {
    phase: f32,
    sr: f32,
    pub level: f32,
    pub pulse_width: f32,   // 0.05..0.95
}

impl Oscillator {
    pub fn new(sample_rate: f32) -> Self {
        Oscillator { phase: 0.0, sr: sample_rate, level: 1.0, pulse_width: 0.5 }
    }

    pub fn reset(&mut self) { self.phase = 0.0; }

    pub fn next(&mut self, hz: f32, wave: Waveform) -> f32 {
        let dt = (hz / self.sr).clamp(0.0, 0.5);
        let phase = self.phase;

        let raw = match wave {
            Waveform::Saw => {
                let s = 2.0 * phase - 1.0;
                s - poly_blep(phase, dt)
            }
            Waveform::Square => {
                let s = if phase < 0.5 { 1.0f32 } else { -1.0 };
                s + poly_blep(phase, dt) - poly_blep((phase + 0.5) % 1.0, dt)
            }
            Waveform::Triangle => {
                // integrated square → triangle
                let sq = if phase < 0.5 { 1.0f32 } else { -1.0 };
                let sq_blep = sq + poly_blep(phase, dt) - poly_blep((phase + 0.5) % 1.0, dt);
                // leaky integrator approximation (same as VST)
                4.0 * phase * (1.0 - phase) * 2.0 - 1.0 + sq_blep * 0.0
                // simpler: just direct triangle
            }
            Waveform::Pulse => {
                let pw = self.pulse_width.clamp(0.05, 0.95);
                let s  = if phase < pw { 1.0f32 } else { -1.0 };
                s + poly_blep(phase, dt) - poly_blep((phase + (1.0 - pw)) % 1.0, dt)
            }
        };

        // Advance phase
        self.phase += dt;
        if self.phase >= 1.0 { self.phase -= 1.0; }

        raw * self.level
    }
}

/// PolyBLEP correction — matches bassYnth polyBlep()
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if dt < 1e-9 { return 0.0; }
    if t < dt {
        let t = t / dt;
        t + t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + t + t + 1.0
    } else {
        0.0
    }
}
