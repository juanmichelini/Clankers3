/// MS-20 Sallen-Key style filter — matches AntigravityDrums MultiModeFilter.cpp.
/// Two cascaded integrators with tanh resonance feedback. 12 dB/oct.
#[derive(Clone)]
pub struct Ms20Filter {
    sample_rate: f32,
    pub mode: FilterMode,
    g: f32,
    k: f32, // = 2 * resonance
    s1: f32,
    s2: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FilterMode {
    Lp,
    Hp,
}

impl Ms20Filter {
    pub fn new(sample_rate: f32) -> Self {
        let mut f = Self {
            sample_rate,
            mode: FilterMode::Lp,
            g: 0.0,
            k: 0.0,
            s1: 0.0,
            s2: 0.0,
        };
        f.set_cutoff(1000.0);
        f
    }

    pub fn set_cutoff(&mut self, hz: f32) {
        let hz = hz.clamp(20.0, self.sample_rate * 0.49);
        self.g = 1.0 - (-2.0 * core::f32::consts::PI * hz / self.sample_rate).exp();
    }

    pub fn set_resonance(&mut self, res: f32) {
        self.k = 2.0 * res.clamp(0.0, 1.0);
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let input = x - self.k * self.s2.tanh();
        self.s1 += self.g * (input.tanh() - self.s1);
        self.s2 += self.g * (self.s1 - self.s2);
        match self.mode {
            FilterMode::Lp => self.s2,
            FilterMode::Hp => x - self.s2,
        }
    }
}
