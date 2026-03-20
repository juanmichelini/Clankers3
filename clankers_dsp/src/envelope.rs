/// Linear ADSR — matches AntigravityDrums Envelope.cpp exactly.
#[derive(Clone)]
pub struct Envelope {
    sample_rate: f32,
    state: State,
    level: f64,
    attack_rate: f64,
    decay_rate: f64,
    release_rate: f64,
    sustain: f32,
}

#[derive(Clone, PartialEq)]
enum State {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Envelope {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            state: State::Idle,
            level: 0.0,
            attack_rate: 0.0,
            decay_rate: 0.0,
            release_rate: 0.0,
            sustain: 0.0,
        }
    }

    pub fn set_adsr(&mut self, attack_s: f32, decay_s: f32, sustain: f32, release_s: f32) {
        self.sustain = sustain;
        let sr = self.sample_rate as f64;
        self.attack_rate = 1.0 / (attack_s.max(0.0001) as f64 * sr);
        self.decay_rate = (1.0 - sustain as f64) / (decay_s.max(0.0001) as f64 * sr);
        self.release_rate = 1.0 / (release_s.max(0.0001) as f64 * sr);
    }

    pub fn note_on(&mut self) {
        self.state = State::Attack;
    }

    pub fn note_off(&mut self) {
        if self.state != State::Idle {
            self.state = State::Release;
        }
    }

    #[inline]
    pub fn process(&mut self) -> f32 {
        match self.state {
            State::Idle => {
                self.level = 0.0;
            }
            State::Attack => {
                self.level += self.attack_rate;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.state = State::Decay;
                }
            }
            State::Decay => {
                self.level -= self.decay_rate;
                if self.level <= self.sustain as f64 {
                    self.level = self.sustain as f64;
                    self.state = State::Sustain;
                }
            }
            State::Sustain => {
                self.level = self.sustain as f64;
            }
            State::Release => {
                self.level -= self.release_rate;
                if self.level <= 0.001 {
                    self.level = 0.0;
                    self.state = State::Idle;
                }
            }
        }
        self.level as f32
    }

    pub fn is_active(&self) -> bool {
        self.state != State::Idle
    }
}
