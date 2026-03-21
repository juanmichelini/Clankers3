/// Pro-One style bass voice
/// Ported from bassYnth/Source/ProOneVoice.cpp
///
/// ClankerBoy CC map (t:2):
///   CC74  Filter cutoff norm (0-127 → 0-1)
///   CC71  Resonance          (0-127 → 0-1)
///   CC73  Amp attack         (0-127 → 0.001-0.5 s)
///   CC75  Amp decay          (0-127 → 0.01-2 s)
///   CC79  Amp sustain        (0-127 → 0-1)
///   CC72  Amp release        (0-127 → 0.01-2 s)
///   CC22  Filter env attack  (0-127 → 0.001-0.5 s)
///   CC23  Filter env decay   (0-127 → 0.01-2 s)
///   CC24  Filter env sustain (0-127 → 0-1)
///   CC25  Filter env release (0-127 → 0.01-2 s)
///   CC18  Osc B detune cents (0-127 → -50..+50 cents)
///   CC5   Glide time         (0-127 → 0 .. 0.5 s)

use crate::envelope::Envelope;
use crate::oscillator::{Oscillator, Waveform};
use crate::rng::Rng;
use crate::tpt_ladder::TptLadder;

const SR: f32 = 44100.0;

/// Params passed per-note trigger (derived from ClankerBoy track CCs)
#[derive(Clone, Copy)]
pub struct BassParams {
    pub cutoff_norm:     f32,  // 0-1  (CC74)
    pub resonance:       f32,  // 0-1  (CC71)
    pub amp_attack:      f32,  // secs (CC73)
    pub amp_decay:       f32,  // secs (CC75)
    pub amp_sustain:     f32,  // 0-1  (CC79)
    pub amp_release:     f32,  // secs (CC72)
    pub flt_env_amount:  f32,  // 0-1  derived from CC74 headroom
    pub flt_decay:       f32,  // secs (CC23)
    pub detune_cents:    f32,  // cents (CC18, default 0)
    pub glide_time:      f32,  // secs (CC5, default 0)
    pub drive:           f32,  // ≥1.0 (fixed for now)
}

impl Default for BassParams {
    fn default() -> Self {
        BassParams {
            cutoff_norm:    0.45,
            resonance:      0.30,
            amp_attack:     0.002,
            amp_decay:      0.4,
            amp_sustain:    0.7,
            amp_release:    0.12,
            flt_env_amount: 0.35,
            flt_decay:      0.18,
            detune_cents:   0.0,
            glide_time:     0.0,
            drive:          1.5,
        }
    }
}

pub struct BassVoice {
    osc_a:    Oscillator,
    osc_b:    Oscillator,
    sub:      Oscillator,
    filter:   TptLadder,
    amp_env:  Envelope,
    flt_env:  Envelope,
    rng:      Rng,

    current_freq: f32,
    target_freq:  f32,
    glide_coeff:  f32,
    active:       bool,
}

impl BassVoice {
    pub fn new(seed: u32) -> Self {
        BassVoice {
            osc_a:    Oscillator::new(SR),
            osc_b:    Oscillator::new(SR),
            sub:      Oscillator::new(SR),
            filter:   TptLadder::new(SR),
            amp_env:  Envelope::new(SR),
            flt_env:  Envelope::new(SR),
            rng:      Rng::new(seed),
            current_freq: 110.0,
            target_freq:  110.0,
            glide_coeff:  1.0,
            active:       false,
        }
    }

    pub fn trigger(&mut self, midi_note: u8, velocity: f32, p: &BassParams) {
        self.target_freq = midi_to_hz(midi_note);
        if !self.active || p.glide_time < 0.001 {
            self.current_freq = self.target_freq;
        }
        self.glide_coeff = if p.glide_time > 0.001 {
            1.0 - (-1.0 / (SR * p.glide_time)).exp()
        } else {
            1.0
        };

        self.amp_env.set_adsr(p.amp_attack, p.amp_decay, p.amp_sustain, p.amp_release);
        self.amp_env.note_on();
        self.flt_env.set_adsr(0.001, p.flt_decay, 0.0, p.flt_decay * 0.5);
        self.flt_env.note_on();

        // Velocity-scale osc levels
        self.osc_a.level = velocity * 0.5;
        self.osc_b.level = velocity * 0.3;
        self.sub.level   = velocity * 0.25;

        self.active = true;
    }

    pub fn release(&mut self) {
        self.amp_env.note_off();
        self.flt_env.note_off();
    }

    /// Render `n` samples into `out` (add, don't overwrite)
    pub fn process(&mut self, out: &mut [f32], p: &BassParams) {
        for s in out.iter_mut() {
            if !self.active { break; }

            // Glide
            self.current_freq += self.glide_coeff * (self.target_freq - self.current_freq);

            // Osc B detune
            let freq_b = self.current_freq * cents_to_ratio(p.detune_cents);

            let sa = self.osc_a.next(self.current_freq, Waveform::Saw);
            let sb = self.osc_b.next(freq_b,            Waveform::Saw);
            let ss = self.sub.next(self.current_freq * 0.5, Waveform::Square);
            let noise = (self.rng.next_f32() * 2.0 - 1.0) * 0.05;

            let mixed = sa + sb + ss + noise;

            // Filter cutoff: base + filter envelope
            let flt_env_val = self.flt_env.process();
            let cutoff_norm = (p.cutoff_norm + flt_env_val * p.flt_env_amount).clamp(0.0, 1.0);
            let cutoff_hz   = norm_to_cutoff_hz(cutoff_norm);

            let filtered = self.filter.process(mixed, cutoff_hz, p.resonance, p.drive);

            let amp = self.amp_env.process();
            if amp < 1e-6 && !self.amp_env.is_active() { self.active = false; }

            *s += filtered * amp * 0.6;
        }
    }

    pub fn is_active(&self) -> bool { self.active }
}

/// Simple polyphonic engine — 8 voices, round-robin allocation
pub struct BassEngine {
    voices: Vec<BassVoice>,
    next_voice: usize,
}

impl BassEngine {
    pub fn new(seed: u32) -> Self {
        let voices = (0..8).map(|i| BassVoice::new(seed.wrapping_add(i * 1234))).collect();
        BassEngine { voices, next_voice: 0 }
    }

    pub fn trigger(&mut self, midi_note: u8, velocity: f32, p: &BassParams) {
        // Steal a voice: prefer idle, otherwise oldest (round-robin)
        let idx = (0..self.voices.len())
            .find(|&i| !self.voices[i].is_active())
            .unwrap_or_else(|| {
                let v = self.next_voice;
                self.next_voice = (v + 1) % self.voices.len();
                v
            });
        self.voices[idx].trigger(midi_note, velocity, p);
    }

    pub fn process(&mut self, buf: &mut [f32], p: &BassParams) {
        for v in self.voices.iter_mut() {
            if v.is_active() { v.process(buf, p); }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

#[inline]
fn cents_to_ratio(cents: f32) -> f32 {
    2.0_f32.powf(cents / 1200.0)
}

/// Exponential sweep 20 Hz → 20 000 Hz matching bassYnth cutoff mapping
#[inline]
pub fn norm_to_cutoff_hz(norm: f32) -> f32 {
    20.0 * 1000.0_f32.powf(norm.clamp(0.0, 1.0))
}
