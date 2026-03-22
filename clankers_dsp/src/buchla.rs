/// Buchla 259/292 voice — percussive LPG pluck with FM + wavefolding
/// Ported from Buchlasystem/Source/DSP/BuchlaVoice.cpp
///
/// ClankerBoy CC map (t:1):
///   CC74  LPG cutoff norm   (0-127 → 0-1)     sweet spot: 56 (0.44)
///   CC71  LPG resonance     (0-127 → 0-0.85)
///   CC20  Wavefold amount   (0-127 → 0-1)      sweet spot: 37 (0.29)
///   CC17  FM depth          (0-127 → 0-1)
///   CC18  FM index          (0-127 → 0-1)
///   CC19  Env decay         (0-127 → 5-800 ms)
///   CC16  Osc volume        (0-127 → 0-1)

use crate::lpg::Lpg;
use crate::oscillator::{Oscillator, Waveform};
use crate::vactrol::Vactrol;
use crate::wavefolder::Wavefolder;

const SR: f32 = 44100.0;

#[derive(Clone, Copy)]
pub struct BuchlaParams {
    pub cutoff_norm: f32,   // CC74 / 127
    pub resonance:   f32,   // CC71 / 127 * 0.85
    pub fold_amount: f32,   // CC20 / 127
    pub fm_depth:    f32,   // CC17 / 127
    pub fm_index:    f32,   // CC18 / 127
    pub decay_s:     f32,   // seconds
    pub volume:      f32,   // CC16 / 127
}

impl Default for BuchlaParams {
    fn default() -> Self {
        BuchlaParams {
            cutoff_norm: 0.44,
            resonance:   0.22,
            fold_amount: 0.29,
            fm_depth:    0.06,
            fm_index:    0.08,
            decay_s:     0.10,
            volume:      1.0,
        }
    }
}

pub struct BuchlaVoice {
    osc_principal: Oscillator,
    osc_mod:       Oscillator,
    lpg:           Lpg,
    vactrol:       Vactrol,
    gate:          f32,
    freq:          f32,
    mod_ratio:     f32,
    active:        bool,
}

impl BuchlaVoice {
    pub fn new() -> Self {
        BuchlaVoice {
            osc_principal: Oscillator::new(SR),
            osc_mod:       Oscillator::new(SR),
            lpg:           Lpg::new(SR),
            vactrol:       Vactrol::new(SR),
            gate:      0.0,
            freq:      440.0,
            mod_ratio: 1.0,
            active:    false,
        }
    }

    pub fn trigger(&mut self, midi_note: u8, velocity: f32, p: &BuchlaParams) {
        self.freq      = midi_to_hz(midi_note);
        self.mod_ratio = 0.5 + p.fm_index * 7.5;
        self.osc_principal.level = velocity * p.volume;
        self.osc_mod.level       = 1.0; // FM depth applied via fm_hz, not osc level

        self.vactrol.set_times(0.001, p.decay_s);
        // Fire at cutoff-dependent level: low cutoff = more closed LPG = darker pluck
        // This ensures base_cutoff actually shapes the tone, not just the tail.
        let fire_level = 0.4 + p.cutoff_norm * 0.6; // 0.4..1.0
        self.vactrol.fire_at(fire_level);
        self.gate   = 0.0;    // gate stays off; vactrol decays from fire_level
        self.active = true;
    }

    pub fn process(&mut self, out: &mut [f32], p: &BuchlaParams) {
        for s in out.iter_mut() {
            if !self.active { break; }

            // Vactrol CV — AD transient: gate fires once then drops
            let cv = self.vactrol.process(self.gate);
            self.gate = 0.0; // drop gate after first sample → pure decay

            // Mod oscillator
            let mod_freq = self.freq * self.mod_ratio;
            let mod_out  = self.osc_mod.next(mod_freq, Waveform::Saw);

            // FM: deviation scales with carrier freq; fm_index only sets mod ratio (above)
            let fm_hz          = p.fm_depth * self.freq;
            let principal_freq = (self.freq + mod_out * fm_hz).max(20.0);

            // Principal oscillator
            let osc_out = self.osc_principal.next(principal_freq, Waveform::Saw);

            // Wavefolder
            let folded = Wavefolder::process(osc_out, p.fold_amount);

            // LPG — cutoff and VCA both driven by vactrol cv
            let filtered = self.lpg.process(folded, cv, p.cutoff_norm, p.resonance);

            if self.vactrol.is_idle() {
                self.active = false;
            }

            *s += filtered;
        }
    }

    pub fn is_active(&self) -> bool { self.active }
}

pub struct BuchlaEngine {
    voices:     Vec<BuchlaVoice>,
    next_voice: usize,
}

impl BuchlaEngine {
    pub fn new() -> Self {
        BuchlaEngine {
            voices:     (0..8).map(|_| BuchlaVoice::new()).collect(),
            next_voice: 0,
        }
    }

    pub fn trigger(&mut self, midi_note: u8, velocity: f32, p: &BuchlaParams) {
        let idx = (0..self.voices.len())
            .find(|&i| !self.voices[i].is_active())
            .unwrap_or_else(|| {
                let v = self.next_voice;
                self.next_voice = (v + 1) % self.voices.len();
                v
            });
        self.voices[idx].trigger(midi_note, velocity, p);
    }

    pub fn process(&mut self, buf: &mut [f32], p: &BuchlaParams) {
        for v in self.voices.iter_mut() {
            if v.is_active() { v.process(buf, p); }
        }
    }
}

#[inline]
fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}
