/// HybridSynth pads voice — sustained chords with Moog ladder + chorus + reverb
/// Ported from HybridSynth/Source/SynthVoice.cpp
///
/// Signal chain:
///   Saw + Triangle mix → Moog ladder → Amp ADSR → Chorus → Reverb
///
/// ClankerBoy CC map (t:6):
///   CC74  Filter cutoff    (0-127 → 20-8000 Hz)
///   CC71  Filter resonance (0-127 → 0-0.9)
///   CC73  Amp attack       (0-127 → 0.05-4 s)
///   CC72  Amp release      (0-127 → 0.1-4 s)
///   CC75  Amp decay        (0-127 → 0.05-2 s)
///   CC79  Amp sustain      (0-127 → 0-1)
///   CC88  Reverb size      (0-127 → 0-1)
///   CC91  Reverb mix       (0-127 → 0-1)
///   CC29  Chorus rate      (0-127 → 0.1-5 Hz)
///   CC30  Chorus depth     (0-127 → 0-1)
///   CC31  Chorus mix       (0-127 → 0-1)

use crate::chorus::Chorus;
use crate::envelope::Envelope;
use crate::moog_ladder::MoogLadder;
use crate::oscillator::{Oscillator, Waveform};
use crate::reverb::Reverb;

const SR: f32 = 44100.0;

#[derive(Clone, Copy)]
pub struct PadsParams {
    pub cutoff_hz:    f32,
    pub resonance:    f32,
    pub amp_attack:   f32,
    pub amp_decay:    f32,
    pub amp_sustain:  f32,
    pub amp_release:  f32,
    pub reverb_size:  f32,
    pub reverb_mix:   f32,
    pub chorus_rate:  f32,
    pub chorus_depth: f32,
    pub chorus_mix:   f32,
}

impl Default for PadsParams {
    fn default() -> Self {
        PadsParams {
            cutoff_hz:    800.0,
            resonance:    0.15,
            amp_attack:   0.4,
            amp_decay:    0.5,
            amp_sustain:  0.8,
            amp_release:  1.2,
            reverb_size:  0.65,
            reverb_mix:   0.45,
            chorus_rate:  0.5,
            chorus_depth: 0.4,
            chorus_mix:   0.35,
        }
    }
}

pub struct PadsVoice {
    osc_saw:        Oscillator,
    osc_tri:        Oscillator,
    filter:         MoogLadder,
    amp_env:        Envelope,
    chorus:         Chorus,
    reverb:         Reverb,
    freq:           f32,
    hold_remaining: usize,
    released:       bool,
    active:         bool,
}

impl PadsVoice {
    pub fn new() -> Self {
        PadsVoice {
            osc_saw:        Oscillator::new(SR),
            osc_tri:        Oscillator::new(SR),
            filter:         MoogLadder::new(SR),
            amp_env:        Envelope::new(SR),
            chorus:         Chorus::new(SR),
            reverb:         Reverb::new(SR),
            freq:           440.0,
            hold_remaining: 0,
            released:       false,
            active:         false,
        }
    }

    /// hold_samples: render note-on for this many samples then release.
    pub fn trigger(&mut self, midi_note: u8, velocity: f32, hold_samples: usize, p: &PadsParams) {
        self.freq = midi_to_hz(midi_note);
        self.osc_saw.level = velocity * 0.6;
        self.osc_tri.level = velocity * 0.4;

        self.amp_env.set_adsr(p.amp_attack, p.amp_decay, p.amp_sustain, p.amp_release);
        self.amp_env.note_on();

        self.hold_remaining = hold_samples;
        self.released = false;
        self.active   = true;
    }

    /// Write stereo output into buf_l / buf_r (additive).
    pub fn process(&mut self, buf_l: &mut [f32], buf_r: &mut [f32], p: &PadsParams) {
        for (sl, sr) in buf_l.iter_mut().zip(buf_r.iter_mut()) {
            if !self.active { break; }

            // Note-off after hold duration
            if self.hold_remaining > 0 {
                self.hold_remaining -= 1;
            } else if !self.released {
                self.amp_env.note_off();
                self.released = true;
            }

            let amp = self.amp_env.process();
            if amp < 1e-6 && !self.amp_env.is_active() {
                self.active = false;
                break;
            }

            let saw  = self.osc_saw.next(self.freq, Waveform::Saw);
            let tri  = self.osc_tri.next(self.freq, Waveform::Triangle);
            let mixed = saw + tri;

            let filtered = self.filter.process(mixed, p.cutoff_hz, p.resonance, 1.0);
            let dry = filtered * amp * 0.8;

            // Chorus → stereo spread
            let (cl, cr) = self.chorus.process(dry, dry, p.chorus_rate, p.chorus_depth, p.chorus_mix);

            // Reverb (mono in, added to both channels)
            let rev     = self.reverb.process_mono((cl + cr) * 0.5, p.reverb_size, 0.4);
            let rev_wet = rev * p.reverb_mix;
            let dry_mix = 1.0 - p.reverb_mix;

            *sl += cl * dry_mix + rev_wet;
            *sr += cr * dry_mix + rev_wet;
        }
    }

    pub fn is_active(&self) -> bool { self.active }
}

/// 8-voice polyphonic engine (stereo output)
pub struct PadsEngine {
    voices:     Vec<PadsVoice>,
    next_voice: usize,
}

impl PadsEngine {
    pub fn new() -> Self {
        PadsEngine {
            voices:     (0..8).map(|_| PadsVoice::new()).collect(),
            next_voice: 0,
        }
    }

    pub fn trigger(&mut self, midi_note: u8, velocity: f32, hold_samples: usize, p: &PadsParams) {
        let idx = (0..self.voices.len())
            .find(|&i| !self.voices[i].is_active())
            .unwrap_or_else(|| {
                let v = self.next_voice;
                self.next_voice = (v + 1) % self.voices.len();
                v
            });
        self.voices[idx].trigger(midi_note, velocity, hold_samples, p);
    }

    pub fn process(&mut self, buf_l: &mut [f32], buf_r: &mut [f32], p: &PadsParams) {
        for v in self.voices.iter_mut() {
            if v.is_active() { v.process(buf_l, buf_r, p); }
        }
    }
}

#[inline]
fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}
