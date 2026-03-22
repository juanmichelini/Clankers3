mod bass;
mod buchla;
mod chorus;
mod drums;
mod envelope;
mod lpg;
mod moog_ladder;
mod ms20_filter;
mod oscillator;
mod pads;
mod reverb;
mod rng;
mod tpt_ladder;
mod vactrol;
mod wavefolder;

use bass::{BassEngine, BassParams};
use buchla::{BuchlaEngine, BuchlaParams};
use drums::DrumsEngine;
use pads::{PadsEngine, PadsParams};
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

// ── Drums ─────────────────────────────────────────────────────────────────────

/// Voice IDs:  0=Kick  1=Snare  2=HiHat Closed  3=HiHat Open
///             4=Tom L  5=Tom M  6=Tom H
///
/// Trigger params (p0..p2):
///   Kick:   p0=pitch(0-1)  p1=sweep_time(0-1)  p2=decay(0-1)
///   Snare:  p0=pitch(0-1)  p1=decay(0-1)        p2=resonance(0-1)
///   HiHat:  p0=decay(0-1)  p1=cutoff(0-1)       p2=resonance(0-1)
///   Tom:    p0=pitch(0-1)  p1=decay(0-1)         p2=unused
#[wasm_bindgen]
pub struct ClankersDrums {
    engine: DrumsEngine,
}

#[wasm_bindgen]
impl ClankersDrums {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> ClankersDrums {
        ClankersDrums { engine: DrumsEngine::new(seed) }
    }

    /// Trigger a hit and immediately render its full tail.
    /// Uses an isolated voice — no shared engine state contamination.
    pub fn trigger_render(&mut self, voice_id: u8, velocity: f32, p0: f32, p1: f32, p2: f32) -> Float32Array {
        let vel = velocity.clamp(0.0, 1.0);
        let rng = &mut self.engine.rng;
        let voice = match voice_id {
            0 => drums::synth_kick(vel, p0, p1, p2, rng),
            1 => drums::synth_snare(vel, p0, p1, p2, rng),
            2 => drums::synth_hihat(vel, false, p0, p1, p2, rng),
            3 => drums::synth_hihat(vel, true,  p0, p1, p2, rng),
            4 => drums::synth_tom(vel, 41, p0, p1),
            5 => drums::synth_tom(vel, 43, p0, p1),
            6 => drums::synth_tom(vel, 45, p0, p1),
            _ => return Float32Array::new_with_length(0),
        };
        Float32Array::from(voice.into_buf().as_slice())
    }
}

// ── Bass ──────────────────────────────────────────────────────────────────────

/// Pro-One style polyphonic bass (8 voices, TPT ladder filter).
///
/// ClankerBoy CC map (all normalised 0-127):
///   CC74 cutoff  CC71 resonance  CC73 amp_attack  CC75 amp_decay
///   CC79 amp_sustain  CC72 amp_release  CC23 flt_decay  CC18 detune_cents
///   CC5  glide_time
///
/// trigger(midi_note, velocity_0_1, cc_json_string)
///   cc_json_string: JSON object of CC values, e.g. '{"74":80,"71":60}'
///
/// render(n_samples) → Float32Array  (call after trigger, before next trigger)
#[wasm_bindgen]
pub struct ClankersBass {
    engine: BassEngine,
}

#[wasm_bindgen]
impl ClankersBass {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> ClankersBass {
        ClankersBass { engine: BassEngine::new(seed) }
    }

    /// Trigger a note. cc_json: '{"74":80,"71":60}' or '{}'.
    /// hold_samples: note-on duration in samples (0 = use amp envelope only)
    pub fn trigger(&mut self, midi_note: u8, velocity: f32, hold_samples: u32, cc_json: &str) {
        let p = parse_bass_params(cc_json);
        self.engine.trigger(midi_note, velocity, hold_samples as usize, &p);
    }

    /// Render n_samples of audio (adds all active voices). Returns Float32Array.
    pub fn render(&mut self, n_samples: u32) -> Float32Array {
        let n = n_samples as usize;
        let mut buf = vec![0.0f32; n];
        // Use default params for render (params were captured at trigger time)
        let p = BassParams::default();
        self.engine.process(&mut buf, &p);
        Float32Array::from(buf.as_slice())
    }

    /// Trigger + render full tail — isolated single voice, no shared state.
    /// Note: ClankerBoy uses MIDI 0-23 for bass roots. We transpose +24 semitones
    /// so the actual synthesis sits in the audible 50-200 Hz range.
    pub fn trigger_render(&mut self, midi_note: u8, velocity: f32, hold_samples: u32, cc_json: &str) -> Float32Array {
        let p = parse_bass_params(cc_json);

        // Fresh voice each call — prevents cross-contamination when chords render
        let mut voice = bass::BassVoice::new(0xba55);
        let transposed = midi_note.saturating_add(48); // +4 octaves into audible range
        voice.trigger(transposed, velocity, hold_samples as usize, &p);

        let max = 44100 * 4;
        let mut buf = vec![0.0f32; max];
        voice.process(&mut buf, &p);

        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| (i + 441).min(max))
            .unwrap_or(1024);

        Float32Array::from(&buf[..end])
    }
}

// ── CC JSON → BassParams ──────────────────────────────────────────────────────

fn parse_bass_params(cc_json: &str) -> BassParams {
    let mut p = BassParams::default();

    // Minimal JSON key:value parser (no external crate needed)
    for (key, val) in parse_cc_map(cc_json) {
        let n = val / 127.0;
        match key {
            74 => p.cutoff_norm    = n,
            71 => p.resonance      = n,
            73 => p.amp_attack     = 0.001 + n * 0.499,
            75 => p.amp_decay      = 0.01  + n * 1.99,
            79 => p.amp_sustain    = n,
            72 => p.amp_release    = 0.01  + n * 1.99,
            23 => p.flt_decay      = 0.01  + n * 0.99,
            18 => p.detune_cents   = (n * 100.0) - 50.0,
            5  => p.glide_time     = n * 0.5,
            _  => {}
        }
    }
    p
}

// ── Buchla ────────────────────────────────────────────────────────────────────

/// Buchla 259/292 — percussive LPG arp with FM + wavefolding (8 voices).
///
/// ClankerBoy CC map (t:1):
///   CC74 cutoff  CC71 resonance  CC20 wavefold  CC17 fm_depth
///   CC18 fm_index  CC19 env_decay  CC16 volume
#[wasm_bindgen]
pub struct ClankersBuchla {
    engine: BuchlaEngine,
}

#[wasm_bindgen]
impl ClankersBuchla {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ClankersBuchla {
        ClankersBuchla { engine: BuchlaEngine::new() }
    }

    /// Trigger + render full tail — isolated single voice.
    pub fn trigger_render(&mut self, midi_note: u8, velocity: f32, cc_json: &str) -> Float32Array {
        let p = parse_buchla_params(cc_json);

        let mut voice = buchla::BuchlaVoice::new();
        voice.trigger(midi_note, velocity, &p);

        let max = 44100 * 3;
        let mut buf = vec![0.0f32; max];
        voice.process(&mut buf, &p);

        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| (i + 441).min(max))
            .unwrap_or(1024);

        Float32Array::from(&buf[..end])
    }
}

// ── Pads ──────────────────────────────────────────────────────────────────────

/// HybridSynth pads — Moog ladder + ADSR + chorus + reverb (8 polyphonic voices).
///
/// trigger_render(midi_note, velocity, hold_samples, cc_json) → stereo Float32Array
/// hold_samples: note-on duration in samples (beat * 60/bpm * 44100)
/// Returns interleaved stereo [L0, R0, L1, R1, ...]
#[wasm_bindgen]
pub struct ClankersPads {
    engine: PadsEngine,
}

#[wasm_bindgen]
impl ClankersPads {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ClankersPads {
        ClankersPads { engine: PadsEngine::new() }
    }

    pub fn trigger_render(
        &mut self,
        midi_note:    u8,
        velocity:     f32,
        hold_samples: u32,
        cc_json:      &str,
    ) -> Float32Array {
        let p    = parse_pads_params(cc_json);
        let hold = hold_samples as usize;

        // Render: attack + hold + full release tail — isolated voice, no shared state
        let release_tail = (p.amp_release * 44100.0) as usize + 4410;
        let total        = hold + release_tail;

        let mut buf_l = vec![0.0f32; total];
        let mut buf_r = vec![0.0f32; total];

        let mut voice = pads::PadsVoice::new();
        voice.trigger(midi_note, velocity, hold, &p);
        voice.process(&mut buf_l, &mut buf_r, &p);

        // Trim trailing silence
        let end = buf_l.iter().zip(buf_r.iter())
            .rposition(|(&l, &r)| l.abs() > 1e-5 || r.abs() > 1e-5)
            .map(|i| (i + 441).min(total))
            .unwrap_or(1024);

        // Interleave stereo
        let mut interleaved = vec![0.0f32; end * 2];
        for i in 0..end {
            interleaved[i * 2]     = buf_l[i];
            interleaved[i * 2 + 1] = buf_r[i];
        }

        Float32Array::from(interleaved.as_slice())
    }
}

fn parse_pads_params(cc_json: &str) -> PadsParams {
    let mut p = PadsParams::default();
    for (key, val) in parse_cc_map(cc_json) {
        let n = val / 127.0;
        match key {
            74 => p.cutoff_hz    = 20.0 + n * 7980.0,   // 20–8000 Hz
            71 => p.resonance    = n * 0.9,
            73 => p.amp_attack   = 0.05 + n * 3.95,
            75 => p.amp_decay    = 0.05 + n * 1.95,
            79 => p.amp_sustain  = n,
            72 => p.amp_release  = 0.1  + n * 3.9,
            88 => p.reverb_size  = n,
            91 => p.reverb_mix   = n,
            29 => p.chorus_rate  = 0.1  + n * 4.9,
            30 => p.chorus_depth = n,
            31 => p.chorus_mix   = n,
            _  => {}
        }
    }
    p
}

fn parse_buchla_params(cc_json: &str) -> BuchlaParams {
    let mut p = BuchlaParams::default();
    for (key, val) in parse_cc_map(cc_json) {
        let n = val / 127.0;
        match key {
            74 => p.cutoff_norm = n,
            71 => p.resonance   = n,          // LPG clamps to 0.85 internally
            20 => p.fold_amount = n,
            17 => p.fm_depth    = n,
            18 => p.fm_index    = n,
            19 => p.decay_s     = 0.005 + n * 0.795,  // 5 ms .. 800 ms
            16 => p.volume      = n,
            _  => {}
        }
    }
    p
}

/// Parse a flat JSON CC object: {"74": 80, "71": 60} → [(74, 80.0), (71, 60.0)]
fn parse_cc_map(s: &str) -> Vec<(u8, f32)> {
    let mut out = Vec::new();
    let s = s.trim().trim_start_matches('{').trim_end_matches('}');
    for pair in s.split(',') {
        let mut parts = pair.splitn(2, ':');
        let k = parts.next().unwrap_or("").trim().trim_matches('"').trim();
        let v = parts.next().unwrap_or("").trim().trim_matches('"').trim();
        if let (Ok(kn), Ok(vf)) = (k.parse::<u8>(), v.parse::<f32>()) {
            out.push((kn, vf));
        }
    }
    out
}
