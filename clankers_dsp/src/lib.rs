mod bass;
mod buchla;
mod drums;
mod envelope;
mod lpg;
mod ms20_filter;
mod oscillator;
mod rng;
mod tpt_ladder;
mod vactrol;
mod wavefolder;

use bass::{BassEngine, BassParams};
use buchla::{BuchlaEngine, BuchlaParams};
use drums::DrumsEngine;
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
    pub fn trigger_render(&mut self, voice_id: u8, velocity: f32, p0: f32, p1: f32, p2: f32) -> Float32Array {
        self.engine.trigger(voice_id, velocity, p0, p1, p2);

        let max = 44100 * 3;
        let mut buf = vec![0.0f32; max];
        self.engine.process(&mut buf);

        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| i + 1)
            .unwrap_or(512);

        Float32Array::from(&buf[..end])
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
    pub fn trigger(&mut self, midi_note: u8, velocity: f32, cc_json: &str) {
        let p = parse_bass_params(cc_json);
        self.engine.trigger(midi_note, velocity, &p);
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

    /// Trigger + render full tail in one call (like ClankersDrums.trigger_render).
    pub fn trigger_render(&mut self, midi_note: u8, velocity: f32, cc_json: &str) -> Float32Array {
        let p = parse_bass_params(cc_json);
        self.engine.trigger(midi_note, velocity, &p);

        // Render up to 4 s, trim trailing silence
        let max = 44100 * 4;
        let mut buf = vec![0.0f32; max];
        self.engine.process(&mut buf, &p);

        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| (i + 441).min(max)) // keep a short tail
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

    /// Trigger + render full tail. cc_json: '{"74":72,"20":37,"17":8,"19":5}'
    pub fn trigger_render(&mut self, midi_note: u8, velocity: f32, cc_json: &str) -> Float32Array {
        let p = parse_buchla_params(cc_json);
        self.engine.trigger(midi_note, velocity, &p);

        let max = 44100 * 3;
        let mut buf = vec![0.0f32; max];
        self.engine.process(&mut buf, &p);

        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| (i + 441).min(max))
            .unwrap_or(1024);

        Float32Array::from(&buf[..end])
    }
}

fn parse_buchla_params(cc_json: &str) -> BuchlaParams {
    let mut p = BuchlaParams::default();
    for (key, val) in parse_cc_map(cc_json) {
        let n = val / 127.0;
        match key {
            74 => p.cutoff_norm = n,
            71 => p.resonance   = n * 0.85,
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
