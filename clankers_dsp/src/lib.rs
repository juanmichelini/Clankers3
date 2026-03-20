mod drums;
mod envelope;
mod ms20_filter;
mod rng;

use drums::DrumsEngine;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

/// Drums engine — main-thread rendering via AudioBufferSourceNode.
///
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
        ClankersDrums {
            engine: DrumsEngine::new(seed),
        }
    }

    /// Trigger a hit and immediately render its full tail.
    /// Returns a Float32Array ready to load into an AudioBuffer.
    /// Trailing silence is trimmed so the buffer is as short as needed.
    pub fn trigger_render(&mut self, voice_id: u8, velocity: f32, p0: f32, p1: f32, p2: f32) -> Float32Array {
        self.engine.trigger(voice_id, velocity, p0, p1, p2);

        // Up to 3 s at 44100 Hz — plenty for any hit tail
        let max = 44100 * 3;
        let mut buf = vec![0.0f32; max];
        self.engine.process(&mut buf);

        // Trim trailing near-silence
        let end = buf.iter()
            .rposition(|&s| s.abs() > 1e-5)
            .map(|i| i + 1)
            .unwrap_or(512);

        Float32Array::from(&buf[..end])
    }
}
