/// Drum voice synthesis — modelled after AntigravityDrums VST source.
///
/// Voice IDs:
///   0 = Kick    1 = Snare   2 = HiHat Closed   3 = HiHat Open
///   4 = Tom L   5 = Tom M   6 = Tom H
///
/// Trigger params (p0..p2) per voice:
///   Kick:        p0=pitch_norm(0..1)  p1=sweep_time(0..1)  p2=decay(0..1)
///   Snare:       p0=pitch_norm        p1=decay             p2=resonance(0..1)
///   HiHat C/O:   p0=decay_norm        p1=cutoff_norm       p2=resonance(0..1)
///   Tom L/M/H:   p0=pitch_norm        p1=decay             p2=unused

use crate::envelope::Envelope;
use crate::ms20_filter::{FilterMode, Ms20Filter};
use crate::rng::Rng;

const SR: f32 = 44100.0;
const PI: f32 = core::f32::consts::PI;
const MAX_VOICE_SAMPLES: usize = SR as usize * 3; // 3 s max tail

pub struct DrumVoice {
    pub(crate) buf: Vec<f32>,
    pos: usize,
    active: bool,
}

impl DrumVoice {
    fn empty() -> Self {
        Self { buf: Vec::new(), pos: 0, active: false }
    }

    fn from_buf(buf: Vec<f32>) -> Self {
        let active = !buf.is_empty();
        Self { buf, pos: 0, active }
    }

    pub fn into_buf(self) -> Vec<f32> {
        self.buf
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Read one sample; returns 0.0 when exhausted.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }
        let s = self.buf[self.pos];
        self.pos += 1;
        if self.pos >= self.buf.len() {
            self.active = false;
        }
        s
    }
}

// ─── Kick ────────────────────────────────────────────────────────────────────

pub fn synth_kick(velocity: f32, pitch_norm: f32, sweep_time: f32, decay_norm: f32, rng: &mut Rng) -> DrumVoice {
    // pitch_norm 0..1 → end_freq 20..200 Hz  (matches VST: 20 + pitch*180)
    let end_freq = 20.0 + pitch_norm * 180.0;
    // sweep_amount matches VST: 50 + sweepTime*600
    let sweep_amount = 50.0 + sweep_time * 600.0;
    let start_freq = end_freq + sweep_amount;

    // Amp envelope: attack=1ms, decay=decay_norm*2 s (matches VST: p.decay*2)
    let decay_s = 0.02 + decay_norm * 1.98_f32; // 0.02..2.0 s
    let total = ((decay_s * 2.0).max(0.2) * SR) as usize;
    let total = total.min(MAX_VOICE_SAMPLES);

    let mut amp_env = Envelope::new(SR);
    amp_env.set_adsr(0.001, decay_s, 0.0, 0.05);
    amp_env.note_on();

    let mut pitch_env = Envelope::new(SR);
    pitch_env.set_adsr(0.001, sweep_time * 0.4 + 0.005, 0.0, 0.0);
    pitch_env.note_on();

    // Click decay coeff: exp(-1 / (sr * 0.003))  ~3 ms
    let click_decay = (-1.0 / (SR * 0.003)).exp();
    let mut click_level = 0.6 * velocity;
    let mut noise_state: u32 = 0x12345678u32.wrapping_add((velocity * 1000.0) as u32);

    let mut phase = 0.0_f32;
    let mut buf = Vec::with_capacity(total);

    for _ in 0..total {
        let pitch_mod = pitch_env.process();
        let current_freq = end_freq + (start_freq - end_freq) * pitch_mod;
        phase += current_freq / SR;
        let tone = (phase * 2.0 * PI).sin();

        let noise = Rng::lcg_f32(&mut noise_state);
        let click = noise * click_level;
        click_level *= click_decay;

        let amp = amp_env.process();
        let raw = (tone + click) * amp * velocity;
        let sample = (raw * 1.5).tanh(); // soft clip (matches VST tanh(raw*1.5))
        buf.push(sample);

        if !amp_env.is_active() {
            break;
        }
    }

    DrumVoice::from_buf(buf)
}

// ─── Snare ───────────────────────────────────────────────────────────────────

pub fn synth_snare(velocity: f32, pitch_norm: f32, decay_norm: f32, resonance: f32, rng: &mut Rng) -> DrumVoice {
    // tone freq: 100 + pitch*200  (matches VST)
    let tone_freq = 100.0 + pitch_norm * 200.0;
    let decay_s = 0.02 + decay_norm * 0.58; // 0.02..0.6 s

    let mut tone_env = Envelope::new(SR);
    tone_env.set_adsr(0.001, 0.15, 0.0, 0.02); // tone decay fixed at 0.15 s (VST)
    tone_env.note_on();

    let mut noise_env = Envelope::new(SR);
    noise_env.set_adsr(0.001, decay_s, 0.0, 0.02);
    noise_env.note_on();

    let mut filter = Ms20Filter::new(SR);
    filter.mode = FilterMode::Hp;
    filter.set_cutoff(200.0);
    filter.set_resonance(resonance * 0.5);

    let total = ((decay_s * 3.0).max(0.25) * SR) as usize;
    let total = total.min(MAX_VOICE_SAMPLES);

    let mut tone_phase = 0.0_f32;
    let mut buf = Vec::with_capacity(total);

    for _ in 0..total {
        let t_amp = tone_env.process();
        let n_amp = noise_env.process();

        tone_phase += tone_freq / SR;
        let tone = (tone_phase * 2.0 * PI).sin();
        let noise = rng.next_f32();

        // Mix: tone*0.6 + noise*0.5  (matches VST)
        let mixed = tone * t_amp * 0.6 + noise * n_amp * 0.5;
        let out = filter.process(mixed * velocity);
        buf.push(out);

        if !tone_env.is_active() && !noise_env.is_active() {
            break;
        }
    }

    DrumVoice::from_buf(buf)
}

// ─── HiHat ───────────────────────────────────────────────────────────────────

pub fn synth_hihat(velocity: f32, open: bool, decay_norm: f32, cutoff_norm: f32, resonance: f32, rng: &mut Rng) -> DrumVoice {
    // Closed: 0.02 + decay*0.08 s,  Open: decay*2 s min 0.08  (matches VST)
    let decay_s = if open {
        (decay_norm * 2.0).max(0.08)
    } else {
        0.02 + decay_norm * 0.08
    };

    let mut env = Envelope::new(SR);
    env.set_adsr(0.001, decay_s, 0.0, 0.01);
    env.note_on();

    // Cutoff: 800 + cutoff_norm * 12000  (matches VST: 800 + filterCutoff*12000)
    let cutoff_hz = 800.0 + cutoff_norm * 12000.0;
    let mut filter = Ms20Filter::new(SR);
    filter.mode = FilterMode::Hp;
    filter.set_cutoff(cutoff_hz);
    filter.set_resonance(resonance * 0.4);

    let total = ((decay_s * 3.0).max(0.05) * SR) as usize;
    let total = total.min(MAX_VOICE_SAMPLES);
    let mut buf = Vec::with_capacity(total);

    for _ in 0..total {
        let amp = env.process();
        let noise = rng.next_f32();
        // VST output scale: 0.5
        let out = filter.process(noise * amp * velocity * 0.5);
        buf.push(out);

        if !env.is_active() {
            break;
        }
    }

    DrumVoice::from_buf(buf)
}

// ─── Tom ─────────────────────────────────────────────────────────────────────

pub fn synth_tom(velocity: f32, midi_note: u8, pitch_norm: f32, decay_norm: f32) -> DrumVoice {
    // Base freq from MIDI note * pitch multiplier  (matches VST TomVoice)
    let note_freq = 440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);
    let pitch_mult = 2.0_f32.powf((pitch_norm - 0.5) * 2.0);
    let base_freq = note_freq * pitch_mult;

    let decay_s = 0.05 + decay_norm * 0.45; // 0.05..0.5 s

    let mut amp_env = Envelope::new(SR);
    amp_env.set_adsr(0.001, decay_s, 0.0, 0.02);
    amp_env.note_on();

    // Pitch envelope: 100 ms decay, adds pitchMod * 0.5 * baseFreq  (matches VST)
    let mut pitch_env = Envelope::new(SR);
    pitch_env.set_adsr(0.001, 0.1, 0.0, 0.0);
    pitch_env.note_on();

    let mut filter = Ms20Filter::new(SR);
    filter.mode = FilterMode::Lp;
    filter.set_cutoff(base_freq * 8.0);
    filter.set_resonance(0.1);

    let total = ((decay_s * 2.5).max(0.15) * SR) as usize;
    let total = total.min(MAX_VOICE_SAMPLES);

    let mut phase = 0.0_f32;
    let mut buf = Vec::with_capacity(total);

    for _ in 0..total {
        let amp = amp_env.process();
        let pitch_mod = pitch_env.process();

        if !amp_env.is_active() {
            break;
        }

        let current_freq = base_freq * (1.0 + pitch_mod * 0.5);
        phase += current_freq / SR;
        let sample = filter.process((phase * 2.0 * PI).sin() * amp * velocity);
        buf.push(sample);
    }

    DrumVoice::from_buf(buf)
}

// ─── Drums Engine ─────────────────────────────────────────────────────────────

pub struct DrumsEngine {
    voices: Vec<DrumVoice>,
    pub(crate) rng: Rng,
}

impl DrumsEngine {
    pub fn new(seed: u32) -> Self {
        Self {
            voices: Vec::with_capacity(16),
            rng: Rng::new(seed),
        }
    }

    /// Trigger a drum voice.
    /// voice_id: 0=Kick 1=Snare 2=HiHatClosed 3=HiHatOpen 4=TomL 5=TomM 6=TomH
    pub fn trigger(&mut self, voice_id: u8, velocity: f32, p0: f32, p1: f32, p2: f32) {
        let vel = velocity.clamp(0.0, 1.0);
        let voice = match voice_id {
            0 => synth_kick(vel, p0, p1, p2, &mut self.rng),
            1 => synth_snare(vel, p0, p1, p2, &mut self.rng),
            2 => synth_hihat(vel, false, p0, p1, p2, &mut self.rng),
            3 => synth_hihat(vel, true,  p0, p1, p2, &mut self.rng),
            4 => synth_tom(vel, 41, p0, p1),
            5 => synth_tom(vel, 43, p0, p1),
            6 => synth_tom(vel, 45, p0, p1),
            _ => return,
        };
        // Replace stale voice for same voice_id if any, else push
        if let Some(slot) = self.voices.iter_mut().find(|v| !v.is_active()) {
            *slot = voice;
        } else {
            self.voices.push(voice);
        }
    }

    /// Mix all active voices into output slice (add-accumulate).
    pub fn process(&mut self, output: &mut [f32]) {
        for v in self.voices.iter_mut() {
            if !v.is_active() {
                continue;
            }
            for s in output.iter_mut() {
                *s += v.next_sample();
            }
        }
        // Soft clip bus
        for s in output.iter_mut() {
            *s = s.tanh();
        }
    }
}
