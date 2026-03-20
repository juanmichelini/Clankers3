"""
The Clankers 3 -- Harmony Agent
Buchla Systems (arpeggios/stabs) + HybridSynth (Clouds granular pads).

run() now returns {"buchla": AudioSegment, "hybrid": AudioSegment}
instead of a single blended segment.

Music Sheet key: agents.harmony
  synth.buchla  -- arp parameters (attack_s, decay_s, sustain, release_s, waveshape, fm_depth, fm_index, filter_cutoff)
  synth.hybrid  -- pad parameters (attack_s, decay_s, sustain, release_s, detune, reverb, chorus)
  synth.hybrid.cloud -- Clouds-style granular (position, size, density, texture, spread, mix, freeze)

Envelope shape (full ADSR):
  attack_s  -- ramp from 0 to peak
  decay_s   -- ramp from peak down to sustain level  (0 = skip decay stage)
  sustain   -- 0.0-1.0 level held until note end  (1.0 = no decay, full sustain)
  release_s -- ramp from sustain level to 0 at note end
"""

import os
import re
import json
import requests
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize

try:
    import dawdreamer as daw
    HAS_DAWDREAMER = True
except ImportError:
    HAS_DAWDREAMER = False

try:
    from scipy.signal import butter, sosfilt
    def _lowpass(signal, cutoff_hz, sr):
        sos = butter(2, min(cutoff_hz, sr * 0.45), btype='low', fs=sr, output='sos')
        return sosfilt(sos, signal).astype(np.float32)
    HAS_SCIPY = True
except ImportError:
    def _lowpass(signal, cutoff_hz, sr):
        window = max(1, int(sr / (cutoff_hz * 4)))
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same').astype(np.float32)
    HAS_SCIPY = False


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE       = 44100
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


# ─── SEQUENCE GENERATION (LLM) ────────────────────────────────────────────────

def generate_progression(sheet: dict, api_key: str | None = None) -> dict:
    """
    Ask Claude to generate a chord progression + arpeggio pattern.
    Returns dict with 'chords' and 'arpeggio' keys.
    """
    harmony          = sheet["agents"]["harmony"]
    bpm              = sheet.get("bpm", 120)
    bars             = sheet.get("bars", 8)
    key              = sheet.get("key", "C minor")
    time_sig         = sheet.get("timeSignature", "4/4")
    mood             = sheet.get("mood", "")
    structure        = sheet.get("structure", "")
    global_notes     = sheet.get("globalNotes", "")
    tension          = float(sheet.get("tension", 0.4))
    harmonic_rhythm  = sheet.get("harmonic_rhythm", "medium")
    texture          = harmony.get("texture", "")
    instruction      = harmony.get("instruction", "")

    # Translate harmonic_rhythm + tension into concrete bar-duration guidance
    if harmonic_rhythm == "fast" or tension >= 0.7:
        chord_dur_hint = (
            "Fast harmonic rhythm: change chords every 1 bar. "
            "More chords = more harmonic movement = more tension."
        )
    elif harmonic_rhythm == "slow" or tension <= 0.25:
        chord_dur_hint = (
            "Slow harmonic rhythm: each chord lasts 4 bars. "
            "Fewer changes = more space and stillness."
        )
    elif harmonic_rhythm == "mixed":
        chord_dur_hint = (
            "Mixed harmonic rhythm: vary chord durations (1, 2, and 4 bars). "
            "Irregular changes create unpredictability and interest."
        )
    else:
        chord_dur_hint = (
            "Medium harmonic rhythm: change chords every 2 bars. "
            "Balanced -- enough movement without restlessness."
        )

    prompt = f"""You are a harmony synthesizer for an AI electronic music band.
Generate a chord progression and arpeggio pattern.

Key: {key}
BPM: {bpm}
Bars: {bars}
Time signature: {time_sig}
Section: {structure}
Mood: {mood}
Tension: {tension:.2f} (0=minimal, 1=peak)
Harmonic rhythm: {harmonic_rhythm}
Global notes: {global_notes}
Texture: {texture}
Instruction: {instruction}

{chord_dur_hint}

Return a JSON object with:
- "chords": array of chord objects, each with:
    - "name": chord name (e.g. "Dm7")
    - "midi_notes": array of 3-5 MIDI note numbers (middle register, e.g. 60-84)
    - "bars": how many bars this chord lasts (must sum to {bars} total)
- "arpeggio": object with:
    - "active": boolean
    - "pattern": array of indices into chord's midi_notes (e.g. [0,1,2,3,2,1])
    - "speed": "quarter", "8th", or "16th"
- "pad_cutoff": LP filter cutoff in Hz (200-2000) -- lower = darker/warmer

VOICE LEADING RULES for MIDI notes:
- Each chord's lowest MIDI note (root/bass) should move by step (≤2 semitones) or common tone
  from the previous chord's lowest note where musically possible -- avoid arbitrary large jumps
- Shared notes between chords: keep them on the same MIDI pitch (common-tone retention)
- The bass register (notes 55-67) should resolve down by step or hold when tension resolves

Rules:
- Chord bars must sum exactly to {bars}
- MIDI notes should be in the 55-84 range (bass-mid register for pads)
- Let the genre, mood, and section shape chord choices
- If texture mentions specific chords (e.g. "Dm7 -> Fmaj7"), use those
- Section matters: bridge = harmonic contrast, outro = resolution, climax = tension
- Match the mood: dark/dissociative = lower cutoff, lush/warm = higher

Return ONLY valid JSON. No explanation."""

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers=headers,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        text = resp.json()["content"][0]["text"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception as e:
        print(f"  [llm error] {e}")
        return _fallback_progression(sheet)


def _fallback_progression(sheet: dict) -> dict:
    """Minor iv -> VI fallback in the sheet's key."""
    return {
        "chords": [
            {"name": "i",    "midi_notes": [60, 63, 67, 70], "bars": sheet.get("bars", 8) // 2},
            {"name": "bIII", "midi_notes": [63, 67, 70, 74], "bars": sheet.get("bars", 8) // 2},
        ],
        "arpeggio": {"active": True, "pattern": [0, 1, 2, 3, 2, 1], "speed": "8th"},
        "pad_cutoff": 600,
    }


# ─── PAD SYNTHESIS (HYBRID) ───────────────────────────────────────────────────

def _sawtooth(freq: float, n: int) -> np.ndarray:
    phase = np.cumsum(np.full(n, freq / SAMPLE_RATE)) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def synth_pad_chord(midi_notes: list[int], duration_s: float, cutoff_hz: float,
                    attack_s: float = 0.5, decay_s: float = 0.0,
                    sustain: float = 1.0, release_s: float = 0.4,
                    detune_cents: int = 4) -> np.ndarray:
    """
    Synthesize a chord pad: layered detuned sawtooth oscillators per note,
    full ADSR envelope. Low-pass filtered for warmth.
    decay_s=0 / sustain=1.0 = classic long pad (holds at full level).
    decay_s>0 / sustain<1.0 = evolving pad that blooms then settles.
    """
    n   = max(1, int(SAMPLE_RATE * duration_s))
    out = np.zeros(n, dtype=np.float32)

    for midi in midi_notes:
        freq  = midi_to_hz(midi)
        cents = max(0, int(detune_cents))
        for dc in [0, cents, -cents]:
            f   = freq * (2 ** (dc / 1200))
            osc = _sawtooth(f, n) * 0.25
            out += osc

    # -- Full ADSR envelope --
    sus     = float(np.clip(sustain, 0.0, 1.0))
    att_n   = min(int(attack_s * SAMPLE_RATE), n)
    dec_n   = min(int(decay_s  * SAMPLE_RATE), n - att_n)
    rel_n   = min(int(release_s * SAMPLE_RATE), n)

    env = np.full(n, sus, dtype=np.float64)
    # Attack: 0 -> 1
    if att_n > 0:
        env[:att_n] = np.linspace(0.0, 1.0, att_n)
    # Decay: 1 -> sustain level
    if dec_n > 0:
        env[att_n:att_n + dec_n] = np.linspace(1.0, sus, dec_n)
    # Release: sustain -> 0 at end of note
    if rel_n > 0 and rel_n < n:
        env[n - rel_n:] = np.linspace(sus, 0.0, rel_n)

    out *= env.astype(np.float32)

    if HAS_SCIPY:
        out = _lowpass(out, cutoff_hz, SAMPLE_RATE)

    return out


# ─── ARP SYNTHESIS (BUCHLA) ───────────────────────────────────────────────────

def synth_arp_note(midi: int, duration_s: float,
                   attack_s: float = 0.008, decay_s: float = 0.0,
                   sustain: float = 1.0, release_s: float | None = None,
                   waveshape: float = 0.0) -> np.ndarray:
    """Single arpeggio note -- triangle/complex wave, full ADSR envelope.
    waveshape: 0.0=pure triangle (Buchla-style mellow), 1.0=folded waveshaper (complex).
    decay_s=0 and sustain=1.0 reproduce the old flat-sustain behaviour.
    """
    n    = max(1, int(SAMPLE_RATE * duration_s))
    freq = midi_to_hz(midi)

    phase = (np.cumsum(np.full(n, freq / SAMPLE_RATE)) % 1.0).astype(np.float32)
    tri   = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0

    if waveshape > 0.05:
        folded = np.sin(tri * np.pi * (1.0 + waveshape * 2.0))
        wave   = tri * (1.0 - waveshape) + folded * waveshape
    else:
        wave = tri

    # -- Full ADSR envelope --
    rel_s   = release_s if release_s is not None else duration_s * 0.6
    att_n   = min(int(attack_s  * SAMPLE_RATE), n)
    dec_n   = min(int(decay_s   * SAMPLE_RATE), n - att_n)
    rel_n   = min(int(rel_s     * SAMPLE_RATE), n)
    sus     = float(np.clip(sustain, 0.0, 1.0))

    env = np.full(n, sus, dtype=np.float64)
    # Attack: 0 -> 1
    if att_n > 0:
        env[:att_n] = np.linspace(0.0, 1.0, att_n)
    # Decay: 1 -> sustain level
    if dec_n > 0:
        env[att_n:att_n + dec_n] = np.linspace(1.0, sus, dec_n)
    # Release: sustain -> 0 at end of note
    if rel_n > 0 and rel_n < n:
        env[n - rel_n:] = np.linspace(sus, 0.0, rel_n)

    return (wave * env * 0.5).astype(np.float32)


def synth_stab_chord(midi_notes: list[int], duration_s: float,
                     attack_s: float = 0.005, decay_s: float = 0.0,
                     sustain: float = 1.0, release_s: float = 0.15,
                     waveshape: float = 0.0) -> np.ndarray:
    """
    Short chord stab for Buchla when arp is inactive.
    Layered arp notes played simultaneously -- brief pluck/stab character.
    """
    n   = max(1, int(SAMPLE_RATE * duration_s))
    out = np.zeros(n, dtype=np.float32)
    for midi in midi_notes:
        note = synth_arp_note(midi, duration_s,
                              attack_s=attack_s, decay_s=decay_s,
                              sustain=sustain, release_s=release_s,
                              waveshape=waveshape)
        if len(note) < n:
            note = np.pad(note, (0, n - len(note)))
        out += note[:n] * (0.5 / max(1, len(midi_notes)))
    return out


# ─── GRANULAR PROCESSOR (CLOUDS-STYLE) ───────────────────────────────────────

def _granulate(
    signal:   np.ndarray,
    sr:       int,
    position: float = 0.5,
    size:     float = 0.2,
    density:  float = 0.4,
    texture:  float = 0.3,
    spread:   float = 0.2,
    mix:      float = 0.5,
    freeze:   bool  = False,
) -> np.ndarray:
    """
    Clouds-style granular processor (mono float32 in -> float32 out).

    position  0-1   read head start position in the signal buffer
    size      0-1   grain duration: maps 20ms - 500ms
    density   0-1   grains/sec: maps 2 - 60
    texture   0-1   window shape: 0=Hann (smooth), 1=rectangular (harsh)
    spread    0-1   position scatter: randomises each grain's source position
    mix       0-1   wet/dry blend (0=dry, 1=full granular)
    freeze    bool  True -> all grains read from the same fixed position
    """
    n = len(signal)
    if n < 64 or mix < 0.005:
        return signal

    grain_n   = max(64, min(int((20.0 + size * 480.0) / 1000.0 * sr), n))
    rate_hz   = 2.0 + density * 58.0
    step_n    = max(1, int(sr / rate_hz))
    scatter_n = int(spread * grain_n)     # scatter radius in samples

    # Window: Hann (texture=0) -> rectangular (texture=1)
    hann      = np.hanning(grain_n).astype(np.float32)
    window    = hann * (1.0 - float(texture)) + float(texture)

    out    = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    rng        = np.random.default_rng(42)   # deterministic seed
    freeze_src = int(np.clip(position * (n - grain_n), 0, n - grain_n))
    bias       = int((float(position) - 0.5) * n)

    for onset in range(0, n, step_n):
        if freeze:
            src = freeze_src
        else:
            # Sequential playback biased by position:
            #   position=0.5 -> reads from current onset (normal)
            #   position<0.5 -> reads behind (stretched/echoing)
            #   position>0.5 -> reads ahead (compressed/rushing)
            src = int(np.clip(onset + bias, 0, n - grain_n))

        if scatter_n > 0:
            src += int(rng.integers(-scatter_n, scatter_n + 1))
        src = int(np.clip(src, 0, n - grain_n))

        grain = signal[src: src + grain_n] * window
        end   = min(onset + grain_n, n)
        wr    = end - onset
        out[onset:end]    += grain[:wr]
        counts[onset:end] += window[:wr]

    safe = counts > 1e-9
    out[safe] /= counts[safe]

    return signal * (1.0 - float(mix)) + out * float(mix)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _to_segment(signal: np.ndarray) -> AudioSegment:
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.88
    pcm = (signal * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


def _set_vst_param(plugin, name: str, value: float) -> None:
    """Silently try to set a DawDreamer VST parameter by name."""
    try:
        plugin.set_parameter(name, float(value))
    except Exception:
        pass


# ─── EFFECTS ──────────────────────────────────────────────────────────────────

def _chorus(audio: AudioSegment, depth_ms: float = 8.0, rate_hz: float = 0.4) -> AudioSegment:
    """Simple chorus via LFO-modulated delay."""
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    sr      = audio.frame_rate
    t       = np.arange(len(samples)) / sr
    lfo     = (np.sin(2 * np.pi * rate_hz * t) * 0.5 + 0.5)
    delays  = (lfo * depth_ms / 1000 * sr).astype(int)
    out     = samples.copy()
    import time
    for i in range(len(samples)):
        d = delays[i]
        if i >= d:
            out[i] += samples[i - d] * 0.4
        if i % 8192 == 0:
            time.sleep(0)
    peak = np.max(np.abs(out))
    if peak > 32767:
        out = out / peak * 32767
    pcm = out.astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=audio.channels)


def _wash(audio: AudioSegment, delay_ms: int = 80, feedback: float = 0.3) -> AudioSegment:
    """Long wash reverb via feedback delay."""
    delay_n = int(audio.frame_rate * delay_ms / 1000) * audio.channels
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    out     = samples.copy()
    import time
    step = 0
    for i in range(delay_n, len(out)):
        out[i] += out[i - delay_n] * feedback
        step += 1
        if step % 8192 == 0:
            time.sleep(0)
    peak = np.max(np.abs(out))
    if peak > 32767:
        out = out / peak * 32767
    pcm = out.astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


# ─── NUMPY RENDERS ────────────────────────────────────────────────────────────

def render_buchla(
    progression:  dict,
    bpm:          int,
    bars:         int,
    synth_params: dict | None = None,
) -> AudioSegment:
    """
    Render Buchla Systems arpeggios (or stabs when arp is off).
    Returns a single AudioSegment -- Buchla track only, no pads.
    """
    sp       = synth_params or {}
    buchla_p = sp.get("buchla", {})

    beat_ms  = 60000 / bpm
    bar_ms   = beat_ms * 4
    total_ms = int(bar_ms * bars)
    output   = AudioSegment.silent(duration=total_ms)

    chords      = progression.get("chords", [])
    arpeggio    = progression.get("arpeggio", {})
    arp_active  = arpeggio.get("active", False)
    arp_pattern = arpeggio.get("pattern", [0, 1, 2, 1])
    arp_speed   = arpeggio.get("speed", "8th")

    arp_step_ms = {
        "quarter": beat_ms,
        "8th":     beat_ms / 2,
        "16th":    beat_ms / 4,
    }.get(arp_speed, beat_ms / 2)

    raw_bcut    = buchla_p.get("filter_cutoff")
    arp_cutoff  = (200.0 + float(raw_bcut) * 3300.0) if raw_bcut is not None else None
    arp_attack  = float(buchla_p.get("attack_s",  0.008))
    arp_decay   = float(buchla_p.get("decay_s",   0.0))
    arp_sustain = float(buchla_p.get("sustain",    1.0))
    arp_release = float(buchla_p.get("release_s",  0.3)) if "release_s" in buchla_p else None
    arp_wave    = float(buchla_p.get("waveshape",  0.0))

    cursor_ms = 0.0
    import time
    for chord_idx, chord in enumerate(chords):
        if chord_idx % 4 == 0:
            time.sleep(0)
        chord_bars = int(chord.get("bars", 1))
        midi_notes = chord.get("midi_notes", [60, 64, 67])
        duration_s = chord_bars * bar_ms / 1000.0
        chord_ms   = chord_bars * bar_ms

        if arp_active and midi_notes and arp_pattern:
            # ── Buchla arpeggio ──────────────────────────────────────────────
            arp_cursor_ms  = cursor_ms
            arp_note_dur_s = (arp_step_ms * 1.6) / 1000

            while arp_cursor_ms < cursor_ms + chord_ms - arp_step_ms:
                pattern_idx = int((arp_cursor_ms - cursor_ms) / arp_step_ms) % len(arp_pattern)
                note_idx    = arp_pattern[pattern_idx] % len(midi_notes)
                midi        = midi_notes[note_idx]

                arp_raw = synth_arp_note(midi, arp_note_dur_s,
                                         attack_s=arp_attack,
                                         decay_s=arp_decay,
                                         sustain=arp_sustain,
                                         release_s=arp_release,
                                         waveshape=arp_wave)
                if arp_cutoff is not None and HAS_SCIPY:
                    arp_raw = _lowpass(arp_raw, arp_cutoff, SAMPLE_RATE)

                arp_seg = _to_segment(arp_raw)
                output  = output.overlay(arp_seg - 3, position=int(arp_cursor_ms))
                arp_cursor_ms += arp_step_ms
        else:
            # ── Buchla stabs (arp off -> short chord hit per chord change) ───
            stab_raw = synth_stab_chord(midi_notes, duration_s,
                                         attack_s=arp_attack,
                                         decay_s=arp_decay,
                                         sustain=arp_sustain,
                                         release_s=arp_release or 0.2,
                                         waveshape=arp_wave)
            if arp_cutoff is not None and HAS_SCIPY:
                stab_raw = _lowpass(stab_raw, arp_cutoff, SAMPLE_RATE)
            stab_seg = _to_segment(stab_raw)
            output   = output.overlay(stab_seg - 3, position=int(cursor_ms))

        cursor_ms += chord_ms

    return normalize(output)


def render_hybrid(
    progression:  dict,
    bpm:          int,
    bars:         int,
    synth_params: dict | None = None,
) -> AudioSegment:
    """
    Render HybridSynth pads with Clouds granular applied.
    Signal chain: synth pads -> chorus/reverb (optional) -> granular
    Returns a single AudioSegment -- hybrid pad track only.
    """
    sp       = synth_params or {}
    hybrid_p = sp.get("hybrid", {})
    cloud_p  = hybrid_p.get("cloud", {})

    beat_ms  = 60000 / bpm
    bar_ms   = beat_ms * 4
    total_ms = int(bar_ms * bars)
    output   = AudioSegment.silent(duration=total_ms)

    chords = progression.get("chords", [])

    raw_cutoff  = hybrid_p.get("filter_cutoff")
    pad_cutoff  = (200.0 + float(raw_cutoff) * 3300.0) if raw_cutoff is not None \
                  else float(progression.get("pad_cutoff", 700))
    pad_attack  = float(hybrid_p.get("attack_s",    0.5))
    pad_decay   = float(hybrid_p.get("decay_s",     0.0))
    pad_sustain = float(hybrid_p.get("sustain",     1.0))
    pad_release = float(hybrid_p.get("release_s",   0.4))
    pad_detune  = int(hybrid_p.get("detune_cents",   4))

    cursor_ms = 0.0
    import time
    for chord_idx, chord in enumerate(chords):
        if chord_idx % 4 == 0:
            time.sleep(0)
        chord_bars = int(chord.get("bars", 1))
        midi_notes = chord.get("midi_notes", [60, 64, 67])
        duration_s = chord_bars * bar_ms / 1000.0
        chord_ms   = chord_bars * bar_ms

        pad_raw = synth_pad_chord(midi_notes, duration_s, pad_cutoff,
                                  attack_s=pad_attack, decay_s=pad_decay,
                                  sustain=pad_sustain, release_s=pad_release,
                                  detune_cents=pad_detune)
        pad_seg = _to_segment(pad_raw)
        output  = output.overlay(pad_seg - 6, position=int(cursor_ms))
        cursor_ms += chord_ms

    # ── Effects: chorus / reverb (HybridSynth: Synth -> Chorus -> Delay -> Reverb -> Granular)
    if "chorus" in hybrid_p:
        if hybrid_p["chorus"]:
            output = _chorus(output)
    if "reverb" in hybrid_p:
        reverb_amt = float(hybrid_p["reverb"])
        if reverb_amt > 0.05:
            output = _wash(output, feedback=0.1 + reverb_amt * 0.45)

    # ── Granular stage ─────────────────────────────────────────────────────────
    if cloud_p and float(cloud_p.get("mix", 0.0)) > 0.005:
        raw = np.frombuffer(output.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        raw = _granulate(
            raw,
            sr       = output.frame_rate,
            position = float(cloud_p.get("position", 0.5)),
            size     = float(cloud_p.get("size",     0.2)),
            density  = float(cloud_p.get("density",  0.4)),
            texture  = float(cloud_p.get("texture",  0.3)),
            spread   = float(cloud_p.get("spread",   0.2)),
            mix      = float(cloud_p.get("mix",      0.4)),
            freeze   = bool(cloud_p.get("freeze",    False)),
        )
        peak = np.max(np.abs(raw))
        if peak > 1e-9:
            raw = raw / peak * 0.88
        pcm    = (raw * 32767).astype(np.int16).tobytes()
        output = AudioSegment(pcm, frame_rate=output.frame_rate,
                              sample_width=2, channels=output.channels)

    return normalize(output)


# ─── VST RENDERS ──────────────────────────────────────────────────────────────

def _render_vst_mono(progression: dict, bpm: int, bars: int,
                     vst_path: str, arp_on: bool = True,
                     vst_params: dict | None = None) -> AudioSegment:
    """
    Core DawDreamer VST renderer.
    arp_on: True -> arpeggios (buchla mode), False -> sustained pads (hybrid mode)
    vst_params: dict of {param_name: value} to set before rendering.
    NOTE: Exact parameter names depend on VST; failures are silently skipped.
    """
    beat_s  = 60.0 / bpm
    bar_s   = beat_s * 4
    total_s = bar_s * bars

    engine = daw.RenderEngine(SAMPLE_RATE, 512)
    synth  = engine.make_plugin_processor("inst", vst_path)

    # Apply VST parameters (best-effort -- silently skip unknown names)
    for param_name, value in (vst_params or {}).items():
        _set_vst_param(synth, param_name, value)

    chords      = progression.get("chords", [])
    arpeggio    = progression.get("arpeggio", {})
    arp_active  = arpeggio.get("active", False) and arp_on
    arp_pattern = arpeggio.get("pattern", [0, 1, 2, 1])
    arp_speed   = arpeggio.get("speed", "8th")

    arp_step_s = {
        "quarter": beat_s,
        "8th":     beat_s / 2,
        "16th":    beat_s / 4,
    }.get(arp_speed, beat_s / 2)

    cursor_s = 0.0
    for chord in chords:
        chord_bars = int(chord.get("bars", 1))
        midi_notes = chord.get("midi_notes", [60, 64, 67])
        dur_s      = chord_bars * bar_s
        chord_end  = cursor_s + dur_s

        if arp_active and midi_notes and arp_pattern:
            arp_cursor = cursor_s
            arp_note_dur = arp_step_s * 1.5
            step_i = 0
            while arp_cursor < chord_end - arp_step_s * 0.5:
                idx  = arp_pattern[step_i % len(arp_pattern)] % len(midi_notes)
                midi = midi_notes[idx]
                note_dur = min(arp_note_dur, chord_end - arp_cursor)
                synth.add_midi_note(midi, 80, arp_cursor, note_dur)
                arp_cursor += arp_step_s
                step_i += 1
        else:
            # Sustained chord (pads or stabs depending on arp_on)
            vel = 75 if not arp_on else 80
            for midi in midi_notes:
                synth.add_midi_note(midi, vel, cursor_s, dur_s * 0.95)

        cursor_s += dur_s

    engine.load_graph([(synth, [])])
    engine.render(total_s)

    audio_np = engine.get_audio()
    mono = (audio_np[0] + audio_np[1]) / 2 if audio_np.ndim == 2 else audio_np
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.88
    pcm = (mono * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


def render_vst_buchla(progression: dict, bpm: int, bars: int,
                      vst_path: str, synth_params: dict | None = None) -> AudioSegment:
    """
    Render Buchla Systems VST -- arpeggios/stabs.
    Maps Music Sheet buchla synth params to likely VST parameter names.
    Exact names depend on Buchla Systems.vst3 ParameterIDs.h -- failures are silently skipped.
    """
    sp        = (synth_params or {}).get("buchla", {})
    waveshape = float(sp.get("waveshape", 0.0))
    fm_depth  = float(sp.get("fm_depth",  0.0))
    fm_index  = float(sp.get("fm_index",  0.0))
    attack_s  = float(sp.get("attack_s",  0.01))
    decay_s   = float(sp.get("decay_s",   0.3))
    sustain   = float(np.clip(sp.get("sustain", 1.0), 0.0, 1.0))
    release_s = float(sp.get("release_s", 0.3))

    # Buchla Systems APVTS param IDs (from ParameterIDs.h).
    # env_1_mode: 0.0 = Transient, 0.5 = Sustained, 1.0 = Cycle (3-choice normalized).
    # Force Transient so envelopes don't self-trigger on MIDI notes.
    _ENV_TRANSIENT = 0.0

    vst_params = {
        # Oscillator (Complex Oscillator + Waveshaper)
        "osc_wavefold_amount":   waveshape,
        "osc_mod_depth_fm":      fm_depth,
        "osc_fm_index":          fm_index / 20.0,
        # LPG1 cutoff/resonance (primary voice filter)
        "lpg_1_cutoff":          float(sp.get("filter_cutoff", 0.65)),
        "lpg_1_resonance":       float(sp.get("resonance", 0.25)),
        # Envelope 1 (amp / LPG vactrol driver)
        "env_1_attack":          attack_s,
        "env_1_decay":           decay_s,
        "env_1_sustain":         sustain,
        "env_1_mode":            _ENV_TRANSIENT,
        # Envelope 2 (modulation / filter sweep)
        "env_2_attack":          attack_s,
        "env_2_decay":           decay_s,
        "env_2_sustain":         sustain,
        "env_2_mode":            _ENV_TRANSIENT,
    }

    return _render_vst_mono(progression, bpm, bars, vst_path,
                             arp_on=True, vst_params=vst_params)


def render_vst_hybrid(progression: dict, bpm: int, bars: int,
                      vst_path: str, synth_params: dict | None = None) -> AudioSegment:
    """
    Render HybridSynth VST -- sustained pads only (arp forced off).
    Applies Clouds granular params to the VST.
    NOTE: VST parameter names are best-effort guesses -- exact names TBD.
    """
    sp      = (synth_params or {}).get("hybrid", {})
    cloud_p = sp.get("cloud", {})

    # HybridSynth APVTS param IDs (from SynthAudioProcessor.cpp).
    # All values normalized 0-1 unless noted.
    vst_params = {
        # Filter
        "FILTER_CUTOFF":      float(sp.get("filter_cutoff", 0.5)),
        "FILTER_RES":         float(sp.get("resonance", 0.1)),
        # Amp envelope (ENV1)
        "ENV1_ATTACK":        float(sp.get("attack_s",  0.5)),
        "ENV1_DECAY":         float(sp.get("decay_s",   0.0)),
        "ENV1_SUSTAIN":       float(sp.get("sustain",   1.0)),
        "ENV1_RELEASE":       float(sp.get("release_s", 0.4)),
        # Filter envelope (ENV2)
        "ENV2_ATTACK":        float(sp.get("attack_s",  0.5)),
        "ENV2_DECAY":         float(sp.get("decay_s",   0.0)),
        "ENV2_SUSTAIN":       float(sp.get("sustain",   1.0)),
        "ENV2_RELEASE":       float(sp.get("release_s", 0.4)),
        "ENV2_AMOUNT":        float(sp.get("filter_env_amount", 0.0)),
        # Chorus / Delay / Reverb
        "CHORUS_RATE":        float(sp.get("chorus_rate",  0.3)),
        "CHORUS_DEPTH":       float(sp.get("chorus_depth", 0.5)),
        "CHORUS_MIX":         float(sp.get("chorus_mix",   0.4)) if sp.get("chorus") else 0.0,
        "REVERB_SIZE":        float(sp.get("reverb", 0.5)),
        "REVERB_DAMPING":     float(sp.get("reverb_damping", 0.5)),
        "REVERB_MIX":         float(sp.get("reverb", 0.0)),
        # Granular (was "cloud_*" -- correct IDs from SynthAudioProcessor.cpp)
        "granular_position":  float(cloud_p.get("position", 0.5)),
        "granular_size":      float(cloud_p.get("size",     0.2)),
        "granular_density":   float(cloud_p.get("density",  0.4)),
        "granular_texture":   float(cloud_p.get("texture",  0.3)),
        "granular_spread":    float(cloud_p.get("spread",   0.2)),
        "granular_mix":       float(cloud_p.get("mix",      0.0)),
        "granular_freeze":    1.0 if cloud_p.get("freeze", False) else 0.0,
    }

    return _render_vst_mono(progression, bpm, bars, vst_path,
                             arp_on=False, vst_params=vst_params)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(
    sheet:        dict,
    output_path:  str = "harmony_output.wav",
    api_key:      str | None = None,
    vst_path:     str | None = None,   # Buchla Systems.vst3
    vst_path_pad: str | None = None,   # HybridSynth.vst3
) -> dict[str, AudioSegment] | None:
    """
    Render Buchla (arps) and HybridSynth (granular pads) separately.

    Returns {"buchla": AudioSegment, "hybrid": AudioSegment}
    or None if harmony is inactive.

    output_path: base path -- buchla exported as <base>_buchla.wav,
                              hybrid exported as <base>_hybrid.wav.
    """
    harmony = sheet["agents"]["harmony"]

    if not harmony.get("active", False):
        print("Harmony inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    instruction = harmony.get("instruction", "")
    mood        = sheet.get("mood", "")
    synth       = harmony.get("synth", {})
    buchla_p    = synth.get("buchla", {})
    hybrid_p    = synth.get("hybrid", {})
    cloud_p     = hybrid_p.get("cloud", {})
    total_ms    = int(bars * 4 * (60000 / bpm))

    print(f"\n── HARMONY AGENT ─────────────────────────────")
    print(f"Key        : {sheet.get('key', '?')}")
    print(f"Duration   : {total_ms}ms  ({bars} bars @ {bpm} bpm)")
    print(f"Texture    : {harmony.get('texture', '')}")
    print(f"Buchla     : wave={buchla_p.get('waveshape', 0):.2f}  "
          f"fm_depth={buchla_p.get('fm_depth', 0):.2f}  "
          f"attack={buchla_p.get('attack_s', 0.01):.3f}s")
    print(f"Hybrid     : attack={hybrid_p.get('attack_s', 0.5):.2f}s  "
          f"reverb={hybrid_p.get('reverb', 0):.2f}  "
          f"chorus={hybrid_p.get('chorus', False)}")
    if cloud_p:
        print(f"Granular   : pos={cloud_p.get('position', 0.5):.2f}  "
              f"size={cloud_p.get('size', 0.2):.2f}  "
              f"density={cloud_p.get('density', 0.4):.2f}  "
              f"mix={cloud_p.get('mix', 0):.2f}  "
              f"freeze={cloud_p.get('freeze', False)}")
    print(f"Generating : progression via LLM...")

    progression = generate_progression(sheet, api_key=api_key)
    chords      = progression.get("chords", [])
    arp         = progression.get("arpeggio", {})
    print(f"Chords     : {' -> '.join(c.get('name','?') for c in chords)}")
    print(f"Arpeggio   : {'on (' + arp.get('speed','?') + ')' if arp.get('active') else 'off'}")

    # ── Render ──────────────────────────────────────────────────────────────────

    if HAS_DAWDREAMER and vst_path and vst_path_pad:
        print(f"Renderer   : DUAL VST")
        print(f"  Buchla   : {os.path.basename(vst_path)}")
        print(f"  Hybrid   : {os.path.basename(vst_path_pad)}")
        buchla_audio = render_vst_buchla(progression, bpm, bars, vst_path, synth)
        hybrid_audio = render_vst_hybrid(progression, bpm, bars, vst_path_pad, synth)
    elif HAS_DAWDREAMER and vst_path:
        print(f"Renderer   : VST (Buchla only) + numpy (Hybrid)")
        buchla_audio = render_vst_buchla(progression, bpm, bars, vst_path, synth)
        hybrid_audio = render_hybrid(progression, bpm, bars, synth_params=synth)
    elif HAS_DAWDREAMER and vst_path_pad:
        print(f"Renderer   : numpy (Buchla) + VST (Hybrid)")
        buchla_audio = render_buchla(progression, bpm, bars, synth_params=synth)
        hybrid_audio = render_vst_hybrid(progression, bpm, bars, vst_path_pad, synth)
    else:
        if (vst_path or vst_path_pad) and not HAS_DAWDREAMER:
            print("  [warn] dawdreamer not installed -- numpy synthesis for both")
        print(f"Renderer   : numpy (Buchla + Hybrid)")
        buchla_audio = render_buchla(progression, bpm, bars, synth_params=synth)
        hybrid_audio = render_hybrid(progression, bpm, bars, synth_params=synth)

    # ── Pad/trim + export ──────────────────────────────────────────────────────

    def _fit(seg: AudioSegment) -> AudioSegment:
        if len(seg) < total_ms:
            return seg + AudioSegment.silent(duration=total_ms - len(seg))
        return seg[:total_ms]

    buchla_audio = _fit(buchla_audio)
    hybrid_audio = _fit(hybrid_audio)

    # Derive output paths from base path
    base      = str(output_path)
    if base.endswith(".wav"):
        buchla_path = base[:-4] + "_buchla.wav"
        hybrid_path = base[:-4] + "_hybrid.wav"
    else:
        buchla_path = base + "_buchla.wav"
        hybrid_path = base + "_hybrid.wav"

    buchla_audio.export(buchla_path, format="wav")
    hybrid_audio.export(hybrid_path, format="wav")

    print(f"Buchla out : {buchla_path}")
    print(f"Hybrid out : {hybrid_path}")
    print(f"──────────────────────────────────────────────\n")

    return {"buchla": buchla_audio, "hybrid": hybrid_audio}


# ─── EXAMPLE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_sheet = {
        "title": "Fracture",
        "bpm": 90,
        "key": "D minor",
        "bars": 8,
        "mood": "dark mental rant, lush, dissociative",
        "structure": "breakdown",
        "timeSignature": "4/4",
        "agents": {
            "harmony": {
                "active": True,
                "texture": "lush pads with slow arpeggios, Dm7 -> Fmaj7",
                "instruction": "fill all space the sampler leaves, warm contrast to dark content",
                "synth": {
                    "buchla": {
                        "attack_s": 0.01,
                        "release_s": 0.3,
                        "waveshape": 0.3,
                        "fm_depth": 0.2,
                        "fm_index": 2.0,
                        "filter_cutoff": 0.6,
                    },
                    "hybrid": {
                        "attack_s": 0.8,
                        "release_s": 1.2,
                        "filter_cutoff": 0.4,
                        "detune_cents": 6,
                        "reverb": 0.5,
                        "chorus": True,
                        "cloud": {
                            "position": 0.5,
                            "size":     0.35,
                            "density":  0.5,
                            "texture":  0.2,
                            "spread":   0.3,
                            "mix":      0.45,
                            "freeze":   False,
                        }
                    }
                }
            }
        }
    }

    key = os.environ.get("ANTHROPIC_API_KEY")
    result = run(example_sheet, output_path="harmony_output.wav", api_key=key)
    if result:
        print(f"Buchla: {len(result['buchla'])/1000:.1f}s")
        print(f"Hybrid: {len(result['hybrid'])/1000:.1f}s")
