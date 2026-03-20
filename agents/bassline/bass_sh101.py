"""
The Clankers 2.0 -- Bass Pro-One Agent (SYNTHESIZER)
Sequential Circuits Pro-One clone bass synthesizer.
Two oscillators (saw + variable pulse width), sub-oscillator, 4-pole filter.
Punchier and more cutting than the 303 -- drives melodic basslines with
controlled aggression. LFO vibrato, portamento/glide, optional chorus.

Music Sheet key: agents.bass_sh101
"""

import os
import re
import json
import requests
import numpy as np
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
        # 4th-order Butterworth (-24 dB/oct) for a much more sine-like tone
        sos = butter(4, min(cutoff_hz, sr * 0.45), btype='low', fs=sr, output='sos')
        return sosfilt(sos, signal).astype(np.float32)
    HAS_SCIPY = True
except ImportError:
    def _lowpass(signal, cutoff_hz, sr):
        # Two-pass box filter to approximate steeper rolloff
        window = max(1, int(sr / (cutoff_hz * 4)))
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        out = np.convolve(signal, kernel, mode='same')
        out = np.convolve(out, kernel, mode='same')
        return out.astype(np.float32)
    HAS_SCIPY = False


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

NOTE_SEMITONES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian":    [0, 1, 3, 5, 6, 8, 10],
}


# ─── KEY PARSING ──────────────────────────────────────────────────────────────

def parse_key(key_str: str) -> tuple[int, list[int]]:
    parts = key_str.strip().split()
    note_str = parts[0] if parts else "C"
    mode_str = parts[1].lower() if len(parts) > 1 else "minor"
    root = NOTE_SEMITONES.get(note_str, 0)
    intervals = SCALES.get(mode_str, SCALES["minor"])
    return root, intervals


def degree_to_midi(degree: int, octave: int, root: int, intervals: list[int]) -> int:
    """Convert scale degree (1-based) to MIDI note number. Octave 2 = bass range (C2 = 36)."""
    idx = (degree - 1) % len(intervals)
    extra_octs = (degree - 1) // len(intervals)
    return 36 + root + intervals[idx] + (octave - 2 + extra_octs) * 12


def degree_to_freq(degree: int, octave: int, root: int, intervals: list[int]) -> float:
    """
    Convert scale degree (1-based) to Hz.
    Octave 2 = bass range (C2 = 65.4 Hz).
    """
    midi = degree_to_midi(degree, octave, root, intervals)
    return 440.0 * (2 ** ((midi - 69) / 12))


# ─── SEQUENCE GENERATION (LLM) ────────────────────────────────────────────────

def _build_harmonic_map_brief(sheet: dict) -> str:
    """
    Render the harmonic_map from the sheet into a compact per-bar root-degree table
    the bass LLM can use to anchor notes to chord changes.
    """
    hmap = sheet.get("harmonic_map", [])
    if not hmap:
        return ""
    lines = ["Bar-by-bar chord roots (anchor bass to these):"]
    for entry in hmap:
        bar  = entry.get("bar", "?")
        deg  = entry.get("root_degree", 1)
        name = entry.get("chord_name", "")
        lines.append(f"  bar {bar}: degree {deg}  ({name})")
    return "\n".join(lines)


def generate_sequence(sheet: dict, api_key: str | None = None) -> list[dict]:
    """
    Ask Claude to generate a melodic note sequence suited for the Pro-One character.
    Returns a list of note dicts.
    """
    bpm           = sheet.get("bpm", 120)
    bars          = sheet.get("bars", 8)
    key           = sheet.get("key", "C minor")
    time_sig      = sheet.get("timeSignature", "4/4")
    mood          = sheet.get("mood", "")
    structure     = sheet.get("structure", "")
    global_notes  = sheet.get("globalNotes", "")
    tension       = float(sheet.get("tension", 0.4))
    agent         = sheet["agents"]["bass_sh101"]
    pattern       = agent.get("pattern", "")
    instruction   = agent.get("instruction", "")
    beats_per_bar = int(time_sig.split("/")[0]) if "/" in time_sig else 4
    total_beats   = bars * beats_per_bar

    harmonic_brief = _build_harmonic_map_brief(sheet)

    # Scale rhythmic complexity hint from tension
    if tension >= 0.7:
        density_hint = "High tension section: use shorter durations (0.25-0.5 beats), more notes, push rhythmic energy."
    elif tension >= 0.4:
        density_hint = "Medium tension: mix short and medium durations (0.5-2 beats), keep forward motion."
    else:
        density_hint = "Low tension section: longer, more sustained notes (1-4 beats), space and breath."

    prompt = f"""You are a Pro-One bass synthesizer programmer. Generate a melodic bass line.

Key: {key}
BPM: {bpm}
Bars: {bars}
Time signature: {time_sig}
Total beats: {total_beats}
Section: {structure}
Mood: {mood}
Tension: {tension:.2f} (0=minimal, 1=peak density)
Global notes: {global_notes}
Pattern: {pattern}
Instruction: {instruction}

{harmonic_brief}

{density_hint}

The Pro-One (Sequential Circuits) is a two-oscillator monosynth -- punchy and cutting,
thicker than the 303 but still very musical. Let the mood, section, and pattern
drive your choices: a bridge needs contrast, a climax needs density, an outro dissolves.

Return a JSON object with a "sequence" array. Each element has:
- "degree": scale degree 1-7 (1 = root note)
- "octave": 1 (sub-bass), 2 (bass), or 3 (upper bass) -- vary this for interest
- "beats": duration in beats -- use 0.25, 0.5, 1, 2, or 4; mix values for rhythmic life
- "accent": boolean -- slightly louder, more presence
- "slide": boolean -- glide smoothly from previous note (legato)
- "rest": boolean -- silence
- "vibrato": boolean -- apply LFO pitch wobble on this note

VOICE LEADING RULES (critical for musical bass lines):
- Anchor the FIRST note of each bar to the chord root degree shown in the harmonic map above
- Prefer stepwise motion (scale degrees ±1-2) or small leaps (±3-4 semitones) between adjacent notes
- Avoid leaping more than a 5th (7 semitones) except for intentional dramatic effect
- Use slide=true on notes that are close in pitch -- it makes leaps sound smooth and intentional
- Resolve tension by returning to the root degree at phrase endings (every 2-4 bars)
- Total beats must sum to exactly {total_beats}

Return ONLY valid JSON. No explanation, no markdown."""

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
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        text = resp.json()["content"][0]["text"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        data = json.loads(text)
        return data.get("sequence", [])
    except Exception as e:
        print(f"  [llm error] {e}")
        return _fallback_sequence(total_beats)


def _smooth_voice_leading(notes: list[dict], root: int, intervals: list[int]) -> list[dict]:
    """
    Post-process LLM sequence to constrain large leaps.
    Jumps > 7 semitones (a 5th) are pulled back by inverting the octave of the destination
    note when possible (octave 1-3 bounds respected).
    Only adjusts pitch, never changes degree/rhythm/slide/accent.
    """
    prev_midi: int | None = None
    for note in notes:
        if note.get("rest", False):
            prev_midi = None
            continue
        degree = int(note.get("degree", 1))
        octave = int(note.get("octave", 2))
        midi   = degree_to_midi(degree, octave, root, intervals)

        if prev_midi is not None:
            jump = midi - prev_midi
            if jump > 7 and octave > 1:          # leaping too high -- step down an octave
                candidate = degree_to_midi(degree, octave - 1, root, intervals)
                if abs(candidate - prev_midi) < abs(jump):
                    note["octave"] = octave - 1
                    midi = candidate
            elif jump < -7 and octave < 3:        # leaping too low -- step up an octave
                candidate = degree_to_midi(degree, octave + 1, root, intervals)
                if abs(candidate - prev_midi) < abs(jump):
                    note["octave"] = octave + 1
                    midi = candidate

        prev_midi = midi
    return notes


def _fallback_sequence(total_beats: int) -> list[dict]:
    notes = []
    remaining = float(total_beats)
    while remaining > 0:
        b = min(2.0, remaining)
        notes.append({"degree": 1, "octave": 2, "beats": b,
                       "accent": False, "slide": False, "rest": False, "vibrato": False})
        remaining -= b
    return notes


# ─── SYNTHESIS ────────────────────────────────────────────────────────────────

def _square(freq_start: float, freq_end: float, n: int, pulse_width: float = 0.5) -> np.ndarray:
    """
    Phase-accumulating square/pulse wave with frequency interpolation.
    pulse_width=0.5 -> square; <0.5 -> narrower pulse, brighter tone.
    """
    freqs = np.linspace(freq_start, freq_end, n)
    phase = np.cumsum(freqs / SAMPLE_RATE) % 1.0
    return np.where(phase < pulse_width, 1.0, -1.0).astype(np.float32)


def _sub_osc(freq_start: float, freq_end: float, n: int) -> np.ndarray:
    """Square wave one octave below (half frequency). Adds low-end body."""
    return _square(freq_start / 2.0, freq_end / 2.0, n)


def _apply_vibrato(freq_start: float, freq_end: float, n: int,
                   rate_hz: float = 5.5, depth_cents: float = 18.0) -> np.ndarray:
    """
    Generate frequency array with LFO vibrato applied.
    Vibrato fades in after the first ~80ms (delayed onset, more natural).
    """
    base = np.linspace(freq_start, freq_end, n)
    t = np.arange(n) / SAMPLE_RATE
    lfo = np.sin(2 * np.pi * rate_hz * t)
    # Fade in over ~80ms so the onset is clean
    onset = min(n, int(0.08 * SAMPLE_RATE))
    fade = np.ones(n)
    fade[:onset] = np.linspace(0.0, 1.0, onset)
    mod = 2 ** ((lfo * depth_cents * fade) / 1200.0)
    return (base * mod).astype(np.float32)


def _amp_envelope(n: int, accent: bool,
                  decay_param: float | None = None,
                  release_param: float | None = None,
                  sustain_param: float | None = None) -> np.ndarray:
    """
    Slower decay than 303 -- warmer, more sustained feel.
    decay_param  : 0.0 (very sustained/pad) -> 1.0 (percussive stab); None = default
    release_param: 0.0 (hard cutoff) -> 1.0 (long fade-out tail); None = no release shaping
    sustain_param: 0.0 (no floor) -> 1.0 (hold at full amplitude); clips exp decay floor
    """
    t = np.arange(n) / SAMPLE_RATE

    # Map 0.0-1.0 -> 0.3 (very slow) … 8.0 (very fast)
    if decay_param is not None:
        base_decay = 0.3 + float(decay_param) * 7.7
    else:
        base_decay = 1.8 if accent else 3.0

    env = np.exp(-t * base_decay)
    attack = min(440, n)  # ~10ms
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    if accent:
        env = np.minimum(env * 1.3, 1.0)

    # Sustain floor: prevent decay from dropping below sustain_param after attack
    if sustain_param is not None and sustain_param > 0.0 and n > attack:
        floor = float(np.clip(sustain_param, 0.0, 1.0))
        env[attack:] = np.maximum(env[attack:], floor)

    # Release tail: linear fade over last release_param * 50% of note
    if release_param is not None and release_param > 0.0 and n > 1:
        rel_samples = max(1, min(int(float(release_param) * 0.5 * n), n))
        env[-rel_samples:] *= np.linspace(1.0, 0.0, rel_samples)

    return env.astype(np.float32)


def synth_note(
    freq_start: float,
    freq_end: float,
    duration_s: float,
    accent: bool,
    vibrato: bool,
    cutoff_hz: float,
    sub_mix: float = 0.45,
    pulse_width: float = 0.5,
    decay_param: float | None = None,
    release_param: float | None = None,
    sustain_param: float | None = None,
) -> np.ndarray:
    """
    Synthesize one Pro-One-style note.
    Square oscillator + sub-oscillator -> LP filter -> amplitude envelope.
    """
    n = max(1, int(SAMPLE_RATE * duration_s))

    if vibrato:
        freqs = _apply_vibrato(freq_start, freq_end, n)
        phase = np.cumsum(freqs / SAMPLE_RATE) % 1.0
        osc = np.where(phase < pulse_width, 1.0, -1.0).astype(np.float32)
        sub = _sub_osc(freq_start, freq_end, n)  # sub doesn't vibrate
    else:
        osc = _square(freq_start, freq_end, n, pulse_width)
        sub = _sub_osc(freq_start, freq_end, n)

    raw = osc + sub * sub_mix
    raw /= (1.0 + sub_mix)  # normalise mix level

    note_cutoff = min(cutoff_hz * (1.4 if accent else 1.0), SAMPLE_RATE * 0.45)
    raw = _lowpass(raw, note_cutoff, SAMPLE_RATE)
    raw *= _amp_envelope(n, accent, decay_param=decay_param, release_param=release_param,
                         sustain_param=sustain_param)
    return raw


def render_sequence(
    notes: list[dict],
    bpm: int,
    root: int,
    intervals: list[int],
    cutoff_hz: float,
    pulse_width: float,
    decay_param: float | None = None,
    release_param: float | None = None,
    sustain_param: float | None = None,
    swing: float = 0.0,
    humanize: bool = False,
) -> AudioSegment:
    """
    Synthesize all notes onto a timeline.
    swing    : 0.0-0.5  -- push every other beat-subdivision forward (triplet feel)
    humanize : bool     -- add per-note ±10ms onset jitter and ±3% velocity variation
    """
    import time as _time
    import random

    beat_s   = 60.0 / bpm
    beat_smp = beat_s * SAMPLE_RATE

    # Compute total duration from note beats (exact)
    total_beats = sum(float(n.get("beats", 1)) for n in notes)
    total_n     = max(1, int(total_beats * beat_smp))
    timeline    = np.zeros(total_n, dtype=np.float32)

    cursor_smp  = 0.0      # exact float cursor in samples
    beat_counter = 0.0     # running beat count for swing alignment
    prev_freq    = None

    for idx, note in enumerate(notes):
        if idx > 0 and idx % 12 == 0:
            _time.sleep(0)   # yield GIL

        dur_beats  = float(note.get("beats", 1))
        dur_smp    = dur_beats * beat_smp
        n          = max(1, int(dur_smp))

        # ── Timing: swing + humanize ──────────────────────────────────────
        onset = cursor_smp

        # Swing: if this note starts on an off-beat 8th (beat_counter is a half-beat away),
        # push it forward. We check if the nearest 0.5-beat grid position is "odd".
        if swing > 0.0:
            half_beat_idx = round(beat_counter * 2)  # which half-beat we're on
            if half_beat_idx % 2 == 1:               # off-beat
                onset += beat_smp * 0.5 * swing

        if humanize:
            jitter_smp = random.uniform(-0.010, 0.010) * SAMPLE_RATE  # ±10 ms
            onset     += jitter_smp

        onset_i = max(0, min(int(onset), total_n - 1))

        if note.get("rest", False):
            prev_freq     = None
            cursor_smp   += dur_smp
            beat_counter += dur_beats
            continue

        degree  = int(note.get("degree", 1))
        octave  = int(note.get("octave", 2))
        accent  = bool(note.get("accent", False))
        slide   = bool(note.get("slide", False))
        vibrato = bool(note.get("vibrato", False))
        freq    = degree_to_freq(degree, octave, root, intervals)

        # Velocity humanize: ±5% amplitude
        vel_mult = (1.0 + random.uniform(-0.05, 0.05)) if humanize else 1.0

        freq_start = prev_freq if (slide and prev_freq is not None) else freq
        chunk = synth_note(freq_start, freq, n / SAMPLE_RATE, accent, vibrato,
                           cutoff_hz, pulse_width=pulse_width,
                           decay_param=decay_param, release_param=release_param,
                           sustain_param=sustain_param)
        chunk *= vel_mult

        end_i = min(onset_i + len(chunk), total_n)
        timeline[onset_i:end_i] += chunk[:end_i - onset_i]

        prev_freq     = freq
        cursor_smp   += dur_smp
        beat_counter += dur_beats

    peak = np.max(np.abs(timeline))
    if peak > 0:
        timeline = timeline / peak * 0.85
    pcm = (timeline * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


# ─── VST RENDER ───────────────────────────────────────────────────────────────

def _set_bass_vst_params(synth, synth_params: dict) -> None:
    """
    Push sheet synth params into bassYnth Pro-One APVTS.
    bassYnth Pro-One APVTS parameter IDs (from PluginProcessor.cpp):
      filterCutoff, filterRes, filterEnvAmt, filterAttack, filterDecay,
      filterSustain, filterRelease, ampAttack, ampDecay, ampSustain, ampRelease,
      lfoRate, lfoAmount, lfoShape, lfoToFilter, lfoToPitch, glide,
      oscAOctave, subOscLevel, oscALevel, oscBLevel, oscBTune, oscBPW, noiseLevel
    All are normalized 0-1 in the APVTS unless otherwise noted.
    """
    def _set(name, value):
        try:
            synth.set_parameter(name, float(value))
        except Exception:
            pass

    # Filter
    raw_cutoff = synth_params.get("filter_cutoff")
    if raw_cutoff is not None:
        _set("filterCutoff", float(raw_cutoff))
    _set("filterRes",    synth_params.get("resonance",       0.33))
    _set("filterEnvAmt", synth_params.get("filter_env_amt",  0.5))
    # Filter envelope
    _set("filterAttack",  synth_params.get("filter_attack",  0.01))
    _set("filterDecay",   synth_params.get("filter_decay",   0.3))
    _set("filterSustain", synth_params.get("filter_sustain", 0.0))
    _set("filterRelease", synth_params.get("filter_release", 0.1))
    # Amp envelope
    _set("ampAttack",  synth_params.get("attack",   0.005))
    _set("ampDecay",   synth_params.get("decay",    0.5) if synth_params.get("decay") is not None else 0.5)
    _set("ampSustain", synth_params.get("sustain",  0.0) if synth_params.get("sustain") is not None else 0.0)
    _set("ampRelease", synth_params.get("release",  0.1) if synth_params.get("release") is not None else 0.1)
    # LFO
    _set("lfoRate",   synth_params.get("lfo_rate",   0.3))
    _set("lfoAmount", synth_params.get("lfo_amount", 0.0))
    # Oscillator / sub
    _set("subOscLevel", synth_params.get("sub_level", 0.45))
    pw = synth_params.get("pulse_width")
    if pw is not None:
        _set("oscBPW", float(pw))
    glide = synth_params.get("glide", 0.0)
    _set("glide", float(glide))


def render_sequence_vst(
    notes: list[dict],
    bpm: int,
    root: int,
    intervals: list[int],
    vst_path: str,
    total_ms: int,
    synth_params: dict | None = None,
) -> AudioSegment:
    """Render the note sequence through a VST3 instrument via dawdreamer."""
    engine   = daw.RenderEngine(SAMPLE_RATE, 512)
    synth    = engine.make_plugin_processor("bass_sh101", vst_path)

    # Push sheet synth params to the VST's APVTS before rendering
    if synth_params:
        _set_bass_vst_params(synth, synth_params)

    beat_s   = 60.0 / bpm
    cursor_s = 0.0

    for note in notes:
        dur_s = float(note.get("beats", 1)) * beat_s
        if not note.get("rest", False):
            midi = degree_to_midi(int(note.get("degree", 1)), int(note.get("octave", 2)),
                                  root, intervals)
            vel  = 95 if note.get("accent") else 70
            synth.add_midi_note(midi, vel, cursor_s, dur_s * 0.95)  # longer gate for legato
        cursor_s += dur_s

    engine.load_graph([(synth, [])])
    engine.render(total_ms / 1000.0)

    audio_np = engine.get_audio()
    mono = (audio_np[0] + audio_np[1]) / 2 if audio_np.ndim == 2 else audio_np
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.85
    pcm = (mono * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


# ─── EFFECTS ──────────────────────────────────────────────────────────────────

def _chorus(audio: AudioSegment, depth_ms: float = 8.0, rate_hz: float = 0.5) -> AudioSegment:
    """
    Simple chorus: mix original with a slowly-modulated delay copy.
    Adds width and warmth characteristic of the Pro-One with chorus.
    """
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    t = np.arange(len(samples)) / audio.frame_rate
    # Modulate delay depth sinusoidally
    mod = (np.sin(2 * np.pi * rate_hz * t) * 0.5 + 0.5)  # 0..1
    max_delay = int(depth_ms * audio.frame_rate / 1000)
    out = samples.copy()
    import time
    step = 0
    for i in range(max_delay, len(samples)):
        d = int(mod[i] * max_delay)
        out[i] = samples[i] * 0.7 + samples[i - d] * 0.5
        step += 1
        if step % 8192 == 0:
            time.sleep(0)  # yield GIL so GUI stays responsive
    peak = np.max(np.abs(out))
    if peak > 32767:
        out = out / peak * 32767
    pcm = out.astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


def _overdrive(audio: AudioSegment, amount: float = 2.5) -> AudioSegment:
    """Soft tanh overdrive -- gentler than the 303's distortion."""
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    samples = np.tanh(samples * amount) / np.tanh(np.array(amount))
    pcm = (samples * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


def apply_effects(audio: AudioSegment, instruction: str) -> AudioSegment:
    instr = instruction.lower()
    if any(w in instr for w in ["chorus", "wide", "lush", "warm", "fat"]):
        audio = _chorus(audio)
    if any(w in instr for w in ["overdrive", "drive", "grit", "dirty", "crunch"]):
        audio = _overdrive(audio)
    return normalize(audio)


# ─── VOICE SETTINGS FROM CONTEXT ──────────────────────────────────────────────

def pick_cutoff(instruction: str, mood: str) -> float:
    """Pick LPF cutoff. Lower values = more sine-like (fewer harmonics pass through)."""
    combined = (instruction + " " + mood).lower()
    if any(w in combined for w in ["sine", "sinwave", "sin wave", "sine wave"]):
        return 320.0   # near-pure sine: kills almost all harmonics
    if any(w in combined for w in ["bright", "open", "sharp", "cutting"]):
        return 2200.0
    elif any(w in combined for w in ["dark", "heavy", "deep", "sub", "mud"]):
        return 500.0
    elif any(w in combined for w in ["warm", "soft", "gentle", "dreamy", "smooth", "mellow"]):
        return 420.0   # warm/dreamy -> very sine-like
    return 900.0  # neutral default (reduced from 1400 for cleaner tone)


def pick_pulse_width(instruction: str) -> float:
    """Square (0.5) for full, narrower pulse (<0.5) for brighter/thinner tone."""
    instr = instruction.lower()
    if any(w in instr for w in ["thin", "bright", "nasal", "reedy"]):
        return 0.3
    elif any(w in instr for w in ["hollow", "woody", "mid"]):
        return 0.25
    return 0.5  # square wave default


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(sheet: dict, output_path: str = "bass_sh101_output.wav",
        api_key: str | None = None, vst_path: str | None = None) -> AudioSegment | None:
    agent = sheet["agents"]["bass_sh101"]

    if not agent.get("active", False):
        print("Bass Pro-One inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    key_str     = sheet.get("key", "C minor")
    instruction = agent.get("instruction", "")
    mood        = sheet.get("mood", "")
    synth       = agent.get("synth", {})

    root, intervals = parse_key(key_str)

    # Synth params take priority over keyword heuristics
    raw_cutoff    = synth.get("filter_cutoff")
    cutoff_hz     = (300.0 + float(raw_cutoff) * 3700.0) if raw_cutoff is not None \
                    else pick_cutoff(instruction, mood)
    pulse_width   = float(synth.get("pulse_width", pick_pulse_width(instruction)))
    sub_level     = float(synth.get("sub_level", 0.45))
    raw_decay     = synth.get("decay")
    decay_param   = float(raw_decay) if raw_decay is not None else None
    raw_release   = synth.get("release")
    release_param = float(raw_release) if raw_release is not None else None
    raw_sustain   = synth.get("sustain")
    sustain_param = float(raw_sustain) if raw_sustain is not None else None
    total_ms      = int(bars * 4 * (60000 / bpm))

    print(f"\n── BASS PRO-ONE AGENT ───────────────")
    print(f"Key        : {key_str}")
    print(f"Duration   : {total_ms}ms  ({bars} bars @ {bpm} bpm)")
    print(f"Filter     : {cutoff_hz:.0f} Hz  ({'scipy' if HAS_SCIPY else 'box approx'})")
    print(f"Pulse/Sub  : pw={pulse_width:.2f}  sub={sub_level:.2f}")
    print(f"Envelope   : decay={decay_param if decay_param is not None else 'auto'}  sustain={sustain_param if sustain_param is not None else 'none'}  release={release_param if release_param is not None else 'none'}")
    print(f"Chorus     : {synth.get('chorus', False)}  depth={synth.get('chorus_depth_ms', 8.0):.1f}ms  rate={synth.get('chorus_rate_hz', 0.5):.2f}Hz")
    print(f"Generating : sequence via LLM...")

    notes = generate_sequence(sheet, api_key=api_key)

    # Voice leading post-processor: smooth large leaps in the LLM sequence
    notes = _smooth_voice_leading(notes, root, intervals)

    total_beats = sum(float(n.get("beats", 1)) for n in notes)
    print(f"Sequence   : {len(notes)} notes / {total_beats} beats")

    # Microtiming: swing + humanize
    swing_amt = float(agent.get("swing", 0.0))
    humanize  = bool(agent.get("humanize", False))
    if swing_amt == 0.0:
        # Fall back to mood/instruction heuristic (same as drums pick_swing)
        combined = (instruction + " " + mood).lower()
        if any(w in combined for w in ["jazz", "bebop", "swing hard", "big band"]):
            swing_amt = 0.35
        elif any(w in combined for w in ["swing", "shuffle", "groove", "funk", "latin"]):
            swing_amt = 0.22
        elif any(w in combined for w in ["loose", "human", "warm", "laid back"]):
            swing_amt = 0.08
    print(f"Microtiming: swing={swing_amt:.2f}  humanize={humanize}")

    if vst_path and HAS_DAWDREAMER:
        print(f"Renderer   : VST  ({os.path.basename(vst_path)})")
        audio = render_sequence_vst(notes, bpm, root, intervals, vst_path, total_ms,
                                    synth_params=synth)
    else:
        if vst_path and not HAS_DAWDREAMER:
            print("  [warn] dawdreamer not installed -- falling back to numpy synthesis")
        audio = render_sequence(notes, bpm, root, intervals, cutoff_hz, pulse_width,
                                decay_param=decay_param, release_param=release_param,
                                sustain_param=sustain_param,
                                swing=swing_amt, humanize=humanize)

    # Numeric chorus params override keyword detection
    if synth.get("chorus", False):
        audio = _chorus(audio,
                        depth_ms=float(synth.get("chorus_depth_ms", 8.0)),
                        rate_hz=float(synth.get("chorus_rate_hz", 0.5)))
        audio = apply_effects(audio, "")   # skip keyword chorus re-application
    else:
        audio = apply_effects(audio, instruction)

    if len(audio) < total_ms:
        audio = audio + AudioSegment.silent(duration=total_ms - len(audio))
    else:
        audio = audio[:total_ms]

    audio.export(output_path, format="wav")
    print(f"Output     : {output_path}")
    print(f"────────────────────────────────────\n")
    return audio


# ─── EXAMPLE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_sheet = {
        "title": "Fracture",
        "bpm": 90,
        "key": "D minor",
        "bars": 16,
        "mood": "dark mental rant, lush, dissociative",
        "structure": "breakdown",
        "timeSignature": "4/4",
        "agents": {
            "bass_sh101": {
                "active": True,
                "pattern": "slow melodic movement, two notes per bar, legato phrasing",
                "instruction": "warm and lush, chorus, sustained"
            }
        }
    }

    key = os.environ.get("ANTHROPIC_API_KEY")
    run(example_sheet, output_path="bass_sh101_output.wav", api_key=key)
