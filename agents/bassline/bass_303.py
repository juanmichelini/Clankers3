"""
The Clankers 2.0 -- Bass 303 Agent (SYNTHESIZER)
Roland TB-303-style acid bass synthesizer.
Reads key, bpm, bars, and pattern from the Music Sheet,
asks Claude to generate a note sequence, synthesizes audio.

Music Sheet key: agents.bass303
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
        sos = butter(2, min(cutoff_hz, sr * 0.45), btype='low', fs=sr, output='sos')
        return sosfilt(sos, signal).astype(np.float32)
    HAS_SCIPY = True
except ImportError:
    def _lowpass(signal, cutoff_hz, sr):
        # Box filter approximation -- crude but scipy-free
        window = max(1, int(sr / (cutoff_hz * 4)))
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same').astype(np.float32)
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
    """
    Parse "D minor", "C# dorian", "F major", etc.
    Returns (root_semitone, scale_intervals).
    """
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
    Convert scale degree (1-based) to frequency in Hz.
    Octave 2 = bass range (C2 = 65.4 Hz). Octave 3 = upper bass (~130 Hz).
    """
    midi = degree_to_midi(degree, octave, root, intervals)
    return 440.0 * (2 ** ((midi - 69) / 12))


# ─── SEQUENCE GENERATION (LLM) ────────────────────────────────────────────────

def generate_sequence(sheet: dict, api_key: str | None = None) -> list[dict]:
    """
    Ask Claude to generate a note sequence from the pattern description.
    Returns a list of note dicts.
    """
    bpm           = sheet.get("bpm", 120)
    bars          = sheet.get("bars", 8)
    key           = sheet.get("key", "C minor")
    time_sig      = sheet.get("timeSignature", "4/4")
    mood          = sheet.get("mood", "")
    structure     = sheet.get("structure", "")
    global_notes  = sheet.get("globalNotes", "")
    bassline      = sheet["agents"]["bass303"]
    pattern       = bassline.get("pattern", "")
    instruction   = bassline.get("instruction", "")
    beats_per_bar = int(time_sig.split("/")[0]) if "/" in time_sig else 4
    total_beats   = bars * beats_per_bar

    prompt = f"""You are a 303 acid bassline sequencer. Generate a note sequence.

Key: {key}
BPM: {bpm}
Bars: {bars}
Time signature: {time_sig}
Total beats: {total_beats}
Section: {structure}
Mood: {mood}
Global notes: {global_notes}
Pattern: {pattern}
Instruction: {instruction}

Return a JSON object with a "sequence" array. Each element has:
- "degree": scale degree 1-7 (1 = root note)
- "octave": 1 (very low sub-bass), 2 (bass, default), or 3 (upper bass)
- "beats": duration in beats -- use 0.25, 0.5, 1, 2, or 4
- "accent": boolean -- louder, more open filter
- "slide": boolean -- glide pitch from previous note
- "rest": boolean -- silence (ignore degree/octave if true)

Rules:
- Total beats must sum to exactly {total_beats}
- Let the mood, section, and pattern drive rhythm and density -- don't always default to even 16ths
- Vary octave across the sequence for melodic interest
- Use accents to create rhythmic shape (roughly 1 in 4 notes)
- Slides connect melodically adjacent notes naturally
- Rests add space and tension -- use them musically

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


def _fallback_sequence(total_beats: int) -> list[dict]:
    """Fallback: root notes on every beat."""
    notes = []
    remaining = float(total_beats)
    while remaining > 0:
        b = min(1.0, remaining)
        notes.append({"degree": 1, "octave": 2, "beats": b,
                       "accent": False, "slide": False, "rest": False})
        remaining -= b
    return notes


# ─── SYNTHESIS ────────────────────────────────────────────────────────────────

def _sawtooth(freq_start: float, freq_end: float, n: int) -> np.ndarray:
    """Phase-accumulating sawtooth. Frequency interpolates from start to end (slide)."""
    freqs = np.linspace(freq_start, freq_end, n)
    phase = np.cumsum(freqs / SAMPLE_RATE) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def _amp_envelope(n: int, accent: bool) -> np.ndarray:
    """Exponential decay envelope. Accent = slower decay + louder."""
    t = np.arange(n) / SAMPLE_RATE
    decay = 2.5 if accent else 4.5
    env = np.exp(-t * decay)
    attack = min(220, n)  # ~5ms attack ramp
    env[:attack] *= np.linspace(0, 1, attack)
    if accent:
        env = np.minimum(env * 1.4, 1.0)
    return env.astype(np.float32)


def synth_note(
    freq_start: float,
    freq_end: float,
    duration_s: float,
    accent: bool,
    cutoff_hz: float,
) -> np.ndarray:
    """Synthesize one 303-style note: sawtooth -> filter -> envelope."""
    n = max(1, int(SAMPLE_RATE * duration_s))
    raw = _sawtooth(freq_start, freq_end, n)
    note_cutoff = min(cutoff_hz * (1.6 if accent else 1.0), SAMPLE_RATE * 0.45)
    raw = _lowpass(raw, note_cutoff, SAMPLE_RATE)
    raw *= _amp_envelope(n, accent)
    return raw


def render_sequence(
    notes: list[dict],
    bpm: int,
    root: int,
    intervals: list[int],
    cutoff_hz: float,
) -> AudioSegment:
    """Synthesize all notes and concatenate into one AudioSegment."""
    import time
    beat_s = 60.0 / bpm
    chunks = []
    prev_freq = None

    for idx, note in enumerate(notes):
        if idx > 0 and idx % 12 == 0:
            time.sleep(0)  # yield GIL so GUI stays responsive
        duration_s = float(note.get("beats", 1)) * beat_s
        n = max(1, int(SAMPLE_RATE * duration_s))

        if note.get("rest", False):
            chunks.append(np.zeros(n, dtype=np.float32))
            prev_freq = None
            continue

        degree = int(note.get("degree", 1))
        octave = int(note.get("octave", 2))
        accent = bool(note.get("accent", False))
        slide  = bool(note.get("slide", False))
        freq   = degree_to_freq(degree, octave, root, intervals)

        freq_start = prev_freq if (slide and prev_freq is not None) else freq
        chunk = synth_note(freq_start, freq, duration_s, accent, cutoff_hz)
        chunks.append(chunk)
        prev_freq = freq

    if not chunks:
        return AudioSegment.silent(duration=1000)

    combined = np.concatenate(chunks)
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = combined / peak * 0.85
    pcm = (combined * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


# ─── VST RENDER ───────────────────────────────────────────────────────────────

def _apply_vst_params(synth, synth_params: dict) -> None:
    """Set VST parameters from sheet (filter_cutoff, resonance, env_mod 0..1). No-op if API missing."""
    if not synth_params:
        return
    try:
        # dawdreamer: get_parameters_description() or get_parameter_name(i); set_parameter(index, value) 0..1
        get_desc = getattr(synth, "get_parameters_description", None)
        get_name = getattr(synth, "get_parameter_name", None)
        set_param = getattr(synth, "set_parameter", None)
        if not set_param:
            return
        # Map param name substrings to sheet keys. MonoBass 303 (Ableton_Mapping_guide): Filter Cutoff, Filter Resonance, Filter Env Amount.
        # Also 303-style CC names (CC 74 = cutoff, CC 71 = resonance).
        want = {
            "filter cutoff": "filter_cutoff", "cutoff": "filter_cutoff",
            "cc 74": "filter_cutoff", "cc74": "filter_cutoff", "74": "filter_cutoff",
            "filter resonance": "resonance", "resonance": "resonance",
            "cc 71": "resonance", "cc71": "resonance", "71": "resonance",
            "filter env amount": "env_mod", "filter env": "env_mod", "env amount": "env_mod",
            "env mod": "env_mod", "env_mod": "env_mod", "envelope mod": "env_mod",
        }
        if get_desc:
            desc = get_desc()
            if isinstance(desc, list):
                for p in desc:
                    name = (p.get("name") or p.get("label") or "").lower().strip()
                    idx = p.get("index", p.get("parameterIndex"))
                    if idx is None:
                        continue
                    for key, param_key in want.items():
                        if key in name and param_key in synth_params:
                            val = float(synth_params[param_key])
                            set_param(int(idx), max(0.0, min(1.0, val)))
                            break
            return
        if get_name:
            for i in range(256):
                try:
                    name = (get_name(i) or "").lower()
                except Exception:
                    break
                for key, param_key in want.items():
                    if key in name and param_key in synth_params:
                        val = float(synth_params[param_key])
                        set_param(i, max(0.0, min(1.0, val)))
                        break
    except Exception:
        pass


def render_sequence_vst(
    notes: list[dict],
    bpm: int,
    root: int,
    intervals: list[int],
    vst_path: str,
    total_ms: int,
    synth_params: dict | None = None,
) -> AudioSegment:
    """Render the note sequence through a VST3 instrument via dawdreamer.
    synth_params: optional dict with filter_cutoff, resonance, env_mod (0..1) to set plugin state.
    """
    engine   = daw.RenderEngine(SAMPLE_RATE, 512)
    synth    = engine.make_plugin_processor("bass303", vst_path)
    _apply_vst_params(synth, synth_params or {})
    beat_s   = 60.0 / bpm
    cursor_s = 0.0

    for note in notes:
        dur_s = float(note.get("beats", 1)) * beat_s
        if not note.get("rest", False):
            midi  = degree_to_midi(int(note.get("degree", 1)), int(note.get("octave", 2)),
                                   root, intervals)
            vel   = 100 if note.get("accent") else 70
            synth.add_midi_note(midi, vel, cursor_s, dur_s * 0.9)
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

def _distort(audio: AudioSegment, amount: float = 4.0) -> AudioSegment:
    """Tanh waveshaping distortion."""
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    samples = np.tanh(samples * amount) / np.tanh(np.array(amount))
    pcm = (samples * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


def _delay(audio: AudioSegment, delay_ms: int = 375, feedback: float = 0.35) -> AudioSegment:
    """Single-tap delay with feedback."""
    import time
    delay_samples = int(audio.frame_rate * delay_ms / 1000) * audio.channels
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
    out = samples.copy()
    step = 0
    for i in range(delay_samples, len(out)):
        out[i] += samples[i - delay_samples] * feedback
        step += 1
        if step % 8192 == 0:
            time.sleep(0)  # yield GIL so GUI stays responsive
    peak = np.max(np.abs(out))
    if peak > 32767:
        out = out / peak * 32767
    pcm = out.astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


def apply_effects(audio: AudioSegment, instruction: str) -> AudioSegment:
    instr = instruction.lower()
    if any(w in instr for w in ["distort", "overdrive", "drive", "grit", "dirty", "crunch"]):
        audio = _distort(audio)
    if any(w in instr for w in ["delay", "echo", "repeat"]):
        audio = _delay(audio)
    return normalize(audio)


# ─── CUTOFF SELECTION ─────────────────────────────────────────────────────────

def pick_cutoff(instruction: str, mood: str) -> float:
    """Choose LP filter cutoff from context. Lower = darker. Higher = more acid."""
    combined = (instruction + " " + mood).lower()
    if any(w in combined for w in ["acid", "bright", "open", "squelch", "harsh", "sharp"]):
        return 2000.0
    elif any(w in combined for w in ["dark", "heavy", "deep", "mud", "sub", "thick"]):
        return 400.0
    return 800.0


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def _synth_cutoff_from_params(synth: dict, instruction: str, mood: str) -> float:
    """Resolve filter cutoff Hz from synth params (preferred) or keyword fallback."""
    raw = synth.get("filter_cutoff")
    if raw is not None:
        # 0.0 -> 200 Hz (dark), 1.0 -> 4000 Hz (acid bright)
        return 200.0 + float(raw) * 3800.0
    return pick_cutoff(instruction, mood)


def run(sheet: dict, output_path: str = "bass303_output.wav",
        api_key: str | None = None, vst_path: str | None = None) -> AudioSegment | None:
    bassline = sheet["agents"]["bass303"]

    if not bassline.get("active", False):
        print("Bassline inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    key_str     = sheet.get("key", "C minor")
    instruction = bassline.get("instruction", "")
    mood        = sheet.get("mood", "")
    synth       = bassline.get("synth", {})

    root, intervals = parse_key(key_str)
    cutoff_hz = _synth_cutoff_from_params(synth, instruction, mood)
    total_ms  = int(bars * 4 * (60000 / bpm))

    print(f"\n── BASSLINE 303 AGENT ───────────────")
    print(f"Key        : {key_str}")
    print(f"Duration   : {total_ms}ms  ({bars} bars @ {bpm} bpm)")
    print(f"Filter     : {cutoff_hz:.0f} Hz  resonance={synth.get('resonance', 0):.2f}  env_mod={synth.get('env_mod', 0):.2f}")
    print(f"Distortion : {synth.get('distortion', 0):.2f}  delay={synth.get('delay', False)}")
    print(f"Generating : sequence via LLM...")

    notes = generate_sequence(sheet, api_key=api_key)
    total_beats = sum(float(n.get("beats", 1)) for n in notes)
    print(f"Sequence   : {len(notes)} notes / {total_beats} beats")

    if vst_path and HAS_DAWDREAMER:
        print(f"Renderer   : VST  ({os.path.basename(vst_path)})")
        # Pass synth params so VST gets known filter/resonance state (no CCs sent otherwise)
        cutoff_norm = (cutoff_hz - 200.0) / 3800.0 if cutoff_hz else 0.5
        synth_params = {
            "filter_cutoff": max(0.0, min(1.0, cutoff_norm)),
            "resonance": max(0.0, min(1.0, float(synth.get("resonance", 0)))),
            "env_mod": max(0.0, min(1.0, float(synth.get("env_mod", 0)))),
        }
        audio = render_sequence_vst(notes, bpm, root, intervals, vst_path, total_ms, synth_params)
    else:
        if vst_path and not HAS_DAWDREAMER:
            print("  [warn] dawdreamer not installed -- falling back to numpy synthesis")
        audio = render_sequence(notes, bpm, root, intervals, cutoff_hz)

    # Apply distortion from synth params if set, otherwise fall through to keyword FX
    dist = synth.get("distortion", 0.0)
    if dist > 0.05:
        audio = _distort(audio, amount=1.0 + dist * 8.0)

    # Apply delay from synth params if set
    if synth.get("delay", False):
        fb = float(synth.get("delay_feedback", 0.35))
        audio = _delay(audio, feedback=fb)

    # Keyword-driven FX for anything not covered by numeric params
    audio = apply_effects(audio, instruction)

    # Pad or trim to exact duration
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
            "bass303": {
                "active": True,
                "pattern": "slow, syncopated, one note per bar with slides",
                "instruction": "heavy and deliberate, no rush"
            }
        }
    }

    key = os.environ.get("ANTHROPIC_API_KEY")
    run(example_sheet, output_path="bass303_output.wav", api_key=key)
