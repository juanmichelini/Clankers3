"""
The Clankers 2.0 — Drums Agent
Electronic percussion synthesizer.
Reads bpm, bars, pattern, and instruction from the Music Sheet,
asks Claude to generate a 16-step sequence, synthesizes audio.

Voices: kick · snare · hihat_closed · hihat_open · clap

Music Sheet key: agents.drums
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
    def _highpass(signal, cutoff_hz, sr):
        sos = butter(2, min(cutoff_hz, sr * 0.45), btype='high', fs=sr, output='sos')
        return sosfilt(sos, signal).astype(np.float32)
    HAS_SCIPY = True
except ImportError:
    def _lowpass(signal, cutoff_hz, sr):
        window = max(1, int(sr / (cutoff_hz * 4)))
        if window <= 1:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same').astype(np.float32)
    def _highpass(signal, cutoff_hz, sr):
        return signal  # no-op fallback
    HAS_SCIPY = False


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE       = 44100
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
VOICES            = ["kick", "snare", "hihat_closed", "hihat_open", "clap"]

# GM-standard MIDI note numbers for each drum voice.
# If your Antigravity Drums VST uses a custom map, adjust these values.
DRUM_MIDI = {
    "kick":         36,   # Bass Drum 1
    "snare":        38,   # Acoustic Snare
    "hihat_closed": 42,   # Closed Hi Hat
    "hihat_open":   46,   # Open Hi Hat
    "clap":         39,   # Hand Clap
}


# ─── SEQUENCE GENERATION (LLM) ────────────────────────────────────────────────

def generate_sequence(sheet: dict, api_key: str | None = None) -> dict:
    """
    Ask Claude to generate a 16-step drum pattern.
    Returns dict of { voice: [step, ...] } where each step is 0.0–1.0 velocity (0 = silent).
    """
    drums        = sheet["agents"]["drums"]
    bpm          = sheet.get("bpm", 120)
    bars         = sheet.get("bars", 8)
    time_sig     = sheet.get("timeSignature", "4/4")
    pattern      = drums.get("pattern", "")
    instruction  = drums.get("instruction", "")
    mood         = sheet.get("mood", "")
    structure    = sheet.get("structure", "")
    global_notes = sheet.get("globalNotes", "")

    prompt = f"""You are an electronic drum machine sequencer. Generate a 16-step drum pattern.

BPM: {bpm}
Bars: {bars}
Time signature: {time_sig}
Section: {structure}
Mood: {mood}
Global notes: {global_notes}
Pattern description: {pattern}
Instruction: {instruction}

Return a JSON object with a "pattern" key containing one array per drum voice.
Each array has exactly 16 values: 0 = silent, or a float 0.1–1.0 = velocity (louder = higher).

Voices: kick, snare, hihat_closed, hihat_open, clap

Rules:
- The pattern must fit the genre, mood, and section — do NOT default to four-on-the-floor
- Funky/jazz/groove patterns: syncopated kick, ghost snares, busy hihats with velocity variation
- Minimal/breakdown sections: sparse, lots of zeros, tension through absence
- Climax/peak sections: dense, hard accents, open hats, claps
- Use velocity variation heavily — ghost notes (0.1-0.3), medium hits (0.4-0.7), accents (0.8-1.0)
- hihat_open on upbeats and syncopated positions adds swing feel
- Clap layers on backbeat snare for emphasis
- The pattern loops for all {bars} bars

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
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        text = resp.json()["content"][0]["text"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        data = json.loads(text)
        return data.get("pattern", {})
    except Exception as e:
        print(f"  [llm error] {e}")
        return _fallback_pattern()


def _fallback_pattern() -> dict:
    """Basic four-on-the-floor fallback."""
    return {
        "kick":         [1.0, 0, 0, 0,  1.0, 0, 0, 0,  1.0, 0, 0, 0,  1.0, 0, 0, 0],
        "snare":        [0,   0, 0, 0,  0.9, 0, 0, 0,  0,   0, 0, 0,  0.9, 0, 0, 0],
        "hihat_closed": [0.6, 0, 0.6, 0, 0.6, 0, 0.6, 0, 0.6, 0, 0.6, 0, 0.6, 0, 0.6, 0],
        "hihat_open":   [0] * 16,
        "clap":         [0] * 16,
    }


# ─── SYNTHESIS ────────────────────────────────────────────────────────────────

def _to_segment(signal: np.ndarray) -> AudioSegment:
    """Convert float32 numpy array (–1..1) to mono AudioSegment."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.9
    pcm = (signal * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


def synth_kick(velocity: float = 1.0,
               pitch_hz: float = 40.0,
               punch: float = 0.5,
               decay_s: float = 0.3) -> np.ndarray:
    """Sine wave with pitch drop + noise click transient.
    pitch_hz: floor pitch of the kick body (40-90 Hz).
    punch:    transient click level (0=none, 1=loud click).
    decay_s:  amp envelope decay in seconds.
    """
    n   = int(SAMPLE_RATE * max(decay_s * 2, 0.2))
    t   = np.arange(n) / SAMPLE_RATE

    # Pitch envelope: (pitch_hz + 110) → pitch_hz over 80ms
    freq  = pitch_hz + 110.0 * np.exp(-t / 0.08)
    phase = np.cumsum(freq / SAMPLE_RATE) * 2 * np.pi
    body  = np.sin(phase)

    amp = np.exp(-t / max(decay_s, 0.05))
    ramp = min(220, n)
    amp[:ramp] *= np.linspace(0, 1, ramp)

    click_n = int(SAMPLE_RATE * 0.003)
    transient = np.zeros(n)
    if click_n > 0:
        transient[:click_n] = np.random.randn(click_n) * (0.8 * punch * 2)

    signal = (body + transient) * amp * velocity
    return signal.astype(np.float32)


def synth_snare(velocity: float = 1.0,
                tuning_hz: float = 200.0,
                snappy: float = 0.7) -> np.ndarray:
    """Noise burst + short tone body.
    tuning_hz: tone body fundamental (150-300 Hz).
    snappy:    noise-to-tone ratio (0=all tone, 1=all noise/wire).
    """
    n = int(SAMPLE_RATE * 0.25)
    t = np.arange(n) / SAMPLE_RATE

    noise     = np.random.randn(n)
    noise_env = np.exp(-t / 0.07)

    tone     = np.sin(2 * np.pi * tuning_hz * t)
    tone_env = np.exp(-t / 0.03)

    noise_level = 0.4 + snappy * 0.6
    tone_level  = 0.8 * (1.0 - snappy * 0.5)
    signal = noise * noise_env * noise_level + tone * tone_env * tone_level
    signal *= velocity

    if HAS_SCIPY:
        signal = _highpass(signal, 300, SAMPLE_RATE)
    return signal.astype(np.float32)


def synth_hihat_closed(velocity: float = 1.0,
                       decay_ms: float = 12.0,
                       highpass_hz: float = 6000.0) -> np.ndarray:
    """Short filtered noise burst."""
    highpass_hz = max(highpass_hz, 100.0)   # guard against 0/negative from evolved sheets
    n   = max(1, int(SAMPLE_RATE * decay_ms / 1000 * 4))
    t   = np.arange(n) / SAMPLE_RATE
    sig = np.random.randn(n) * np.exp(-t / max(decay_ms / 1000, 0.001)) * velocity
    if HAS_SCIPY:
        sig = _highpass(sig, min(highpass_hz, SAMPLE_RATE * 0.45), SAMPLE_RATE)
    return sig.astype(np.float32)


def synth_hihat_open(velocity: float = 1.0,
                     decay_ms: float = 120.0,
                     highpass_hz: float = 6000.0) -> np.ndarray:
    """Longer filtered noise — open hi-hat."""
    highpass_hz = max(highpass_hz, 100.0)   # guard against 0/negative from evolved sheets
    n   = max(1, int(SAMPLE_RATE * decay_ms / 1000 * 3))
    t   = np.arange(n) / SAMPLE_RATE
    sig = np.random.randn(n) * np.exp(-t / max(decay_ms / 1000, 0.001)) * velocity
    if HAS_SCIPY:
        sig = _highpass(sig, min(highpass_hz, SAMPLE_RATE * 0.45), SAMPLE_RATE)
    return sig.astype(np.float32)


def synth_clap(velocity: float = 1.0) -> np.ndarray:
    """Stacked noise bursts — classic clap layering."""
    n      = int(SAMPLE_RATE * 0.2) # 200ms
    signal = np.zeros(n)
    t_full = np.arange(n) / SAMPLE_RATE

    # Three staggered bursts
    for offset_ms in [0, 8, 16]:
        off = int(offset_ms / 1000 * SAMPLE_RATE)
        burst_n = int(SAMPLE_RATE * 0.010)  # 10ms
        if off + burst_n <= n:
            tb  = np.arange(burst_n) / SAMPLE_RATE
            env = np.exp(-tb / 0.006)
            signal[off:off + burst_n] += np.random.randn(burst_n) * env

    signal *= np.exp(-t_full / 0.08) * velocity
    if HAS_SCIPY:
        signal = _highpass(signal, 800, SAMPLE_RATE)
    return signal.astype(np.float32)


SYNTH_FNS = {
    "kick":         synth_kick,
    "snare":        synth_snare,
    "hihat_closed": synth_hihat_closed,
    "hihat_open":   synth_hihat_open,
    "clap":         synth_clap,
}


# ─── RENDER ───────────────────────────────────────────────────────────────────

def render_pattern(
    pattern: dict,
    bpm: int,
    bars: int,
    swing: float = 0.0,
    synth_params: dict | None = None,
) -> AudioSegment:
    """
    Render the step sequence to audio.
    Pattern loops for `bars` bars. Each 16-step pattern = 1 bar (16 sixteenth notes).
    synth_params: the sheet's agents.drums.synth dict for per-voice sound shaping.
    """
    sp         = synth_params or {}
    kick_p     = sp.get("kick",  {})
    snare_p    = sp.get("snare", {})
    hihat_p    = sp.get("hihat", {})

    step_ms    = (60000 / bpm) / 4          # one 16th note in ms
    bar_ms     = step_ms * 16
    total_ms   = int(bar_ms * bars)
    output     = AudioSegment.silent(duration=total_ms)

    import time
    for voice, steps in pattern.items():
        synth_fn = SYNTH_FNS.get(voice)
        if synth_fn is None:
            continue

        steps = steps[:16]  # enforce 16 steps

        for bar in range(bars):
            if bar > 0 and bar % 2 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive
            for step_idx, velocity in enumerate(steps):
                if not velocity:
                    continue
                velocity = float(velocity)

                # Base position
                position_ms = bar * bar_ms + step_idx * step_ms

                # Swing: push odd 16th steps forward
                if swing > 0 and step_idx % 2 == 1:
                    position_ms += step_ms * swing * 0.5

                position_ms = max(0, min(position_ms, total_ms - 1))

                # Synthesize + convert (pass per-voice synth params if available)
                if voice == "kick":
                    raw = synth_fn(velocity,
                                   pitch_hz=float(kick_p.get("pitch_hz", 40.0)),
                                   punch=float(kick_p.get("punch", 0.5)),
                                   decay_s=float(kick_p.get("decay_s", 0.3)))
                elif voice == "snare":
                    raw = synth_fn(velocity,
                                   tuning_hz=float(snare_p.get("tuning_hz", 200.0)),
                                   snappy=float(snare_p.get("snappy", 0.7)))
                elif voice in ("hihat_closed", "hihat_open"):
                    raw = synth_fn(velocity,
                                   decay_ms=float(hihat_p.get("decay_ms",
                                                               12.0 if voice == "hihat_closed" else 120.0)),
                                   highpass_hz=float(hihat_p.get("highpass_hz", 6000.0)))
                else:
                    raw = synth_fn(velocity)
                chunk = _to_segment(raw)

                output = output.overlay(chunk, position=int(position_ms))

    return output


# ─── VST RENDER ───────────────────────────────────────────────────────────────

def render_pattern_vst(
    pattern: dict,
    bpm: int,
    bars: int,
    swing: float,
    vst_path: str,
) -> AudioSegment:
    """Render the drum pattern through a VST3 instrument via dawdreamer."""
    step_s  = (60.0 / bpm) / 4   # one 16th note in seconds
    bar_s   = step_s * 16
    total_s = bar_s * bars

    engine = daw.RenderEngine(SAMPLE_RATE, 512)
    drums  = engine.make_plugin_processor("drums", vst_path)

    for voice, steps in pattern.items():
        midi = DRUM_MIDI.get(voice)
        if midi is None:
            continue
        steps = steps[:16]
        for bar in range(bars):
            for step_idx, velocity in enumerate(steps):
                if not velocity:
                    continue
                vel_int = max(1, min(127, int(float(velocity) * 127)))
                t = bar * bar_s + step_idx * step_s
                if swing > 0 and step_idx % 2 == 1:
                    t += step_s * swing * 0.5
                t = max(0.0, min(t, total_s - 0.01))
                drums.add_midi_note(midi, vel_int, t, 0.05)  # 50ms trigger pulse

    engine.load_graph([(drums, [])])
    engine.render(total_s)

    audio_np = engine.get_audio()
    mono = (audio_np[0] + audio_np[1]) / 2 if audio_np.ndim == 2 else audio_np
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.9
    pcm = (mono * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)


# ─── EFFECTS ──────────────────────────────────────────────────────────────────

def _distort(audio: AudioSegment, amount: float = 3.0) -> AudioSegment:
    """Tanh waveshaping distortion on the drum bus."""
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    samples = np.tanh(samples * amount) / np.tanh(np.array(amount, dtype=np.float32))
    pcm = (samples * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=audio.frame_rate,
                        sample_width=2, channels=audio.channels)


def _room(audio: AudioSegment, delay_ms: int = 25, feedback: float = 0.2) -> AudioSegment:
    """Short room reverb via feedback delay."""
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
    # NOTE: distortion is NOT applied here — it is already handled explicitly via
    # synth.distortion in run(). Applying it again from instruction keywords causes
    # double-distortion. Room reverb is also covered by synth.room; this is a final
    # normalize-only pass for the numpy path.
    return normalize(audio)


# ─── SWING / MOOD ─────────────────────────────────────────────────────────────

def pick_swing(instruction: str, mood: str) -> float:
    combined = (instruction + " " + mood).lower()
    if any(w in combined for w in ["jazz", "bebop", "swing hard", "big band"]):
        return 0.38   # strong jazz triplet swing
    if any(w in combined for w in ["swing", "shuffle", "groove", "funk", "funky",
                                    "latin", "bossa", "samba", "afro", "soul"]):
        return 0.25
    if any(w in combined for w in ["loose", "human", "warm", "laid back", "relaxed"]):
        return 0.1
    return 0.0


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(sheet: dict, output_path: str = "drums_output.wav",
        api_key: str | None = None, vst_path: str | None = None) -> AudioSegment | None:
    drums = sheet["agents"]["drums"]

    if not drums.get("active", False):
        print("Drums inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    instruction = drums.get("instruction", "")
    mood        = sheet.get("mood", "")
    synth       = drums.get("synth", {})
    total_ms    = int(bars * 4 * (60000 / bpm))
    swing       = pick_swing(instruction, mood)

    kick_p  = synth.get("kick",  {})
    snare_p = synth.get("snare", {})
    hihat_p = synth.get("hihat", {})

    print(f"\n── DRUMS AGENT ──────────────────────")
    print(f"BPM        : {bpm}")
    print(f"Duration   : {total_ms}ms  ({bars} bars)")
    print(f"Swing      : {swing:.2f}  ({'scipy' if HAS_SCIPY else 'no filters'})")
    print(f"Kick       : pitch={kick_p.get('pitch_hz', 40):.0f}Hz  punch={kick_p.get('punch', 0.5):.2f}  decay={kick_p.get('decay_s', 0.3):.2f}s")
    print(f"Snare      : tuning={snare_p.get('tuning_hz', 200):.0f}Hz  snappy={snare_p.get('snappy', 0.7):.2f}")
    print(f"Hihat      : decay={hihat_p.get('decay_ms', 12):.0f}ms  hp={hihat_p.get('highpass_hz', 6000):.0f}Hz")
    print(f"Bus FX     : room={synth.get('room', False)}  room_size={synth.get('room_size', 0):.2f}  dist={synth.get('distortion', 0):.2f}")
    print(f"Generating : pattern via LLM...")

    pattern = generate_sequence(sheet, api_key=api_key)
    active_voices = [v for v, steps in pattern.items() if any(steps)]
    print(f"Voices     : {', '.join(active_voices)}")

    if vst_path and HAS_DAWDREAMER:
        print(f"Renderer   : VST  ({os.path.basename(vst_path)})")
        audio = render_pattern_vst(pattern, bpm, bars, swing, vst_path)
        audio = apply_effects(audio, instruction)
        audio = audio + 6
    else:
        if vst_path and not HAS_DAWDREAMER:
            print("  [warn] dawdreamer not installed — falling back to numpy synthesis")
        audio = render_pattern(pattern, bpm, bars, swing=swing, synth_params=synth)

        # Apply numeric bus FX from synth params
        dist = synth.get("distortion", 0.0)
        if dist > 0.05:
            audio = _distort(audio, amount=1.0 + dist * 5.0)
        if synth.get("room", False):
            room_size = float(synth.get("room_size", 0.2))
            audio = _room(audio, feedback=0.1 + room_size * 0.35)

        audio = apply_effects(audio, instruction)

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
            "drums": {
                "active": True,
                "pattern": "minimal four-on-floor, ghost hits on snare",
                "instruction": "stay back in the mix, mechanical and cold",
            }
        }
    }

    key = os.environ.get("ANTHROPIC_API_KEY")
    run(example_sheet, output_path="drums_output.wav", api_key=key)
