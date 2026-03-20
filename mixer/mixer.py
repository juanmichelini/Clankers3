# mixer/mixer.py -- The Clankers 3
#
# mix_section(tracks, sheet)     -> per-track gain + EQ -> blended AudioSegment
# stitch_and_master(sections)    -> crossfade concat -> master compression -> peak norm

import numpy as np
from pydub import AudioSegment

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ─────────────────────────────────────────────────────────────

SR = 44100   # canonical sample rate for all mixing


# ── Per-track mix config ──────────────────────────────────────────────────

# Gain offsets in dB
_GAINS_DB: dict[str, float] = {
    "drums":      +2.0,   # punchy -- kick needs to sit above the pad wash
    "bass_sh101": -2.0,   # Pro-One bass -- sole bass voice
    "buchla":     -1.5,   # Buchla arps -- bright and present, moderate pull
    "hybrid":     -2.5,   # HybridSynth pads -- sit behind, support from underneath
    "sampler":     0.0,
    "voder":      +1.0,
}
_DEFAULT_GAIN_DB = 0.0


# EQ recipe per track.
# Each entry is a list of filter specs:
#   ("low_shelf",  freq_hz, gain_db)
#   ("high_shelf", freq_hz, gain_db)
#   ("low_cut",    freq_hz)           -- 2-pole highpass
#   ("high_cut",   freq_hz)           -- 2-pole lowpass
#   ("peak",       freq_hz, gain_db, Q)
_EQ: dict[str, list[tuple]] = {
    "drums": [
        ("low_shelf",  100,   -3.0),   # clean out sub rumble
        ("high_shelf", 8000,  +2.5),   # air + presence
    ],
    "bass_sh101": [
        ("low_shelf",  80,    +1.5),
        ("high_cut",   5000),
    ],
    "buchla": [
        ("low_cut",    120),           # light mud cut -- arps don't sit in sub
        ("peak",       3000, +2.0, 1.4),  # articulation and presence for arps
        ("high_shelf", 10000, +1.5),   # Buchla shimmer / air
    ],
    "hybrid": [
        ("low_cut",    200),           # clear mud -- granular pads don't need sub
        ("high_shelf", 8000,  +1.0),   # gentle shimmer (granular pads can be dark)
    ],
    "sampler": [
        ("low_cut",    100),           # cut room rumble
        ("peak",       3000, +2.0, 1.4),  # presence / intelligibility
    ],
    "voder": [
        ("low_cut",    120),
        ("peak",       1500, +2.5, 1.2),  # formant clarity
        ("high_shelf", 6000, +1.0),
    ],
}


# ── EQ implementation ─────────────────────────────────────────────────────

def _apply_eq(signal: np.ndarray, recipes: list[tuple]) -> np.ndarray:
    """Apply a list of EQ filter specs to a float64 mono signal."""
    if not HAS_SCIPY or not recipes:
        return signal

    out = signal.copy()
    for spec in recipes:
        kind = spec[0]
        try:
            if kind == "low_cut":
                freq = min(spec[1], SR * 0.45)
                sos  = butter(2, freq, btype="high", fs=SR, output="sos")
                out  = sosfiltfilt(sos, out)

            elif kind == "high_cut":
                freq = min(spec[1], SR * 0.45)
                sos  = butter(2, freq, btype="low", fs=SR, output="sos")
                out  = sosfiltfilt(sos, out)

            elif kind in ("low_shelf", "high_shelf"):
                freq, gain_db = spec[1], spec[2]
                # Implement shelving as band + flat combination:
                #   low_shelf  = flat + (gain-1) * lowpass
                #   high_shelf = flat + (gain-1) * highpass
                g    = 10 ** (gain_db / 20.0)
                freq = min(freq, SR * 0.45)
                if kind == "low_shelf":
                    sos  = butter(2, freq, btype="low",  fs=SR, output="sos")
                    band = sosfiltfilt(sos, out)
                    # shelf: out = (1-g)*band_out + g*out  when gain>0 is boost
                    # equivalently: out = out + (g-1)*band
                    out  = out + (g - 1.0) * band
                else:
                    sos  = butter(2, freq, btype="high", fs=SR, output="sos")
                    band = sosfiltfilt(sos, out)
                    out  = out + (g - 1.0) * band

            elif kind == "peak":
                freq, gain_db, Q = spec[1], spec[2], spec[3]
                # Peaking EQ via bandpass boost/cut
                g      = 10 ** (gain_db / 20.0)
                bw     = freq / Q
                lo     = max(freq - bw / 2, 1.0)
                hi     = min(freq + bw / 2, SR * 0.45)
                sos    = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
                band   = sosfiltfilt(sos, out)
                out    = out + (g - 1.0) * band

        except Exception:
            pass   # bad filter spec -- skip silently

    return out


# ── Normalise segment ─────────────────────────────────────────────────────

def _to_mono_float(seg: AudioSegment) -> np.ndarray:
    """AudioSegment -> mono float64 numpy array normalised to ±1."""
    if seg.channels != 1:
        seg = seg.set_channels(1)
    if seg.frame_rate != SR:
        seg = seg.set_frame_rate(SR)
    if seg.sample_width != 2:
        seg = seg.set_sample_width(2)
    return np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float64) / 32768.0


def _from_float(arr: np.ndarray) -> AudioSegment:
    """Float64 mono numpy array -> AudioSegment."""
    peak = np.max(np.abs(arr))
    if peak > 1e-9:
        arr = arr / peak * 0.92
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SR, sample_width=2, channels=1)


# ── Master compressor ─────────────────────────────────────────────────────

def _compress(
    signal:       np.ndarray,
    threshold_db: float = -18.0,
    ratio:        float = 4.0,
    attack_ms:    float = 10.0,
    release_ms:   float = 100.0,
    knee_db:      float = 6.0,
) -> np.ndarray:
    """
    Feed-forward soft-knee RMS compressor.
    threshold_db: onset of gain reduction (-18 dBFS typical master bus)
    ratio:        gain reduction slope above threshold (4:1)
    attack_ms:    attack time constant
    release_ms:   release time constant
    knee_db:      soft knee width (smooth transition into compression)
    """
    threshold_lin = 10 ** (threshold_db / 20.0)
    knee_low      = 10 ** ((threshold_db - knee_db / 2) / 20.0)
    knee_high     = 10 ** ((threshold_db + knee_db / 2) / 20.0)

    attack_coef  = np.exp(-1.0 / (SR * attack_ms  / 1000.0))
    release_coef = np.exp(-1.0 / (SR * release_ms / 1000.0))

    env    = 0.0
    out    = np.zeros_like(signal)

    for i, sample in enumerate(signal):
        level = abs(sample)

        # Envelope follower
        if level > env:
            env = attack_coef  * env + (1.0 - attack_coef)  * level
        else:
            env = release_coef * env + (1.0 - release_coef) * level

        # Soft-knee gain computation
        if env <= knee_low:
            gain = 1.0
        elif env >= knee_high:
            gain = (threshold_lin / env) ** (1.0 - 1.0 / ratio)
        else:
            # Interpolate smoothly through knee
            t    = (env - knee_low) / (knee_high - knee_low)
            gain = 1.0 - t * (1.0 - (threshold_lin / knee_high) ** (1.0 - 1.0 / ratio))

        out[i] = sample * gain

    return out


# ── Public API ────────────────────────────────────────────────────────────

def mix_section(
    tracks: dict[str, AudioSegment],
    sheet:  dict | None = None,
) -> AudioSegment | None:
    """
    Blend all agent tracks for one section.
      - Normalise to mono 44100 Hz
      - Apply per-track gain + EQ
      - Sum into mix buffer
      - Soft-clip tanh + peak normalise

    sheet is currently unused but reserved for dynamic gain decisions.
    """
    if not tracks:
        return None

    # Normalise all tracks and measure max length
    norm: dict[str, np.ndarray] = {}
    max_len = 0
    for name, seg in tracks.items():
        arr = _to_mono_float(seg)
        norm[name] = arr
        max_len = max(max_len, len(arr))

    mix_buf = np.zeros(max_len, dtype=np.float64)

    for name, arr in norm.items():
        gain_db     = _GAINS_DB.get(name, _DEFAULT_GAIN_DB)
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Pad to mix length
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)))

        # EQ
        eq_recipes = _EQ.get(name, [])
        if eq_recipes:
            arr = _apply_eq(arr, eq_recipes)

        mix_buf += arr * gain_linear
        print(f"  [mix] {name:<12} gain={gain_db:+.1f} dB  peak={np.max(np.abs(arr * gain_linear)):.3f}")

    # Soft-clip + peak normalise
    mix_buf = np.tanh(mix_buf * 0.75) / 0.75
    peak = np.max(np.abs(mix_buf))
    if peak > 1e-9:
        mix_buf = mix_buf / peak * 0.92

    return _from_float(mix_buf)


def _apply_tail_fade(segment: AudioSegment, fade_ms: int = 800) -> AudioSegment:
    """
    Apply a short linear fade-out to the last `fade_ms` of a segment.
    Smooths the tail so crossfades don't blend hard transients at section boundaries.
    """
    if len(segment) <= fade_ms:
        return segment
    head = segment[:-fade_ms]
    tail = segment[-fade_ms:].fade_out(fade_ms)
    return head + tail


def stitch_stem(
    sections:     list[AudioSegment],
    crossfade_ms: int = 2000,
) -> AudioSegment:
    """
    Stitch one stem's sections with the same crossfade timing as stitch_and_master().
    No master compression or normalisation -- leave dynamics intact for DAW mixdown.
    Silent sections are included so all stems stay time-aligned with the full track.
    """
    if not sections:
        return AudioSegment.silent(duration=1000, frame_rate=SR)

    tail_fade_ms = min(crossfade_ms // 2, 1000)
    faded = [_apply_tail_fade(s, tail_fade_ms) for s in sections]

    full = faded[0]
    for s in faded[1:]:
        full = full.append(s, crossfade=crossfade_ms)
    return full


def stitch_and_master(
    sections:      list[AudioSegment],
    crossfade_ms:  int   = 2000,
    target_db:     float = -0.1,
) -> AudioSegment:
    """
    Concatenate sections with crossfade, apply master bus compression,
    and peak-normalise to target_db.
    crossfade_ms=2000 (~half a bar at 90 BPM) gives smooth section blends.
    """
    if not sections:
        raise ValueError("No sections to stitch.")

    # Apply tail fades before crossfading so hard transients don't clash at joins
    tail_fade_ms = min(crossfade_ms // 2, 1000)
    faded = [_apply_tail_fade(s, tail_fade_ms) for s in sections]

    # Concatenate with crossfade
    full = faded[0]
    for s in faded[1:]:
        full = full.append(s, crossfade=crossfade_ms)

    # Master bus: convert to float, compress, normalise
    arr = _to_mono_float(full)
    arr = _compress(arr)

    # Final peak normalise to target
    peak = np.max(np.abs(arr))
    target_lin = 10 ** (target_db / 20.0)
    if peak > 1e-9:
        arr = arr / peak * target_lin

    pcm  = (arr * 32767.0).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=SR, sample_width=2, channels=1)
