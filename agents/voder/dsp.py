"""
Low-level DSP primitives for the Clanker Voice Voder synthesizer.

v2 tuning changes:
  - BiquadBPFBank expanded to 5 formants (F1-F5)
  - Added OnePoleSmooth for parameter smoothing (anti-zipper)
  - GlottalPulseOsc improved with aspiration noise blend
  - Added pitch jitter + amplitude shimmer support

Provides:
  - SawtoothOsc     : Band-limited-ish sawtooth oscillator (voiced source)
  - GlottalPulseOsc : Rosenberg glottal pulse with optional aspiration
  - WhiteNoise      : White noise generator (unvoiced source)
  - OnePoleSmooth   : One-pole smoother for parameter changes
  - BiquadBPF       : Second-order bandpass filter for formant modelling
  - BiquadBPFBank   : Parallel bank of 5 bandpass filters (F1-F5)
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_freq(note: float) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by *t* in [0, 1]."""
    return a + (b - a) * t


def cosine_interp(a: float, b: float, t: float) -> float:
    """Cosine interpolation (smoother than linear) between *a* and *b*."""
    ct = 0.5 * (1.0 - math.cos(t * math.pi))
    return a + (b - a) * ct


# ---------------------------------------------------------------------------
# One-pole smoother (anti-zipper for parameter changes)
# ---------------------------------------------------------------------------

class OnePoleSmooth:
    """
    Simple one-pole low-pass smoother for control-rate parameters.

    Prevents audible zipper noise when filter frequencies change abruptly.
    """

    def __init__(self, smooth_ms: float = 5.0, sample_rate: int = 44100):
        self.value = 0.0
        tau = smooth_ms / 1000.0
        self.coeff = 1.0 - math.exp(-1.0 / (tau * sample_rate)) if tau > 0 else 1.0

    def process(self, target: float) -> float:
        self.value += self.coeff * (target - self.value)
        return self.value

    def set_immediate(self, value: float):
        self.value = value


# ---------------------------------------------------------------------------
# One-pole lowpass (for softening fricatives and plosive bursts)
# ---------------------------------------------------------------------------

class OnePoleLPF:
    """
    One-pole lowpass filter. Used to roll off harsh highs from
    unvoiced fricatives and voiceless plosive bursts.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.y = 0.0
        self._coeff = 0.0  # 0 = full smooth, 1 = pass through

    def set_cutoff(self, freq_hz: float):
        """-3 dB cutoff frequency in Hz."""
        freq_hz = max(20.0, min(freq_hz, self.sr * 0.49))
        # Leaky integrator: y += (x - y) * g,  g = 1 - exp(-2*pi*fc/sr)
        self._coeff = 1.0 - math.exp(-2.0 * math.pi * freq_hz / self.sr)

    def process(self, x: float) -> float:
        self.y += (x - self.y) * self._coeff
        return self.y

    def reset(self):
        self.y = 0.0


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

class SawtoothOsc:
    """Naive sawtooth oscillator with polyBLEP anti-aliasing."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.phase = 0.0

    def next_sample(self, freq: float) -> float:
        dt = freq / self.sr
        # Naive sawtooth: phase ramps 0 -> 1, output maps to -1 -> +1
        out = 2.0 * self.phase - 1.0
        # PolyBLEP correction at discontinuity
        out -= self._poly_blep(self.phase, dt)
        self.phase += dt
        if self.phase >= 1.0:
            self.phase -= 1.0
        return out

    @staticmethod
    def _poly_blep(phase: float, dt: float) -> float:
        if phase < dt:
            t = phase / dt
            return t + t - t * t - 1.0
        elif phase > 1.0 - dt:
            t = (phase - 1.0 + dt) / dt
            return t * t + t + t + 1.0
        return 0.0

    def reset(self):
        self.phase = 0.0


class GlottalPulseOsc:
    """
    Rosenberg-model glottal pulse oscillator with aspiration noise.

    Produces a more voice-like waveform than a raw sawtooth.
    The opening phase (0 .. t_open) is a rising half-sine,
    the closing phase (t_open .. 1) is a falling quarter-sine.

    v2: Adds controllable aspiration noise blend for breathiness.
    Real voices always have some turbulent airflow noise mixed in.
    """

    def __init__(self, sample_rate: int = 44100, open_ratio: float = 0.6):
        self.sr = sample_rate
        self.phase = 0.0
        self.open_ratio = open_ratio
        self.aspiration = 0.005  # Reduced from 0.03 for cleaner tone
        self._rng = np.random.default_rng(42)

    def next_sample(self, freq: float) -> float:
        dt = freq / self.sr
        p = self.phase
        oq = self.open_ratio

        if p < oq:
            # Opening phase: half-sine rise
            out = math.sin(math.pi * p / oq)
        else:
            # Closing phase: cosine fall
            t_close = (p - oq) / (1.0 - oq)
            out = math.cos(0.5 * math.pi * t_close)

        # Blend in aspiration noise (turbulent airflow)
        if self.aspiration > 0:
            noise = self._rng.uniform(-1.0, 1.0)
            out = out * (1.0 - self.aspiration) + noise * self.aspiration

        self.phase += dt
        if self.phase >= 1.0:
            self.phase -= 1.0
        return out

    def reset(self):
        self.phase = 0.0


class WhiteNoise:
    """White noise generator using numpy's random state."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def next_sample(self) -> float:
        return self.rng.uniform(-1.0, 1.0)

    def generate(self, n_samples: int) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=n_samples)


# ---------------------------------------------------------------------------
# Biquad Bandpass Filter
# ---------------------------------------------------------------------------

class BiquadBPF:
    """
    Second-order IIR bandpass filter (constant-skirt-gain design).

    Designed for real-time, per-sample processing with dynamic
    center frequency and bandwidth changes.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        # State variables (direct-form II transposed)
        self.z1 = 0.0
        self.z2 = 0.0
        # Coefficients
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.gain = 1.0
        # Cache to avoid redundant recomputation
        self._last_freq = -1.0
        self._last_bw = -1.0

    def set_params(self, center_freq: float, bandwidth: float, gain: float = 1.0):
        """
        Recalculate coefficients for the given center frequency and bandwidth.

        Uses the cookbook BPF formula (Audio EQ Cookbook, Robert Bristow-Johnson).
        Skips recomputation if parameters haven't changed meaningfully.
        """
        self.gain = gain

        # Skip recomputation if params haven't changed much
        if (abs(center_freq - self._last_freq) < 0.5
                and abs(bandwidth - self._last_bw) < 0.5):
            return

        self._last_freq = center_freq
        self._last_bw = bandwidth

        # Clamp frequency to valid range
        center_freq = max(20.0, min(center_freq, self.sr * 0.45))
        bandwidth = max(10.0, bandwidth)

        w0 = 2.0 * math.pi * center_freq / self.sr
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)

        Q = center_freq / bandwidth
        alpha = sin_w0 / (2.0 * Q)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize
        inv_a0 = 1.0 / a0
        self.b0 = b0 * inv_a0
        self.b1 = b1 * inv_a0
        self.b2 = b2 * inv_a0
        self.a1 = a1 * inv_a0
        self.a2 = a2 * inv_a0

    def process(self, x: float) -> float:
        """Process a single sample through the filter."""
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y * self.gain

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self._last_freq = -1.0
        self._last_bw = -1.0


# ---------------------------------------------------------------------------
# 5-Formant Filter Bank
# ---------------------------------------------------------------------------

NUM_FORMANTS = 5

class BiquadBPFBank:
    """
    Parallel bank of 5 biquad bandpass filters representing F1-F5.

    v2: Expanded from 3 to 5 formants. F4/F5 add vocal "body" and
    "presence" that make the output sound less thin.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.filters = [BiquadBPF(sample_rate) for _ in range(NUM_FORMANTS)]

    def set_formants(
        self,
        freqs: tuple[float, ...],
        bandwidths: tuple[float, ...],
        gains: tuple[float, ...],
    ):
        """Set formant frequencies, bandwidths and gains (up to 5)."""
        for i, filt in enumerate(self.filters):
            if i < len(freqs):
                filt.set_params(freqs[i], bandwidths[i], gains[i])

    def process(self, x: float) -> float:
        """Run a single sample through all filters and sum."""
        return sum(f.process(x) for f in self.filters)

    def reset(self):
        for f in self.filters:
            f.reset()
