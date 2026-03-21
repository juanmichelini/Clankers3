"""
The Clankers 2.0 -- Voder Agent (v2 -- 5-Formant Rosenberg Engine)

Formant speech synthesizer rebuilt using the ClankerVoice DSP stack:
  - GlottalPulseOsc  : Rosenberg glottal pulse model with aspiration noise
  - BiquadBPFBank    : 5-formant bank (F1-F5) -- full vocal presence and body
  - Per-sample synthesis with cosine formant interpolation (CONTROL_RATE=8)
  - Context-dependent coarticulation (V->V 90ms, C->V 35ms, V->C 60ms, etc.)
  - Proper plosive rendering with VOT (voice onset time) aspiration tail
  - 'h' phoneme inherits formants from the following vowel
  - Vibrato (5.2 Hz) + pitch jitter + amplitude shimmer
  - Voiced-fricative blending (z, v = noise + glottal tone)
  - Full phoneme inventory: 10 vowels + 6 plosives + 7 fricatives + 3 nasals
                            + 2 liquids + 2 glides = 30 phonemes total

The LLM generates a word timeline (phoneme list + MIDI note + duration in beats).
The sequencer distributes time within each word (consonants fixed, vowels share rest).
The VoderEngine renders a single contiguous phrase with smooth coarticulation.

Music Sheet key: agents.voder
{
  "active":        bool,
  "instruction":   string  (e.g. "dark, descending, mechanical, eerie"),
  "phoneme_hints": list    (informal hints, e.g. ["oh mah", "sss ee n"]),
  "fundamental_hz": float  (optional base pitch; 80-200; default 130)
}
"""

from __future__ import annotations

import math
import os
import time
import json
import re
import random
import sys
from pathlib import Path

import numpy as np
import requests
from pydub import AudioSegment
from pydub.effects import normalize

# ── Import DSP primitives and phoneme table from sibling files ────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dsp import (
    NUM_FORMANTS,
    BiquadBPF,
    BiquadBPFBank,
    GlottalPulseOsc,
    OnePoleLPF,
    SawtoothOsc,
    WhiteNoise,
    cosine_interp,
    midi_to_freq,
)
from phoneme_table import (
    PHONEME_TABLE,
    VOWELS,
    default_duration_ms,
    is_vowel,
    phoneme_type,
)


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE       = 44100
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Filter coefficient update interval (every N samples, not every sample)
CONTROL_RATE = 8

VALID_PHONEMES = sorted(PHONEME_TABLE.keys())
VALID_VOWELS   = sorted(VOWELS.keys())


# ─── PHONEME EVENT ────────────────────────────────────────────────────────────

class PhonemeEvent:
    """
    A single phoneme with its DSP target parameters and sample-accurate range.
    Adapted from ClankerVoice/voder_engine.py.
    """

    __slots__ = (
        "name", "ptype", "data",
        "start_sample", "end_sample",
        "f1", "f2", "f3", "f4", "f5",
        "bw1", "bw2", "bw3", "bw4", "bw5",
        "g1", "g2", "g3", "g4", "g5",
        "voiced", "amplitude",
    )

    # Default high-formant constants (constant across vowels for adult male voice)
    _F4  = 3300.0;  _F5  = 3750.0
    _BW4 = 250.0;   _BW5 = 300.0
    _G4  = 0.15;    _G5  = 0.08

    def __init__(self, name: str, data: dict, start_sample: int, end_sample: int):
        self.name        = name
        self.ptype       = phoneme_type(name)
        self.data        = data
        self.start_sample = start_sample
        self.end_sample   = end_sample

        f4 = self._F4;  f5 = self._F5
        bw4 = self._BW4; bw5 = self._BW5
        g4 = self._G4;  g5 = self._G5

        ptype = self.ptype

        if ptype in ("vowel", "nasal", "liquid"):
            self.f1  = float(data["f1"])
            self.f2  = float(data["f2"])
            self.f3  = float(data["f3"])
            self.f4  = float(data.get("f4", f4))
            self.f5  = float(data.get("f5", f5))
            self.bw1 = float(data.get("bw1", 90))
            self.bw2 = float(data.get("bw2", 130))
            self.bw3 = float(data.get("bw3", 180))
            self.bw4 = float(data.get("bw4", bw4))
            self.bw5 = float(data.get("bw5", bw5))
            self.g1  = float(data.get("g1", 1.0))
            self.g2  = float(data.get("g2", 0.7))
            self.g3  = float(data.get("g3", 0.3))
            self.g4  = float(data.get("g4", g4))
            self.g5  = float(data.get("g5", g5))

        elif ptype == "glide":
            self.f1  = float(data["start_f1"])
            self.f2  = float(data["start_f2"])
            self.f3  = float(data.get("start_f3", 2500))
            self.f4  = float(data.get("start_f4", f4))
            self.f5  = float(data.get("start_f5", f5))
            self.bw1 = float(data.get("bw1", 80))
            self.bw2 = float(data.get("bw2", 110))
            self.bw3 = float(data.get("bw3", 170))
            self.bw4 = float(data.get("bw4", bw4))
            self.bw5 = float(data.get("bw5", bw5))
            self.g1  = float(data.get("g1", 1.0))
            self.g2  = float(data.get("g2", 0.6))
            self.g3  = float(data.get("g3", 0.25))
            self.g4  = float(data.get("g4", g4))
            self.g5  = float(data.get("g5", g5))

        elif ptype == "fricative":
            lo     = float(data.get("noise_lo", 2000))
            hi     = float(data.get("noise_hi", 8000))
            center = (lo + hi) / 2.0
            bw     = max(200, hi - lo)
            self.f1 = center * 0.5;  self.f2 = center;       self.f3 = center * 1.5
            self.f4 = f4;            self.f5 = f5
            self.bw1 = bw;           self.bw2 = bw;           self.bw3 = bw
            self.bw4 = bw4;          self.bw5 = bw5
            self.g1 = 0.3;           self.g2 = float(data.get("amplitude", 0.4))
            self.g3 = 0.2;           self.g4 = 0.1;           self.g5 = 0.05

        elif ptype == "plosive":
            bf  = float(data.get("burst_freq", 1500))
            bwp = float(data.get("burst_bw", 600))
            self.f1 = bf * 0.5;  self.f2 = bf;          self.f3 = bf * 2.0
            self.f4 = f4;        self.f5 = f5
            self.bw1 = bwp;      self.bw2 = bwp;        self.bw3 = bwp * 1.5
            self.bw4 = bw4;      self.bw5 = bw5
            self.g1 = 0.3;       self.g2 = 0.6;         self.g3 = 0.2
            self.g4 = 0.1;       self.g5 = 0.05

        else:  # Fallback neutral vowel (schwa)
            self.f1 = 500.0;  self.f2 = 1500.0; self.f3 = 2500.0
            self.f4 = f4;     self.f5 = f5
            self.bw1 = 90.0;  self.bw2 = 130.0; self.bw3 = 180.0
            self.bw4 = bw4;   self.bw5 = bw5
            self.g1 = 1.0;    self.g2 = 0.7;    self.g3 = 0.3
            self.g4 = g4;     self.g5 = g5

        self.voiced    = data.get("voiced", True)
        amp_scale      = data.get("amplitude_scale", 1.0)
        if ptype == "nasal":
            self.amplitude = 0.7 * amp_scale
        elif ptype == "liquid":
            self.amplitude = 0.85 * amp_scale
        elif ptype == "glide":
            self.amplitude = 0.8 * amp_scale
        else:
            self.amplitude = 1.0 * amp_scale

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def get_freqs(self) -> tuple[float, ...]:
        return (self.f1, self.f2, self.f3, self.f4, self.f5)

    def get_bws(self) -> tuple[float, ...]:
        return (self.bw1, self.bw2, self.bw3, self.bw4, self.bw5)

    def get_gains(self) -> tuple[float, ...]:
        return (self.g1, self.g2, self.g3, self.g4, self.g5)


# ─── VODER ENGINE ─────────────────────────────────────────────────────────────

class VoderEngine:
    """
    5-formant Rosenberg-model formant speech synthesizer.
    Adapted from ClankerVoice/voder_engine.py.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

        # Sources
        self.saw     = SawtoothOsc(sample_rate)
        self.glottal = GlottalPulseOsc(sample_rate)
        self.noise   = WhiteNoise()

        # Formant filter bank (5 parallel BPFs)
        self.bank = BiquadBPFBank(sample_rate)

        # Separate filter for fricative shaping
        self.fric_filter  = BiquadBPF(sample_rate)
        # Lowpass to soften unvoiced fricatives
        self.fric_lowpass = OnePoleLPF(sample_rate)
        self.fric_lowpass.set_cutoff(3800.0)
        # Lowpass to soften voiceless plosive bursts
        self.burst_lowpass = OnePoleLPF(sample_rate)
        self.burst_lowpass.set_cutoff(4200.0)

        # Use Rosenberg glottal pulse by default (more voice-like than sawtooth)
        self.use_glottal = True

        # Vibrato
        self.vibrato_rate  = 5.2   # Hz (slightly off 5 for a less mechanical feel)
        self.vibrato_depth = 0.08  # semitone depth

        # Pitch jitter + amplitude shimmer (micro-variations for organic sound)
        self.jitter_amount  = 0.002
        self.shimmer_amount = 0.015
        self._jitter_rng    = np.random.default_rng(7)

        # Master output gain
        self.master_gain = 0.8

    # ------------------------------------------------------------------
    # High-level render
    # ------------------------------------------------------------------

    def render(
        self,
        events: list[PhonemeEvent],
        midi_note: float = 60,
        velocity: float = 1.0,
    ) -> np.ndarray:
        """Render a list of PhonemeEvents to a numpy float64 audio array."""
        if not events:
            return np.zeros(0, dtype=np.float64)

        total_samples = max(e.end_sample for e in events)
        output = np.zeros(total_samples, dtype=np.float64)

        base_freq = midi_to_freq(midi_note)
        self._reset_all()

        for idx, evt in enumerate(events):
            prev_evt = events[idx - 1] if idx > 0 else None
            next_evt = events[idx + 1] if idx < len(events) - 1 else None
            self._render_phoneme(output, evt, prev_evt, next_evt, base_freq)

        # Apply master gain + velocity
        output *= self.master_gain * velocity

        # Smooth raised-cosine fade-out at the end (prevent abrupt cutoffs)
        fadeout_ms      = 25.0
        fadeout_samples = min(int(fadeout_ms * self.sr / 1000.0), len(output))
        if fadeout_samples > 0:
            fade = 0.5 * (1.0 + np.cos(
                np.pi * np.linspace(0.0, 1.0, fadeout_samples)
            ))
            output[-fadeout_samples:] *= fade

        # Soft-clip (tanh) for warmth + saturation
        output = np.tanh(output * 1.2) / 1.2

        return output

    # ------------------------------------------------------------------
    # Per-phoneme dispatch
    # ------------------------------------------------------------------

    def _render_phoneme(self, buf, evt, prev_evt, next_evt, base_freq):
        ptype = evt.ptype
        n = evt.end_sample - evt.start_sample
        if n <= 0:
            return

        if ptype == "plosive":
            self._render_plosive(buf, evt, next_evt, base_freq)
        elif ptype == "fricative":
            if evt.data.get("inherit_next_formants") and next_evt is not None:
                self._render_h_phoneme(buf, evt, next_evt, base_freq)
            else:
                self._render_fricative(buf, evt, base_freq)
        else:
            self._render_formant(buf, evt, prev_evt, base_freq)

    # ------------------------------------------------------------------
    # Formant-based rendering (vowels, nasals, liquids, glides)
    # ------------------------------------------------------------------

    def _render_formant(self, buf, evt, prev_evt, base_freq):
        start = evt.start_sample
        end   = evt.end_sample
        n     = end - start

        # Context-dependent transition duration
        if prev_evt is not None:
            trans_ms      = self._transition_ms(prev_evt, evt)
            trans_samples = min(int(trans_ms * self.sr / 1000.0), n)
        else:
            trans_samples = min(int(20.0 * self.sr / 1000.0), n)

        evt_freqs  = evt.get_freqs()
        evt_bws    = evt.get_bws()
        evt_gains  = evt.get_gains()

        if prev_evt is not None:
            prev_freqs = prev_evt.get_freqs()
            prev_bws   = prev_evt.get_bws()
            prev_gains = prev_evt.get_gains()
        else:
            prev_freqs = evt_freqs
            prev_bws   = evt_bws
            prev_gains = evt_gains

        cur_freqs  = list(prev_freqs)
        cur_bws    = list(prev_bws)
        cur_gains  = list(prev_gains)

        for i in range(n):
            if i % 4096 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
            sample_pos = start + i

            # Update formant parameters at control rate (CPU savings + no zipper)
            if i % CONTROL_RATE == 0:
                if i < trans_samples and prev_evt is not None:
                    t = i / trans_samples
                    for k in range(NUM_FORMANTS):
                        cur_freqs[k] = cosine_interp(prev_freqs[k], evt_freqs[k], t)
                        cur_bws[k]   = cosine_interp(prev_bws[k],   evt_bws[k],   t)
                        cur_gains[k] = cosine_interp(prev_gains[k], evt_gains[k], t)
                else:
                    cur_freqs = list(evt_freqs)
                    cur_bws   = list(evt_bws)
                    cur_gains = list(evt_gains)
                self.bank.set_formants(
                    tuple(cur_freqs), tuple(cur_bws), tuple(cur_gains)
                )

            # Source generation
            freq = self._vibrato_freq(base_freq, sample_pos)
            freq = self._apply_jitter(freq)

            if evt.voiced:
                source = (
                    self.glottal.next_sample(freq) if self.use_glottal
                    else self.saw.next_sample(freq)
                )
            else:
                source = self.noise.next_sample()

            sample = self.bank.process(source)

            # Amplitude envelope + shimmer
            amp = evt.amplitude * self._amp_envelope(i, n, evt.ptype)
            amp *= self._apply_shimmer()

            buf[sample_pos] += sample * amp

    # ------------------------------------------------------------------
    # "h" phoneme -- formant-filtered noise shaped by the next vowel
    # ------------------------------------------------------------------

    def _render_h_phoneme(self, buf, evt, next_evt, base_freq):
        """Render 'h' as noise shaped by the NEXT phoneme's formants."""
        start = evt.start_sample
        end   = evt.end_sample
        n     = end - start

        target_freqs = next_evt.get_freqs()
        target_bws   = next_evt.get_bws()
        target_gains = tuple(g * 0.5 for g in next_evt.get_gains())

        self.bank.set_formants(target_freqs, target_bws, target_gains)
        h_amp = float(evt.data.get("amplitude", 0.25))

        for i in range(n):
            if i % 4096 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
            sample_pos = start + i
            sample = self.bank.process(self.noise.next_sample())
            amp    = h_amp * self._amp_envelope(i, n, "fricative")
            buf[sample_pos] += sample * amp

    # ------------------------------------------------------------------
    # Fricative rendering (f, v, s, z, sh, th)
    # ------------------------------------------------------------------

    def _render_fricative(self, buf, evt, base_freq):
        start = evt.start_sample
        end   = evt.end_sample
        n     = end - start
        data  = evt.data

        lo     = float(data.get("noise_lo", 2000))
        hi     = float(data.get("noise_hi", 8000))
        center = (lo + hi) / 2.0
        bw     = hi - lo

        self.fric_filter.set_params(center, max(200, bw), 1.0)
        fric_amp = float(data.get("amplitude", 0.4))

        for i in range(n):
            if i % 4096 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
            sample_pos = start + i
            shaped = self.fric_filter.process(self.noise.next_sample())

            if evt.voiced:
                # Voiced fricatives: blend noise + glottal tone (v, z)
                freq   = self._vibrato_freq(base_freq, sample_pos)
                voiced = (
                    self.glottal.next_sample(freq) if self.use_glottal
                    else self.saw.next_sample(freq)
                )
                sample = 0.55 * shaped + 0.45 * voiced
            else:
                # Unvoiced: soften with lowpass to reduce harshness
                sample = self.fric_lowpass.process(shaped)

            amp = fric_amp * self._amp_envelope_fricative(i, n)
            buf[sample_pos] += sample * amp

    # ------------------------------------------------------------------
    # Plosive rendering with VOT aspiration tail
    # ------------------------------------------------------------------

    def _render_plosive(self, buf, evt, next_evt, base_freq):
        start = evt.start_sample
        end   = evt.end_sample
        data  = evt.data
        total = end - start

        closure_ms    = float(data.get("closure_ms", 50))
        burst_ms      = float(data.get("burst_ms", 20))
        aspiration_ms = float(data.get("aspiration_ms", 15))
        burst_freq    = float(data.get("burst_freq", 1500))
        burst_bw      = float(data.get("burst_bw", 600))

        closure_samples = min(int(closure_ms * self.sr / 1000.0), total)
        burst_samples   = min(
            int(burst_ms * self.sr / 1000.0),
            total - closure_samples
        )
        if next_evt is None:
            # Word-final: use all remaining time for release decay
            asp_samples = total - closure_samples - burst_samples
        else:
            asp_samples = min(
                int(aspiration_ms * self.sr / 1000.0),
                total - closure_samples - burst_samples,
            )

        # Phase 1: Closure (silence or faint voicing bar for voiced plosives)
        for i in range(closure_samples):
            if i % 4096 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
            sample_pos = start + i
            if evt.voiced:
                freq = self._vibrato_freq(base_freq, sample_pos)
                src  = (
                    self.glottal.next_sample(freq) if self.use_glottal
                    else self.saw.next_sample(freq)
                )
                buf[sample_pos] += src * 0.06

        # Phase 2: Burst (bandpass-filtered noise around burst_freq)
        burst_filter = BiquadBPF(self.sr)
        burst_filter.set_params(burst_freq, burst_bw, 1.0)
        burst_level  = 0.35 if evt.voiced else 0.38
        if not evt.voiced:
            self.burst_lowpass.reset()

        burst_start = start + closure_samples
        for i in range(burst_samples):
            if i % 4096 == 0:
                time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
            sample_pos = burst_start + i
            if sample_pos >= end:
                break
            shaped = burst_filter.process(self.noise.next_sample())
            if not evt.voiced:
                shaped = self.burst_lowpass.process(shaped)
            t = i / max(1, burst_samples)
            # Longer attack (first 28%) to avoid click on voiceless plosives
            if t < 0.28:
                attack = 0.5 * (1.0 - math.cos(math.pi * t / 0.28))
            else:
                attack = 1.0
            decay = math.exp(-3.0 * t)
            buf[sample_pos] += shaped * attack * decay * burst_level

        # Phase 3: Aspiration / release tail
        asp_start = burst_start + burst_samples
        if asp_samples > 0:
            if next_evt is not None:
                # Shape aspiration toward the next phoneme's formants
                asp_freqs = next_evt.get_freqs()
                asp_bws   = next_evt.get_bws()
                asp_gains = tuple(g * 0.4 for g in next_evt.get_gains())
            else:
                # Word-final: release noise in the burst frequency region
                asp_freqs = (
                    burst_freq * 0.5, burst_freq, burst_freq * 1.8, 3300.0, 3750.0
                )
                asp_bws   = (burst_bw, burst_bw, burst_bw * 1.5, 250.0, 300.0)
                asp_gains = (0.15, 0.25, 0.1, 0.05, 0.02)

            self.bank.set_formants(asp_freqs, asp_bws, asp_gains)
            for i in range(asp_samples):
                if i % 4096 == 0:
                    time.sleep(0)  # yield GIL so GUI stays responsive (frequent for long sections)
                sample_pos = asp_start + i
                if sample_pos >= end:
                    break
                shaped = self.bank.process(self.noise.next_sample())
                if next_evt is not None:
                    t_asp = i / max(1, asp_samples)
                    env   = 0.5 * (1.0 - math.cos(math.pi * t_asp))
                    buf[sample_pos] += shaped * env * 0.4
                else:
                    t   = i / max(1, asp_samples)
                    env = math.exp(-4.0 * t)
                    buf[sample_pos] += shaped * env * 0.25

    # ------------------------------------------------------------------
    # Transition timing (coarticulation -- context-dependent)
    # ------------------------------------------------------------------

    @staticmethod
    def _transition_ms(prev: PhonemeEvent, cur: PhonemeEvent) -> float:
        """Return the formant transition duration (ms) for a phoneme pair."""
        pp = prev.ptype
        cp = cur.ptype

        if pp == "vowel" and cp == "vowel":
            return 90.0                               # Slow diphthong-like sweep
        if pp in ("plosive", "fricative") and cp == "vowel":
            return 35.0                               # Fast consonant->vowel onset
        if pp == "vowel" and cp in ("plosive", "fricative", "nasal"):
            return 60.0                               # Slightly slower offset
        if pp in ("nasal", "liquid") and cp == "vowel":
            return 50.0
        base_ms = 40.0 if (pp == "glide" and cp == "vowel") else 55.0

        # Dynamic clamp: transition must not eat > 40% of the target phoneme
        target_dur_ms = (cur.end_sample - cur.start_sample) * 1000.0 / 44100.0
        return min(base_ms, target_dur_ms * 0.4)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _vibrato_freq(self, base_freq: float, sample_idx: int) -> float:
        """Add slow sinusoidal vibrato to the base frequency."""
        t = sample_idx / self.sr
        vib = math.sin(2.0 * math.pi * self.vibrato_rate * t)
        return base_freq * (2.0 ** (self.vibrato_depth * vib / 12.0))

    def _apply_jitter(self, freq: float) -> float:
        """Add random pitch micro-variation (jitter)."""
        if self.jitter_amount > 0:
            freq *= 1.0 + self._jitter_rng.uniform(
                -self.jitter_amount, self.jitter_amount
            )
        return freq

    def _apply_shimmer(self) -> float:
        """Return random amplitude multiplier (shimmer)."""
        if self.shimmer_amount > 0:
            return 1.0 + self._jitter_rng.uniform(
                -self.shimmer_amount, self.shimmer_amount
            )
        return 1.0

    @staticmethod
    def _amp_envelope(i: int, n: int, ptype: str = "vowel") -> float:
        """Raised-cosine amplitude envelope, shaped per phoneme type."""
        if n <= 0:
            return 0.0
        if ptype == "vowel":
            attack_frac, release_frac = 0.04, 0.06
        elif ptype == "fricative":
            attack_frac, release_frac = 0.10, 0.10
        elif ptype == "nasal":
            attack_frac, release_frac = 0.08, 0.08
        elif ptype in ("liquid", "glide"):
            attack_frac, release_frac = 0.06, 0.06
        else:
            attack_frac, release_frac = 0.05, 0.08

        attack_end     = max(1, int(n * attack_frac))
        release_start  = int(n * (1.0 - release_frac))

        if i < attack_end:
            return 0.5 * (1.0 - math.cos(math.pi * i / attack_end))
        elif i > release_start and release_start < n:
            t = (i - release_start) / (n - release_start)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return 1.0

    @staticmethod
    def _amp_envelope_fricative(i: int, n: int) -> float:
        """Softer long-ramp envelope for fricatives (avoids clicks)."""
        if n <= 0:
            return 0.0
        attack_end    = max(1, int(n * 0.25))
        release_start = int(n * 0.75)
        if i < attack_end:
            return 0.5 * (1.0 - math.cos(math.pi * i / attack_end))
        if i > release_start and release_start < n:
            t = (i - release_start) / (n - release_start)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return 1.0

    def _reset_all(self):
        """Reset all oscillator and filter state for a fresh render."""
        self.saw.reset()
        self.glottal.reset()
        self.bank.reset()
        self.fric_filter.reset()
        self.fric_lowpass.reset()
        self.burst_lowpass.reset()


# ─── PHONEME SEQUENCER ────────────────────────────────────────────────────────

def _distribute_phonemes(
    phoneme_names: list[str],
    word_start: int,
    word_end: int,
) -> list[PhonemeEvent]:
    """
    Distribute time across the phonemes of a single word.
    - Consonants keep their default duration (clamped if needed)
    - Remaining time is shared among vowels proportionally
    Adapted from ClankerVoice/phoneme_sequencer.py.
    """
    if not phoneme_names:
        return []
    total_samples = word_end - word_start
    if total_samples <= 0:
        return []

    infos: list[dict] = []
    total_consonant_ms     = 0.0
    total_vowel_default_ms = 0.0

    for name in phoneme_names:
        data = PHONEME_TABLE.get(name)
        if data is None:
            continue
        dur_ms = default_duration_ms(name)
        v = is_vowel(name)
        infos.append({"name": name, "data": data, "dur_ms": dur_ms, "is_vowel": v})
        if v:
            total_vowel_default_ms += dur_ms
        else:
            total_consonant_ms += dur_ms

    if not infos:
        return []

    total_ms = (total_samples / SAMPLE_RATE) * 1000.0

    if total_consonant_ms >= total_ms:
        # Consonants alone exceed available time -- scale everything down
        scale = total_ms / (total_consonant_ms + total_vowel_default_ms + 1e-9)
        for info in infos:
            info["dur_ms"] *= scale
    else:
        remaining_ms = total_ms - total_consonant_ms
        if total_vowel_default_ms > 0:
            vowel_scale = remaining_ms / total_vowel_default_ms
        else:
            # No vowels -- distribute extra evenly among consonants
            vowel_scale = 1.0
            extra_per = remaining_ms / len(infos)
            for info in infos:
                info["dur_ms"] += extra_per

        for info in infos:
            if info["is_vowel"]:
                info["dur_ms"] *= vowel_scale

    events: list[PhonemeEvent] = []
    cursor = word_start
    for info in infos:
        dur_samples = max(1, int(info["dur_ms"] * SAMPLE_RATE / 1000.0))
        end = min(cursor + dur_samples, word_end)
        events.append(PhonemeEvent(info["name"], info["data"], cursor, end))
        cursor = end

    # Extend last event to fill any rounding gap
    if events and cursor < word_end:
        events[-1].end_sample = word_end

    return events


def build_phoneme_events(words: list[dict], bpm: int) -> list[PhonemeEvent]:
    """Convert a beat-based word timeline to a flat, sorted list of PhonemeEvents."""
    samples_per_beat = (60.0 / bpm) * SAMPLE_RATE
    all_events: list[PhonemeEvent] = []
    cursor_beats = 0.0

    for word in words:
        duration_beats = float(word.get("duration", 1.0))
        word_start     = int(cursor_beats * samples_per_beat)
        word_end       = int((cursor_beats + duration_beats) * samples_per_beat)
        phonemes       = [p for p in word.get("phonemes", []) if p in PHONEME_TABLE]
        events         = _distribute_phonemes(phonemes, word_start, word_end)
        all_events.extend(events)
        cursor_beats += duration_beats

    all_events.sort(key=lambda e: e.start_sample)
    return all_events


# ─── LLM SEQUENCE GENERATION ──────────────────────────────────────────────────

_FALLBACK_WORDS = [
    {"phonemes": ["oh"],           "note": 48, "duration": 2.0},
    {"phonemes": ["m", "ah"],      "note": 50, "duration": 1.5},
    {"phonemes": ["s", "ee"],      "note": 48, "duration": 1.5},
    {"phonemes": ["n", "oh"],      "note": 46, "duration": 2.0},
    {"phonemes": ["ah", "l"],      "note": 48, "duration": 1.0},
    {"phonemes": ["sh", "uh"],     "note": 43, "duration": 2.0},
    {"phonemes": ["k", "oh", "m"], "note": 48, "duration": 1.5},
    {"phonemes": ["d", "ah", "k"], "note": 45, "duration": 1.5},
]


def generate_sequence(sheet: dict, api_key: str | None = None) -> list[dict]:
    """
    Ask Claude to generate a phoneme word timeline.
    Returns list of dicts: {phonemes, note, duration}.
    """
    voder       = sheet["agents"]["voder"]
    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    mood        = sheet.get("mood", "")
    key         = sheet.get("key", "C minor")
    instruction = voder.get("instruction", "")
    hints       = voder.get("phoneme_hints", [])
    base_hz     = max(80.0, float(voder.get("fundamental_hz") or 130))
    total_beats = bars * 4

    # Convert fundamental_hz to approximate MIDI note for guidance
    base_midi = int(round(69 + 12 * math.log2(base_hz / 440.0)))
    base_midi = max(36, min(72, base_midi))

    prompt = f"""You are programming a formant speech synthesizer (Voder-style) for an AI electronic music band.
Generate a phoneme word timeline that sounds like a machine voice singing or speaking eerily.

Key: {key}
BPM: {bpm}
Bars: {bars}  (total beats available: {total_beats})
Mood: {mood}
Instruction: {instruction}
Base pitch: MIDI note {base_midi} (~{int(base_hz)} Hz)
Phoneme hints (optional inspiration): {hints}

AVAILABLE PHONEMES -- use ONLY these exact names:
  Vowels (voiced, carry melody):
    ee  ih  eh  ae  ah  aw  oh  oo  uh  er
  Nasals (voiced, hum-like):
    m  n  ng
  Fricatives (air/hiss):
    f  v  s  z  sh  th  h
  Plosives (click/burst):
    b  d  g  p  t  k
  Liquids:
    l  r
  Glides:
    w  y

Return this JSON object (NO other text):
{{
  "words": [
    {{"phonemes": ["s", "oh"], "note": {base_midi}, "duration": 1.5}},
    {{"phonemes": ["m", "ah", "n"], "note": {base_midi - 2}, "duration": 2.0}},
    {{"phonemes": ["d", "er", "k"], "note": {base_midi - 4}, "duration": 1.5}},
    ...
  ]
}}

Rules:
- "phonemes": 1-5 phonemes per word, from the lists above ONLY
- "note": MIDI note {base_midi - 7}-{base_midi + 5}. Vary it for melodic movement.
- "duration": beats per word (0.5-3.0). Vowel-heavy words should be longer.
- Total "duration" sum ≈ {total_beats} beats
- 8-20 words total
- Spend most time on vowels (ee, ah, oh, oo, uh) -- they carry the voice
- Use consonants for rhythm and texture
- Create intentional melodic pitch movement (not random)
- Think of a machine SINGING syllables, not random sounds

Return ONLY valid JSON."""

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

        words = data.get("words", [])
        valid = []
        for w in words:
            phonemes = [p for p in w.get("phonemes", []) if p in PHONEME_TABLE]
            if not phonemes:
                continue
            note     = max(36, min(84, int(w.get("note", base_midi))))
            duration = max(0.25, float(w.get("duration", 1.0)))
            valid.append({"phonemes": phonemes, "note": note, "duration": duration})

        return valid if valid else _FALLBACK_WORDS

    except Exception as e:
        print(f"  [llm error] {e}")
        return _FALLBACK_WORDS


# ─── RENDER ───────────────────────────────────────────────────────────────────

def render(words: list[dict], bpm: int, bars: int) -> AudioSegment:
    """
    Render a word timeline (phoneme lists + MIDI notes + beat durations)
    through the VoderEngine, return a pydub AudioSegment.

    Rendering strategy:
      - Build a flat list of PhonemeEvents from all words (for correct
        coarticulation context across word boundaries)
      - Render TWO VoderEngine passes from the same events:
          · upper voice  -- median MIDI note of the timeline (carries speech)
          · sub voice    -- one octave lower (adds presence and body)
      - Blend upper (0.65) + sub (0.40), then normalize to 0.90 peak so
        the perceived output level matches a single-voice render
    """
    total_beats   = bars * 4
    total_ms      = int(total_beats * (60000.0 / bpm))
    total_samples = int(total_beats * (60.0 / bpm) * SAMPLE_RATE)

    # Build flat phoneme event list for the full sequence
    all_events = build_phoneme_events(words, bpm)

    if not all_events:
        print("  [voder] No phoneme events -- returning silence")
        return AudioSegment.silent(duration=total_ms)

    # Use median MIDI note across all words for the upper voice
    notes     = [int(w.get("note", 48)) for w in words if w.get("phonemes")]
    midi_note = int(np.median(notes)) if notes else 48
    midi_note = max(36, min(84, midi_note))
    midi_sub  = max(24, midi_note - 12)   # one octave lower, floor at C1

    print(f"  Phoneme events : {len(all_events)}")
    print(f"  Upper voice    : MIDI {midi_note}  ({midi_to_freq(midi_note):.1f} Hz)")
    print(f"  Sub voice      : MIDI {midi_sub}   ({midi_to_freq(midi_sub):.1f} Hz)  [−1 oct]")

    def _fit(arr: np.ndarray) -> np.ndarray:
        """Pad or trim array to exact total_samples length."""
        if len(arr) < total_samples:
            return np.pad(arr, (0, total_samples - len(arr)))
        return arr[:total_samples]

    # Upper voice -- carries speech intelligibility and timbral character
    engine_hi = VoderEngine(SAMPLE_RATE)
    raw_hi    = _fit(engine_hi.render(all_events, midi_note=midi_note, velocity=0.85))

    # Sub voice -- one octave down, adds presence and body (separate engine state)
    engine_lo = VoderEngine(SAMPLE_RATE)
    raw_lo    = _fit(engine_lo.render(all_events, midi_note=midi_sub,  velocity=0.85))

    # Blend: upper dominant so speech remains clear, sub adds weight
    # Weights are chosen so the combined signal stays well below clipping
    # before normalization, preserving the same peak loudness as before.
    raw = raw_hi * 0.65 + raw_lo * 0.40

    # Normalize to 0.90 of full scale -- matches perceived level of original
    peak = np.max(np.abs(raw))
    if peak > 1e-9:
        raw = raw / peak * 0.90
    else:
        print("  [voder] Output is silent -- check phoneme hints and library")

    pcm   = (raw * 32767).astype(np.int16).tobytes()
    audio = AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

    if len(audio) < total_ms:
        audio = audio + AudioSegment.silent(duration=total_ms - len(audio))
    else:
        audio = audio[:total_ms]

    return audio


# ─── EFFECTS ──────────────────────────────────────────────────────────────────

def apply_effects(audio: AudioSegment, instruction: str, mood: str) -> AudioSegment:
    """Apply post-processing effects based on instruction and mood keywords."""
    combined = (instruction + " " + mood).lower()

    if any(w in combined for w in ["echo", "delay", "space", "wash", "distant", "dissociative"]):
        # Overlay a quiet, delayed copy for a mechanical space effect
        delay_ms  = 130
        feedback  = 0.28
        echo_copy = audio - 11   # -11 dB quieter
        silence   = AudioSegment.silent(duration=delay_ms)
        echo_copy = (silence + echo_copy)[:len(audio)]
        audio     = audio.overlay(echo_copy)

    return normalize(audio)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(
    sheet: dict,
    output_path: str = "voder_output.wav",
    api_key: str | None = None,
) -> AudioSegment | None:

    voder = sheet["agents"].get("voder", {})
    if not voder.get("active", False):
        print("Voder inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    instruction = voder.get("instruction", "")
    mood        = sheet.get("mood", "")
    total_ms    = int(bars * 4 * (60000 / bpm))

    print(f"\n── VODER AGENT (v2 -- 5-formant Rosenberg) ──────────")
    print(f"Key          : {sheet.get('key', '?')}")
    print(f"Duration     : {total_ms}ms  ({bars} bars @ {bpm} bpm)")
    print(f"Phoneme DB   : {len(PHONEME_TABLE)} phonemes  ({len(VOWELS)} vowels)")
    print(f"Hints        : {voder.get('phoneme_hints', [])}")
    print(f"Generating   : word timeline via LLM...")

    words = generate_sequence(sheet, api_key=api_key)

    total_beats = sum(w.get("duration", 1.0) for w in words)
    seq_str = " | ".join(
        " ".join(w["phonemes"]) for w in words[:10]
    )
    print(f"Words        : {len(words)}  ({total_beats:.1f} beats)")
    print(f"Sequence     : {seq_str}{'...' if len(words) > 10 else ''}")

    audio = render(words, bpm, bars)
    audio = apply_effects(audio, instruction, mood)

    audio.export(output_path, format="wav")
    print(f"Output       : {output_path}")
    print(f"────────────────────────────────────────────────────\n")
    return audio


# ─── EXAMPLE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

    example_sheet = {
        "title": "Machine Lament",
        "bpm": 90,
        "key": "D minor",
        "bars": 8,
        "mood": "dark, mechanical, eerie, dissociative",
        "structure": "breakdown",
        "timeSignature": "4/4",
        "agents": {
            "voder": {
                "active": True,
                "fundamental_hz": 110,
                "instruction": "slow, descending, cold, mechanical echo",
                "phoneme_hints": ["oh mah", "sss ee n"],
            }
        }
    }

    key = os.environ.get("ANTHROPIC_API_KEY")
    run(example_sheet, output_path="voder_output.wav", api_key=key)
