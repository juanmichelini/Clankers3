"""
Phoneme data table for the Clanker Voice Voder synthesizer.

Each phoneme entry contains the DSP parameters needed to synthesize it:
  - Vowels: formant frequencies (F1-F5), bandwidths, gains, voicing, duration
  - Consonants: type-specific parameters (plosive bursts, fricative bands, etc.)

Tuning notes (v2):
  - Bandwidths widened significantly -- narrow BW causes metallic ringing.
    Real speech F1 BW ~ 80-130 Hz, F2 ~ 100-160 Hz, F3 ~ 130-200 Hz.
  - F2 gain boosted -- F2 is the most important cue for vowel identity.
  - F4 (~3300 Hz) and F5 (~3700 Hz) added -- roughly constant across vowels,
    they add vocal "presence" and "body" that was missing.
  - Nasal gains boosted to be audible.
  - Plosives given aspiration_ms for VOT (voice onset time).
"""

# ---------------------------------------------------------------------------
# Vowels
# ---------------------------------------------------------------------------
# F4 and F5 are roughly constant across vowels for an adult male voice.
# They contribute timbre/presence rather than vowel identity.

_F4 = 3300
_F5 = 3750
_BW4 = 250
_BW5 = 300
_G4 = 0.15
_G5 = 0.08

VOWELS = {
    "ee": {
        "f1": 270, "f2": 2300, "f3": 3000, "f4": _F4, "f5": _F5,
        "bw1": 80,  "bw2": 120, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.8,  "g3": 0.35, "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "ih": {
        "f1": 390, "f2": 1990, "f3": 2550, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 130, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.75, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 100,
    },
    "eh": {
        "f1": 530, "f2": 1850, "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 130, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.75, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 100,
    },
    "ae": {
        "f1": 660, "f2": 1700, "f3": 2400, "f4": _F4, "f5": _F5,
        "bw1": 100, "bw2": 140, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.7,  "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "ah": {
        "f1": 800, "f2": 1200, "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 110, "bw2": 140, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.65, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "aw": {
        "f1": 500, "f2": 850,  "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 120, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.6,  "g3": 0.25, "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "oh": {
        "f1": 450, "f2": 850,  "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 120, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.6,  "g3": 0.25, "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "oo": {
        "f1": 300, "f2": 870,  "f3": 2250, "f4": _F4, "f5": _F5,
        "bw1": 80,  "bw2": 110, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.55, "g3": 0.2,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
    "uh": {
        "f1": 640, "f2": 1200, "f3": 2400, "f4": _F4, "f5": _F5,
        "bw1": 100, "bw2": 130, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.65, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 100,
    },
    "er": {
        "f1": 490, "f2": 1350, "f3": 1700, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 130, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.7,  "g3": 0.35, "g4": _G4,   "g5": _G5,
        "voiced": True, "duration_ms": 120,
    },
}

# ---------------------------------------------------------------------------
# Consonants
# ---------------------------------------------------------------------------

PLOSIVES = {
    "b": {
        "type": "plosive", "voiced": True,
        "burst_freq": 500, "burst_bw": 400,
        "closure_ms": 50, "burst_ms": 15, "aspiration_ms": 10,
    },
    "d": {
        "type": "plosive", "voiced": True,
        "burst_freq": 1500, "burst_bw": 600,
        "closure_ms": 40, "burst_ms": 15, "aspiration_ms": 10,
    },
    "g": {
        "type": "plosive", "voiced": True,
        "burst_freq": 2000, "burst_bw": 700,
        "closure_ms": 50, "burst_ms": 15, "aspiration_ms": 10,
    },
    "p": {
        "type": "plosive", "voiced": False,
        "burst_freq": 500, "burst_bw": 400,
        "closure_ms": 60, "burst_ms": 20, "aspiration_ms": 30,
    },
    "t": {
        "type": "plosive", "voiced": False,
        "burst_freq": 2000, "burst_bw": 800,
        "closure_ms": 50, "burst_ms": 20, "aspiration_ms": 30,
    },
    "k": {
        "type": "plosive", "voiced": False,
        "burst_freq": 2500, "burst_bw": 900,
        "closure_ms": 60, "burst_ms": 20, "aspiration_ms": 30,
    },
}

FRICATIVES = {
    "f": {
        "type": "fricative", "voiced": False,
        "noise_lo": 1500, "noise_hi": 7000,
        "amplitude": 0.25,
        "duration_ms": 60,
    },
    "v": {
        "type": "fricative", "voiced": True,
        "noise_lo": 1500, "noise_hi": 7000,
        "amplitude": 0.2,
        "duration_ms": 60,
    },
    "s": {
        "type": "fricative", "voiced": False,
        "noise_lo": 4500, "noise_hi": 10000,
        "amplitude": 0.4,
        "duration_ms": 70,
    },
    "z": {
        "type": "fricative", "voiced": True,
        "noise_lo": 4500, "noise_hi": 10000,
        "amplitude": 0.35,
        "duration_ms": 60,
    },
    "sh": {
        "type": "fricative", "voiced": False,
        "noise_lo": 2500, "noise_hi": 7000,
        "amplitude": 0.35,
        "duration_ms": 70,
    },
    "th": {
        "type": "fricative", "voiced": False,
        "noise_lo": 1500, "noise_hi": 5000,
        "amplitude": 0.15,
        "duration_ms": 60,
    },
    "h": {
        "type": "fricative", "voiced": False,
        # "h" is special -- it should inherit formants from the NEXT vowel.
        # The engine handles this; the noise band here is just a fallback.
        "noise_lo": 500, "noise_hi": 5000,
        "amplitude": 0.15,
        "duration_ms": 50,
        "inherit_next_formants": True,
    },
}

NASALS = {
    "m": {
        "type": "nasal", "voiced": True,
        "f1": 250, "f2": 1000, "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 120, "bw2": 180, "bw3": 250, "bw4": _BW4, "bw5": _BW5,
        # Anti-resonance simulated by boosted F1 + attenuated higher formants
        "g1": 1.0,  "g2": 0.35, "g3": 0.15, "g4": 0.05,  "g5": 0.03,
        "duration_ms": 80,
        "amplitude_scale": 0.7,
    },
    "n": {
        "type": "nasal", "voiced": True,
        "f1": 250, "f2": 1500, "f3": 2500, "f4": _F4, "f5": _F5,
        "bw1": 120, "bw2": 180, "bw3": 250, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.35, "g3": 0.15, "g4": 0.05,  "g5": 0.03,
        "duration_ms": 70,
        "amplitude_scale": 0.7,
    },
    "ng": {
        "type": "nasal", "voiced": True,
        "f1": 250, "f2": 2000, "f3": 2800, "f4": _F4, "f5": _F5,
        "bw1": 120, "bw2": 180, "bw3": 250, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.35, "g3": 0.15, "g4": 0.05,  "g5": 0.03,
        "duration_ms": 70,
        "amplitude_scale": 0.7,
    },
}

LIQUIDS = {
    "l": {
        "type": "liquid", "voiced": True,
        "f1": 350, "f2": 1050, "f3": 2400, "f4": _F4, "f5": _F5,
        "bw1": 80,  "bw2": 120, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.55, "g3": 0.25, "g4": _G4,   "g5": _G5,
        "duration_ms": 70,
    },
    "r": {
        "type": "liquid", "voiced": True,
        "f1": 350, "f2": 1300, "f3": 1700, "f4": _F4, "f5": _F5,
        "bw1": 90,  "bw2": 140, "bw3": 180, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.55, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "duration_ms": 70,
    },
}

GLIDES = {
    "w": {
        "type": "glide", "voiced": True,
        "start_f1": 300,  "start_f2": 800,  "start_f3": 2200, "start_f4": _F4, "start_f5": _F5,
        "bw1": 80,  "bw2": 110, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.6,  "g3": 0.25, "g4": _G4,   "g5": _G5,
        "duration_ms": 50,
    },
    "y": {
        "type": "glide", "voiced": True,
        "start_f1": 270,  "start_f2": 2200, "start_f3": 3000, "start_f4": _F4, "start_f5": _F5,
        "bw1": 80,  "bw2": 110, "bw3": 170, "bw4": _BW4, "bw5": _BW5,
        "g1": 1.0,  "g2": 0.65, "g3": 0.3,  "g4": _G4,   "g5": _G5,
        "duration_ms": 50,
    },
}

# ---------------------------------------------------------------------------
# Unified lookup -- merge all categories
# ---------------------------------------------------------------------------

PHONEME_TABLE: dict[str, dict] = {}
PHONEME_TABLE.update(VOWELS)
PHONEME_TABLE.update(PLOSIVES)
PHONEME_TABLE.update(FRICATIVES)
PHONEME_TABLE.update(NASALS)
PHONEME_TABLE.update(LIQUIDS)
PHONEME_TABLE.update(GLIDES)


def get_phoneme(name: str) -> dict:
    """Return the parameter dict for a named phoneme, or raise KeyError."""
    return PHONEME_TABLE[name]


def is_vowel(name: str) -> bool:
    return name in VOWELS


def phoneme_type(name: str) -> str:
    """Return one of: vowel, plosive, fricative, nasal, liquid, glide."""
    if name in VOWELS:
        return "vowel"
    data = PHONEME_TABLE.get(name, {})
    return data.get("type", "unknown")


def default_duration_ms(name: str) -> float:
    """Return the default duration for a phoneme in milliseconds."""
    data = PHONEME_TABLE[name]
    ptype = phoneme_type(name)
    if ptype == "vowel":
        return data["duration_ms"]
    elif ptype == "plosive":
        return data["closure_ms"] + data["burst_ms"] + data.get("aspiration_ms", 0)
    else:
        return data.get("duration_ms", 70)
