# config.py -- The Clankers 3
# Inherits API keys + LLM settings from the_Clankers (sibling project).
# Uses importlib to load by absolute path -- avoids name collision with this file.

import importlib.util
from pathlib import Path

_SIBLING = Path(__file__).resolve().parent.parent / "the_Clankers" / "config.py"
_spec    = importlib.util.spec_from_file_location("_clankers1_config", _SIBLING)
_orig    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)

# ── Inherited ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _orig.ANTHROPIC_API_KEY
GEMINI_API_KEY    = _orig.GEMINI_API_KEY
OPENAI_API_KEY    = _orig.OPENAI_API_KEY
CLAUDE_MODEL      = _orig.CLAUDE_MODEL
GEMINI_MODEL      = _orig.GEMINI_MODEL
CHATGPT_MODEL     = _orig.CHATGPT_MODEL
BAND              = _orig.BAND
GEMINI_TIMEOUT_MS = _orig.GEMINI_TIMEOUT_MS
GEMINI_RPM        = getattr(_orig, "GEMINI_RPM",  140)
CLAUDE_RPM        = getattr(_orig, "CLAUDE_RPM",  200)
CHATGPT_RPM       = getattr(_orig, "CHATGPT_RPM", 200)

# ── Chatroom settings ──────────────────────────────────────────────────────
# Shorter than the original -- we negotiate one section at a time.
MAX_ROUNDS_PER_SESSION = 6

# ── VST paths ──────────────────────────────────────────────────────────────
# Set any value to None to fall back to numpy synthesis for that agent.
VST_PATHS: dict[str, str | None] = {
    "bass_sh101":   r"C:\Program Files\Common Files\VST3\bassYnth Pro-One.vst3",
    "drums":        r"C:\Program Files\Common Files\VST3\Antigravity Drums.vst3",
    "harmony_lead": r"C:\Program Files\Common Files\VST3\Buchla Systems.vst3",
    "harmony_pad":  r"C:\Program Files\Common Files\VST3\HybridSynth.vst3",
    "voder":        None,   # pure formant synthesis -- no VST
    "voice":        None,   # sample-based -- no VST
}

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR  = _ROOT / "output"
SAMPLES_DIR = _ROOT / "samples"
LOGS_DIR    = OUTPUT_DIR / "logs"
