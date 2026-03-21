#!/usr/bin/env python3
# test_chatroom.py -- solo Claude chatroom test
# Runs negotiate_section() with only Claude (no Gemini/ChatGPT).
# Claude takes all turns and produces the Music Sheet JSON on its own.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chatroom.chatroom as _cm

# ── Patch to single-LLM mode ──────────────────────────────────────────────
# Override the module-level speaker list so only Claude runs.
_cm.DEFAULT_ORDER = ["Claude"]

from chatroom.chatroom import Chatroom

room  = Chatroom(session_name="test_verse1")
sheet = room.negotiate_section(
    brief       = "dark industrial EBM. cold acid bass. mechanical drums. voder voice.",
    section_name= "verse1",
    max_rounds  = 3,   # Claude should reach consensus in 1-2 turns
)

import json
print("\n" + "=" * 60)
print("EXTRACTED MUSIC SHEET:")
print("=" * 60)
print(json.dumps(sheet, indent=2))
