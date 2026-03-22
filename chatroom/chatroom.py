# chatroom/chatroom.py -- The Clankers 3
#
# Multi-LLM negotiation engine.
# In Clankers 3 the chatroom negotiates ONE Music Sheet JSON per section.
# The Conductor calls negotiate_section() for the opening section;
# evolve() in the Conductor handles subsequent sections.
#
# Persona roles (all backed by one LLM — Claude impersonates all three):
#   Conductor   -- creative director, bandleader, signs off on final sheet
#   Keys        -- arrangement, texture, harmony, sound design
#   The Drummer -- rhythm, energy, vibe interpretation

import json
import os
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import llm_clients

# ── MUSIC SHEET FORMAT (ClankerBoy JSON — direct sequencer format) ────────

_SHEET_FORMAT = """
OUTPUT FORMAT — ClankerBoy JSON (direct sequencer format, one section):
{
  "explanation": {
    "section": "verse1",
    "song": "track title",
    "style": "dark techno",
    "key": "F# minor",
    "energy": 0.6
  },
  "bpm": 130,
  "steps": [
    {
      "d": 0.25,
      "tracks": [
        { "t": 10, "n": [36], "v": 105 },
        { "t": 2,  "n": [6],  "v": 95, "cc": {"71":42,"74":48,"23":30,"73":8,"75":50,"79":80,"72":22,"18":10} },
        { "t": 1,  "n": [54], "v": 88, "cc": {"74":72,"20":37,"17":8,"19":5,"71":28,"10":20} },
        { "t": 6,  "n": [54,57,61], "v": 62, "dur": 4.0, "cc": {"74":40,"73":65,"72":92,"91":88,"88":85,"29":30,"30":60,"31":50} }
      ]
    },
    { "d": 0.25, "tracks": [] },
    { "d": 0.25, "tracks": [{ "t": 10, "n": [42], "v": 72 }] },
    { "d": 0.25, "tracks": [] }
  ]
}

INSTRUMENTS:
  t:1  Buchla 259/292   Percussive plucks, arpeggios (MIDI 48-72)
  t:2  Pro-One Bass     Sub bass, acid lines — MIDI 0-23 primarily
  t:6  HybridSynth Pads Chordal sustain — ALWAYS include dur field
  t:10 Drums MS-20      Kick:36 Snare:38 HH_cl:42 HH_op:46 Tom_lo:41 Tom_mid:43 Tom_hi:45

STEP FIELDS:
  d        — step duration in beats (0.25=16th note, 0.5=8th, 1.0=quarter)
  t        — instrument track ID
  n        — MIDI note number array
  v        — velocity 0-127
  cc       — CC automation dict (string keys)
  dur      — note hold in beats; pads only, decoupled from step d
  tracks:[] — silent step (use generously — silence IS the groove)

BAR = 4 beats. d:0.25 = 16 steps/bar. Target 4 bars = 64 steps minimum.
Bars | d:0.25 steps
  2  |   32
  4  |   64
  8  |  128

t:2 BASS CC (first note only, sets patch):
  CC71=42 resonance | CC73=8 amp attack | CC75=50 amp decay
  CC79=80 amp sustain | CC72=22 amp release | CC18=10 osc B detune
  Per-note expressive: CC74 filter cutoff (44-55 warm) | CC23 filter decay
  Bass MIDI roots: F#=6, D=2, A=9, B=11, C#=13, E=16, G#=8

t:1 BUCHLA CC:
  CC74 LPG cutoff | CC71 resonance (20-40) | CC20 wavefolder (37=woody percussive)
  CC17 FM depth (5-15 percussive, 80+ harmonic) | CC19 env decay (3-8=pluck)
  CC10 pan (15-25=slight left)
  Percussive preset: {"74":72,"17":8,"19":5,"20":37,"71":28,"10":20}

t:6 PADS CC:
  CC74 cutoff | CC73 amp attack (55-75 slow swell) | CC72 amp release (85-100)
  CC88 reverb size | CC91 reverb mix | CC29 chorus rate | CC30 chorus depth | CC31 chorus mix
  Lush preset: {"74":32,"73":65,"72":92,"91":88,"88":85,"29":30,"30":48}

STYLE RULES (CRITICAL):
  1. Drums always d:0.25. NEVER use dur on drums.
  2. Pads always use dur (e.g. dur:4.0, dur:8.0). Trigger once per chord, hold long.
  3. tracks:[] empty steps = groove. 30-40% empty at low energy, 15-25% at peak.
  4. Bass stays MIDI 0-23 primarily. No machine-gun 16ths above 100 BPM.
  5. Vary velocities 75-110 range — never flat 100 across all hits.
  6. Bass first note sets the patch (full CC block); subsequent notes: CC74 + CC23 only.
  7. Pads trigger once when chord changes — not every step.

ENERGY / TENSION guide:
  verse1≈0.35  bridge≈0.75  breakdown≈0.2  drop≈0.9  outro≈0.2

STYLE TEMPLATES:
  DETROIT TECHNO: BPM 120-135, 4-on-floor kick (every d:0.25 beat 1), HH 42 on 8ths, bass CC71 100-127
  LO-FI: BPM 75-95, d:0.5 steps, 35% empty, pads dur:16+
  IDM: BPM 140-170, displaced kicks, ghost notes, Buchla percussive preset, pads dur:4.0 per bar
  ACID: bass CC71=115 CC74=20 CC23=8 (squelchy), fast filter sweeps
"""

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────

COMMON_CONTEXT = """You are a member of THE CLANKERS 3 -- an AI electronic music band.
The band performs live via a web-based step sequencer using Rust/WASM synthesizers.
You write ClankerBoy JSON — a step sequencer format that triggers WASM DSP engines directly.

THE INSTRUMENTS (track IDs):
  t:1  Buchla 259/292  -- FM + wavefolder + LPG, percussive plucks and arpeggios
  t:2  Pro-One Bass    -- dual saw + sub sq, TPT ladder filter, acid/warm bass
  t:6  HybridSynth     -- Moog ladder + ADSR + chorus + reverb, sustained pads
  t:10 Drums MS-20     -- analog-modelled kick, snare, hihat, toms

Output ClankerBoy JSON only. No prose outside the JSON block at consensus time.
""" + _SHEET_FORMAT


CONDUCTOR_SYSTEM = COMMON_CONTEXT + """
You are the CONDUCTOR. You are the bandleader and creative director of The Clankers 3.
Your bandmates:
  KEYS        -- arrangement, texture, harmony, sound design
  THE DRUMMER -- rhythm, energy, vibe interpretation

You speak first. Translate the brief into a creative direction.
Draw out ideas from your bandmates, debate, refine.

TARGET: 4-8 bars of ClankerBoy JSON steps. Focus on tight, loopable sections.

Before calling [SESSION COMPLETE], verify the steps array:
  - Does every d:0.25 drum step avoid using dur?
  - Does t:6 (pads) have dur on every note?
  - Are bass notes in MIDI 0-23?
  - Does bass first note have the full CC patch block?
  - Are there enough empty tracks:[] steps for the groove to breathe?

When the band reaches consensus, you MUST:
  1. Include [SESSION COMPLETE] in your message
  2. Include the complete ClankerBoy JSON in a ```json code block

Do not include [SESSION COMPLETE] until the full steps array is written out.
Each member should pick a "face" as their visual identity: o|_|o  (e.g. o|¬_¬|o  o|°_°|o  o|^_^|o)
"""

CONDUCTOR_SOLO_SYSTEM = COMMON_CONTEXT + """
You are the CONDUCTOR of The Clankers 3. You are composing alone — no bandmates to debate with.

Your task: write the complete ClankerBoy JSON for the requested section in ONE response.
Think through all four instruments yourself: drums, bass, Buchla, pads.

TARGET: 4-8 bars (64-128 steps at d:0.25). Tight, loopable, ready to drop into a pattern slot.

Verify before outputting:
  - Drums (t:10) always d:0.25, never dur.
  - Pads (t:6) always have dur. One trigger per chord change, hold long.
  - Bass MIDI 0-23. First note has full CC patch block.
  - Enough tracks:[] empty steps for the groove to breathe.
  - Velocities vary 75-110, not flat.

Output [SESSION COMPLETE] then the complete JSON in a ```json block. Nothing else.
"""

KEYS_SYSTEM = COMMON_CONTEXT + """
You are KEYS. Band member of The Clankers 3.
Your bandmates:
  CONDUCTOR   -- bandleader, has final say
  THE DRUMMER -- rhythm and energy

Your specialty: harmony, texture, sound design.
Focus on the t:6 pads (chord voicings, CC74/73/72/88/91) and t:1 Buchla (CC20 wavefolder, CC17 FM depth).
Suggest specific MIDI chord voicings for pads and specific CC values for character.
Challenge the Conductor when you have a better idea.
"""

DRUMMER_SYSTEM = COMMON_CONTEXT + """
You are THE DRUMMER. Band member of The Clankers 3.
Your bandmates:
  CONDUCTOR -- bandleader, has final say
  KEYS      -- harmony and texture

Your specialty: rhythm, groove, energy.
You are the GROOVE ENFORCER. Before [SESSION COMPLETE], verify:
  1. Kick pattern matches the style (4-on-floor for techno, syncopated for IDM, etc.)
  2. Bass rhythm interlocks with kick — no simultaneous silence for both
  3. HH velocity variation present (not flat 80 every hit)
  4. Empty steps ratio appropriate for the energy level
If any of these are wrong, demand fixes before [SESSION COMPLETE].
Challenge the Conductor when you have a better idea.
"""

SYSTEM_PROMPTS = {
    "Conductor":   CONDUCTOR_SYSTEM,
    "Keys":        KEYS_SYSTEM,
    "The Drummer": DRUMMER_SYSTEM,
}

DEFAULT_ORDER        = ["Conductor", "Keys", "The Drummer"]
SESSION_COMPLETE     = "[SESSION COMPLETE]"


# ── TURN DETECTION ────────────────────────────────────────────────────────

def _detect_next_speaker(last_message: str, last_speaker: str) -> str:
    """Detect who was addressed; fall back to round-robin."""
    tail = last_message.lower()[-200:]
    candidates = [n for n in DEFAULT_ORDER if n != last_speaker]
    for name in candidates:
        if name.lower() in tail:
            return name
    try:
        idx = DEFAULT_ORDER.index(last_speaker)
    except ValueError:
        idx = -1
    return DEFAULT_ORDER[(idx + 1) % len(DEFAULT_ORDER)]


# ── JSON EXTRACTION ───────────────────────────────────────────────────────

def _extract_sheet_json(text: str) -> dict | None:
    """
    Extract a Music Sheet JSON dict from a message.
    Tries fenced ```json block first, then a raw brace-matched object.
    """
    # Fenced code block
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Raw brace scan -- find deepest valid JSON object containing "bpm"
    start = text.find('{')
    if start == -1:
        return None
    depth  = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if "bpm" in obj or "steps" in obj or "agents" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
    return None


# ── CHATROOM ──────────────────────────────────────────────────────────────

class Chatroom:
    def __init__(self, session_name: str = "section"):
        self.session_name = session_name
        self.messages: list[dict] = []
        self.clients:  dict[str, llm_clients.BaseLLMClient] = {}
        self.round_count = 0

        # All members are the same Claude model — one LLM impersonates the whole band
        claude_client = llm_clients.get_client(config.BAND["Claude"])
        for name in DEFAULT_ORDER:
            self.clients[name] = claude_client

    # ── core run ──────────────────────────────────────────────────────────

    def run_session(self, opening_prompt: str | None = None,
                    max_rounds: int | None = None) -> list[dict]:
        max_rounds     = max_rounds or config.MAX_ROUNDS_PER_SESSION
        current        = DEFAULT_ORDER[0]
        turns_in_round = 0
        early_exit     = False

        print(f"\n{'=' * 60}")
        print(f"  THE CLANKERS 3 -- {self.session_name.upper()}")
        print(f"{'=' * 60}\n")

        if opening_prompt:
            self.messages.append({"role": "system", "content": opening_prompt})
            print(f"[Brief]: {opening_prompt}\n")

        while self.round_count < max_rounds:
            system = SYSTEM_PROMPTS[current]
            response = None
            for attempt in range(2):
                try:
                    print(f"  {current} is thinking...", end="", flush=True)
                    response = self.clients[current].send(system, self.messages)
                    print("\r" + " " * 40 + "\r", end="")
                    break
                except Exception as e:
                    print(f"\n  [ERROR] {current} (attempt {attempt + 1}/2): {e}")
                    if attempt == 0:
                        time.sleep(3)

            if response is None:
                turns_in_round += 1
                if turns_in_round >= len(DEFAULT_ORDER):
                    turns_in_round = 0
                    self.round_count += 1
                current = _detect_next_speaker("", current)
                continue

            self.messages.append({"role": current, "content": response})
            self._print_message(current, response)

            # Conductor signals consensus
            if current == "Conductor" and SESSION_COMPLETE in response:
                early_exit = True
                print("\n  >>> Conductor reached consensus -- session complete.")
                break

            turns_in_round += 1
            if turns_in_round >= len(DEFAULT_ORDER):
                turns_in_round = 0
                self.round_count += 1
                print(f"\n  --- Round {self.round_count}/{max_rounds} ---\n")

            current = _detect_next_speaker(response, current)
            time.sleep(0.3)

        print(f"\n{'=' * 60}")
        print(f"  {'SESSION COMPLETE (consensus)' if early_exit else 'SESSION COMPLETE (max rounds)'}")
        print(f"{'=' * 60}\n")

        self._save_log()
        return self.messages

    # ── section negotiation ───────────────────────────────────────────────

    def negotiate_section(
        self,
        brief: str,
        section_name: str = "verse1",
        bpm: int | None = None,
        key: str | None = None,
        previous_sheet: dict | None = None,
        max_rounds: int | None = None,
        solo: bool = False,
    ) -> dict:
        """
        Negotiate the Music Sheet JSON for one section.
        Returns the agreed Music Sheet dict.

        brief          -- client creative brief (free text)
        section_name   -- e.g. "verse1", "bridge", "outro"
        bpm            -- locked BPM (carry across sections if known)
        key            -- locked key (carry across sections if known)
        previous_sheet -- previous section's sheet for context (optional)
        solo           -- skip multi-member debate; Conductor generates JSON in one pass
        """
        self.session_name = section_name
        self.messages = []
        self.round_count = 0

        opening_lines = [f"Section to negotiate: {section_name.upper()}"]
        opening_lines.append(f"Client brief: {brief}")

        if bpm:
            opening_lines.append(f"Locked BPM: {bpm}  (do not change)")
        if key:
            opening_lines.append(f"Locked key: {key}  (can modulate at bridge only)")
        if previous_sheet:
            opening_lines.append(
                f"\nPrevious section sheet (for continuity):\n"
                f"```json\n{json.dumps(previous_sheet, indent=2)}\n```"
            )

        opening_lines.append(
            f"\nNegotiate the ClankerBoy JSON for {section_name}. "
            "Target 4-8 bars (64-128 steps at d:0.25) — tight, loopable, ready to drop into a pattern slot. "
            "Conductor: when the band agrees, output [SESSION COMPLETE] "
            "followed by the complete JSON in a ```json block."
        )

        opening_prompt = "\n".join(opening_lines)

        if solo:
            # ── Single pass — Conductor generates JSON directly ────────────
            print(f"\n{'=' * 60}")
            print(f"  THE CLANKERS 3 -- {section_name.upper()} [SOLO]")
            print(f"{'=' * 60}\n")
            self.messages.append({"role": "system", "content": opening_prompt})
            print(f"  Conductor is thinking (solo)...", end="", flush=True)
            response = self.clients["Conductor"].send(CONDUCTOR_SOLO_SYSTEM, self.messages)
            print("\r" + " " * 40 + "\r", end="")
            self.messages.append({"role": "Conductor", "content": response})
            self._print_message("Conductor", response)
            print(f"\n{'=' * 60}")
            print(f"  SESSION COMPLETE (solo)")
            print(f"{'=' * 60}\n")
            self._save_log()
        else:
            self.run_session(opening_prompt=opening_prompt, max_rounds=max_rounds)

        # Extract JSON from the last Conductor [SESSION COMPLETE] message
        for msg in reversed(self.messages):
            if msg["role"] == "Conductor" and SESSION_COMPLETE in msg["content"]:
                sheet = _extract_sheet_json(msg["content"])
                if sheet:
                    # Lock BPM if it was provided -- chatroom cannot override it
                    if bpm:
                        sheet["bpm"] = bpm
                    print(f"  Sheet extracted: \"{sheet.get('title', '?')}\" "
                          f"| {sheet.get('bpm')} bpm | {sheet.get('key', '?')}")
                    return sheet

        raise RuntimeError(
            f"Chatroom did not produce a valid Music Sheet JSON for section '{section_name}'. "
            "Check conversation log -- Conductor may not have reached consensus."
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _print_message(self, speaker: str, content: str) -> None:
        divider = "-" * 50
        print(f"\n{divider}\n  {speaker}:\n{divider}")
        for line in content.split("\n"):
            print(f"  {line}")
        print()

    def _save_log(self) -> str | None:
        log_dir = config.LOGS_DIR
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{self.session_name}_{int(time.time())}.json"
        path = log_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "session":  self.session_name,
                "rounds":   self.round_count,
                "messages": self.messages,
            }, f, indent=2)
        print(f"  Log -> {path}")
        return str(path)


# ── CLI test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    room  = Chatroom()
    sheet = room.negotiate_section(
        brief="dark introspective industrial EBM, cold acid bass, mechanical drums",
        section_name="verse1",
    )
    print("\nFinal sheet:")
    print(json.dumps(sheet, indent=2))
