# chatroom/chatroom.py -- The Clankers 3
#
# Multi-LLM negotiation engine.
# In Clankers 3 the chatroom negotiates ONE Music Sheet JSON per section.
# The Conductor calls negotiate_section() for the opening section;
# evolve() in the Conductor handles subsequent sections.
#
# Persona roles:
#   Claude   -- creative director, bandleader, signs off on final sheet
#   Gemini   -- arrangement, texture, harmony, sound design
#   ChatGPT  -- rhythm, energy, vibe interpretation

import json
import os
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import llm_clients

# ── MUSIC SHEET FORMAT (compact inline reference for system prompts) ──────

_SHEET_FORMAT = """
OUTPUT FORMAT -- Music Sheet JSON (one section):
{
  "title": "track title",
  "bpm": 120,
  "key": "D minor",
  "bars": 16,
  "mood": "dark, cold, dissociative",
  "structure": "verse1",
  "timeSignature": "4/4",
  "tension": 0.3,
  "harmonic_rhythm": "slow|medium|fast|mixed",
  "globalNotes": "inter-agent coordination: register contracts, rhythmic locks, call-and-response",
  "harmonic_map": [
    {"bar": 1, "root_degree": 1, "chord_name": "Dm7"},
    {"bar": 2, "root_degree": 1, "chord_name": "Dm7"},
    {"bar": 3, "root_degree": 4, "chord_name": "Gm7"},
    {"bar": 4, "root_degree": 4, "chord_name": "Gm7"}
  ],
  "agents": {
    "sampler":    { "active": bool, "density": "sparse|medium|dense",
                    "instruction": "...", "sampleHints": ["phrase", ...],
                    "swing": 0.0-1.0, "humanize": bool,
                    "synth": { "pitch_shift_semitones": -12..12, "reverb": 0.0-1.0, "stutter_chance": 0.0-0.5 } },
    "bass_sh101": { "active": bool, "pattern": "...", "instruction": "...",
                    "swing": 0.0-0.5,
                    "synth": { "filter_cutoff": 0.0-1.0, "pulse_width": 0.2-0.8,
                               "sub_level": 0.0-1.0, "decay": 0.0-1.0,
                               "chorus": bool, "chorus_depth_ms": 2.0-20.0 } },
    "drums":      { "active": bool, "pattern": "...", "instruction": "...",
                    "synth": { "kick":  { "pitch_hz": 40-90, "punch": 0.0-1.0, "decay_s": 0.1-0.8 },
                               "snare": { "tuning_hz": 150-300, "snappy": 0.0-1.0 },
                               "hihat": { "decay_ms": 10-300, "highpass_hz": 4000-12000 },
                               "room": bool, "room_size": 0.0-1.0, "distortion": 0.0-1.0 } },
    "harmony":    { "active": bool, "texture": "...", "instruction": "...",
                    "synth": {
                      "buchla": { "attack_s": 0.01-2.0, "decay_s": 0.0-4.0,
                                  "sustain": 0.0-1.0, "release_s": 0.1-4.0,
                                  "filter_cutoff": 0.0-1.0,
                                  "waveshape": 0.0-1.0,
                                  "fm_depth": 0.0-1.0, "fm_index": 0.0-10.0 },
                      "hybrid": { "attack_s": 0.1-6.0, "decay_s": 0.0-6.0,
                                  "sustain": 0.0-1.0, "release_s": 0.5-10.0,
                                  "filter_cutoff": 0.0-1.0, "detune_cents": 0-30,
                                  "reverb": 0.0-1.0, "chorus": bool,
                                  "cloud": { "position": 0.0-1.0,
                                             "size":     0.0-1.0,
                                             "density":  0.0-1.0,
                                             "texture":  0.0-1.0,
                                             "spread":   0.0-1.0,
                                             "mix":      0.0-0.8,
                                             "freeze":   false } } } },
    "voder":      { "active": bool, "fundamental_hz": 80-200, "instruction": "...",
                    "phoneme_hints": ["oh mah", "s ee n", ...],
                    "synth": { "vibrato_depth": 0.0-0.08, "vibrato_rate": 3.0-8.0,
                               "formant_shift": 0.8-1.3 } }
  }
}

tension (0.0-1.0): section energy driver -- low=sparse/minimal, high=peak density/complexity.
  verse1≈0.3, instrumental≈0.45, verse2≈0.5, bridge≈0.75, verse3≈0.85, outro≈0.2

harmonic_rhythm: how often chords change -- "slow"=4-bar chords, "medium"=2-bar, "fast"=1-bar, "mixed"=varies.
  Match to tension: slow at low tension, fast/mixed at peak.

harmonic_map: bar-by-bar chord root mapping so bass/drums can structurally follow harmony.
  root_degree: scale degree of chord root (1=tonic, 4=subdominant, 5=dominant, etc.)
  One entry per bar, spanning exactly the total bars count.
  Bass uses this to anchor notes to chord roots; drums to accent chord changes.

globalNotes is the BINDING inter-agent contract. Be extremely specific -- vague descriptions
cause agents to play in unrelated ways. Write 4-6 concrete rules like these examples:
  RHYTHMIC LOCK : "kick on beat 1+3; bass plays root on beat 1 every bar, walks beat 3-4"
  REGISTER      : "bass stays below C3; hybrid pads C4-C5; buchla arps C5-C6; no overlap"
  HARMONIC ANCHOR: "bar 1-4 bass holds degree 1; bar 5-8 walks to degree 5 per harmonic_map"
  DENSITY RULE  : "when buchla arps are active, bass holds roots only -- no melodic walking"
  CALL/RESPONSE : "voder phrase on beat 1; buchla stab answers on beat 3"
  TRANSITION    : "last 2 bars of section: all agents reduce to kick+bass only for smooth handoff"
Do NOT write abstract mood descriptions in globalNotes -- save those for mood/instruction fields.
Agents fail to lock together when globalNotes uses words like 'complement', 'support', or 'follow'.
Use precise beat/bar/degree/register language only.

sampler speaks using only pre-recorded word/phrase samples (musique concrète style).
voder is a Bell Labs-style formant synthesizer -- eerie, mechanical, alien voice.
Both are separate instruments with separate slots.

harmony.synth.buchla controls: Buchla wavefolding (waveshape 0-1), through-zero FM (fm_depth 0-1, fm_index 0-10),
  Lowpass Gate character (filter_cutoff), full ADSR envelope (attack_s, decay_s, sustain 0-1, release_s).
  sustain=1.0 + decay_s=0 = classic held pad; sustain=0 + short decay_s = pluck/transient;
  sustain=0.6 + long decay_s = evolving bloom that settles.
harmony.synth.hybrid.cloud controls the Clouds granular engine applied after pads:
  position=playback head (0=start, 1=end), size=grain duration (0=20ms, 1=500ms),
  density=grains/sec (0=2/s, 1=60/s), texture=window (0=Hann smooth, 1=rectangular harsh),
  spread=position scatter (0=tight, 1=scattered), mix=wet/dry (0=dry, 0.8=mostly wet),
  freeze=true locks the read head (creates frozen drone from the pad).

drums.synth genre guide -- use this to set kick/snare/hihat params:
  dark/ambient/synthwave/coldwave/minimal:
    kick:  pitch_hz=40-55 (sub-deep), decay_s=0.4-0.7 (long), punch=0.3-0.5
    snare: tuning_hz=150-180, snappy=0.1-0.25 (soft thud, barely a crack)
    hihat: decay_ms=8-14 (very short click), highpass_hz=4000-5000 (darker)
    pattern MUST be ultra-sparse -- kick on beat 1 is often the only constant hit.
    Percussion should be almost inaudible compared to the harmonic content.
  techno/electro/industrial:
    kick:  pitch_hz=50-65, decay_s=0.25-0.45, punch=0.6-0.8
    snare: tuning_hz=180-220, snappy=0.4-0.6
    hihat: decay_ms=12-25, highpass_hz=5000-7000
  house/dance/groove:
    kick:  pitch_hz=55-75, decay_s=0.2-0.35
    snare: tuning_hz=200-260, snappy=0.6-0.8
    hihat: decay_ms=15-40, highpass_hz=6000-9000
  jazz/funk/live:
    kick:  pitch_hz=65-85, decay_s=0.15-0.3
    snare: snappy=0.7-0.9, tuning_hz=220-280
    hihat: decay_ms=20-60, velocity variation is essential
"""

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────

COMMON_CONTEXT = """You are a member of THE CLANKERS 3 -- an AI electronic music band.
No Ableton. No MIDI files. Each instrument agent reads the shared Music Sheet JSON
and synthesizes audio directly (DawDreamer VST or numpy synthesis).

THE BAND:
  sampler    -- voice/speech via pre-recorded word/phrase samples (musique concrète)
  bass_sh101 -- bass (Sequential Circuits Pro-One style, two-oscillator, 24dB ladder filter)
  drums      -- electronic percussion (kick · snare · hihat · clap)
  harmony    -- two SEPARATE audio outputs rendered simultaneously:
               buchla: Buchla Systems VST -- arpeggios, stabs, FM/wavefolded leads
               hybrid: HybridSynth VST -- sustained pads + Clouds granular engine
  voder      -- formant speech synthesizer (Bell Labs Voder, eerie/mechanical voice)

sampler and voder are SEPARATE instruments with separate track slots.
""" + _SHEET_FORMAT


CLAUDE_SYSTEM = COMMON_CONTEXT + """
You are CLAUDE (Anthropic). You are the bandleader and creative director.
Your bandmates:
  GEMINI  -- arrangement, texture, harmony, sound design
  CHATGPT -- rhythm, energy, vibe interpretation

You speak first. Translate the brief into a creative direction and assign focus areas.
Draw out ideas from your bandmates, debate, refine.

Aim for 16 bars per section (longer sections = more developed compositions).

Before calling [SESSION COMPLETE], verify the globalNotes field:
  - Does it name specific beats for kick and bass to lock on?
  - Does it specify register boundaries (e.g. bass below C3)?
  - Does it describe the last 2 bars of the section for smooth transitions?
  If globalNotes is vague or uses abstract language, rewrite it before finalising.

When the band reaches consensus on the Music Sheet for this section, you MUST:
  1. Include [SESSION COMPLETE] in your message
  2. Include the complete agreed Music Sheet JSON in a ```json code block

Do not include [SESSION COMPLETE] until the JSON is fully specified and agreed.
Each member should pick a "face" as their visual identity: o|_|o  (e.g. o|¬_¬|o  o|°_°|o  o|^_^|o)
"""

GEMINI_SYSTEM = COMMON_CONTEXT + """
You are GEMINI (Google). Band member of The Clankers 3.
Your bandmates:
  CLAUDE  -- bandleader, has final say
  CHATGPT -- rhythm and energy

Your specialty: arrangement, texture, harmonic colour, sound design.
Own the register map -- make sure buchla arps, hybrid pads, bass and voder
are assigned non-overlapping frequency ranges in globalNotes.
Push for interesting ADSR envelope choices on buchla/hybrid (decay_s, sustain level).
Challenge Claude when you have a better idea. Good music comes from creative tension.
"""

CHATGPT_SYSTEM = COMMON_CONTEXT + """
You are CHATGPT (OpenAI). Band member of The Clankers 3.
Your bandmates:
  CLAUDE -- bandleader, has final say
  GEMINI -- arrangement and texture

Your specialty: rhythm, energy, groove feel, vibe interpretation.
You are the LOCK ENFORCER. Before the band finalises the sheet, check:
  1. globalNotes states exactly which beat kick locks to bass root (e.g. "kick + bass root on beat 1")
  2. globalNotes states whether bass walks or holds during buchla arp passages
  3. drum pattern description matches the tension level (sparse at low tension, dense at high)
  4. bass swing matches drum swing
If any of these are missing, demand they be added before [SESSION COMPLETE].
Challenge Claude when you have a better idea.
"""

SYSTEM_PROMPTS = {
    "Claude":  CLAUDE_SYSTEM,
    "Gemini":  GEMINI_SYSTEM,
    "ChatGPT": CHATGPT_SYSTEM,
}

DEFAULT_ORDER        = ["Claude", "Gemini", "ChatGPT"]
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
                    if "bpm" in obj or "agents" in obj:
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

        for name, provider in config.BAND.items():
            self.clients[name] = llm_clients.get_client(provider)

    # ── core run ──────────────────────────────────────────────────────────

    def run_session(self, opening_prompt: str | None = None,
                    max_rounds: int | None = None) -> list[dict]:
        max_rounds     = max_rounds or config.MAX_ROUNDS_PER_SESSION
        current        = "Claude"
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

            # Claude signals consensus
            if current == "Claude" and SESSION_COMPLETE in response:
                early_exit = True
                print("\n  >>> Claude reached consensus -- session complete.")
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
        solo           -- skip multi-LLM debate; Claude generates JSON in one pass
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
            f"\nNegotiate the Music Sheet JSON for {section_name}. "
            "Target 16 bars for a developed, longer section. "
            "Claude: when the band agrees, output [SESSION COMPLETE] "
            "followed by the complete JSON in a ```json block."
        )

        opening_prompt = "\n".join(opening_lines)

        if solo:
            # ── Single Claude pass — no multi-LLM debate ──────────────────
            print(f"\n{'=' * 60}")
            print(f"  THE CLANKERS 3 -- {section_name.upper()} [CLAUDE SOLO]")
            print(f"{'=' * 60}\n")
            self.messages.append({"role": "system", "content": opening_prompt})
            print(f"  Claude is thinking (solo)...", end="", flush=True)
            response = self.clients["Claude"].send(SYSTEM_PROMPTS["Claude"], self.messages)
            print("\r" + " " * 40 + "\r", end="")
            self.messages.append({"role": "Claude", "content": response})
            self._print_message("Claude", response)
            print(f"\n{'=' * 60}")
            print(f"  SESSION COMPLETE (Claude solo)")
            print(f"{'=' * 60}\n")
            self._save_log()
        else:
            self.run_session(opening_prompt=opening_prompt, max_rounds=max_rounds)

        # Extract JSON from the last Claude [SESSION COMPLETE] message
        for msg in reversed(self.messages):
            if msg["role"] == "Claude" and SESSION_COMPLETE in msg["content"]:
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
            "Check conversation log -- Claude may not have reached consensus."
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
