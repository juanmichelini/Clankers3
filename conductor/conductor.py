# conductor/conductor.py -- The Clankers 3
#
# Full pipeline:
#   run_track(brief, arc, out_dir)
#     1. Chatroom negotiates verse1 -> Music Sheet JSON
#     2. For each subsequent section: evolve() mutates the sheet
#     3. run_session() fires all agents in parallel (DawDreamer or numpy)
#     4. mixer.mix_section() balances + EQs each section
#     5. mixer.stitch_and_master() concatenates + compresses -> full_track.wav

import io
import os
import re
import sys
import json
import copy
import threading
import requests
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from chatroom.chatroom import Chatroom

# ── Agent imports ─────────────────────────────────────────────────────────

from agents.voice.voice_agent    import run as run_voice
from agents.bassline.bass_sh101  import run as run_sh101

try:
    from agents.drums.drums_agent   import run as run_drums
    HAS_DRUMS = True
except ImportError:
    HAS_DRUMS = False

try:
    from agents.harmony.harmony_agent import run as run_harmony
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False

try:
    from agents.voder.voder_agent   import run as run_voder
    HAS_VODER = True
except ImportError:
    HAS_VODER = False

# ── Per-thread stdout router ──────────────────────────────────────────────
# Agent threads write to their own log file; main/worker thread reaches GUI.

_tls = threading.local()


class _AgentStream(io.TextIOBase):
    def __init__(self, default_stream):
        self._default = default_stream

    def write(self, s: str) -> int:
        dest = getattr(_tls, "stream", None) or self._default
        dest.write(s)
        return len(s)

    def flush(self):
        dest = getattr(_tls, "stream", None) or self._default
        try:
            dest.flush()
        except Exception:
            pass


# ── Arc ───────────────────────────────────────────────────────────────────

DEFAULT_ARC = ["verse1", "instrumental", "verse2", "bridge", "verse3", "outro"]

# All stem names that can appear in a run_session() result
ALL_STEMS = ["sampler", "bass_sh101", "drums", "buchla", "hybrid", "voder"]

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
EVOLVE_MODEL   = "gpt-4o"   # stable model for evolve() -- independent of config.CHATGPT_MODEL

# Canonical tension curve per section name.
# Agents use this to scale density, velocity, fill probability, etc.
_SECTION_TENSION: dict[str, float] = {
    "verse1":       0.30,
    "instrumental": 0.45,
    "verse2":       0.52,
    "bridge":       0.75,
    "verse3":       0.85,
    "outro":        0.20,
}

# ── EVOLVE ────────────────────────────────────────────────────────────────

_EVOLVE_SYSTEM = """You are the music evolution engine for The Clankers 3.
You receive a Music Sheet JSON and a target section name.
Return a mutated version that fits the new section's energy and arc.
Preserve title and bpm (bpm is ALWAYS locked -- never change it).
Key can modulate at bridge only.

Section arc guidance:
  verse1       -- full band enters, establish core groove, sampler + voder introduce themes
  instrumental -- sampler goes inactive, instruments step forward, bass and drums develop
  verse2       -- sampler returns, revisit theme with twist, fuller texture
  bridge       -- hard contrast: strip back or flip density, harmonic surprise, key can shift
  verse3       -- climax: all agents fully active, maximum density, voder most expressive
  outro        -- dissolution, elements drop one by one, density falling, voder last to leave

The band has ONE bass voice: bass_sh101 (Pro-One style). There is no bass303.
Mutate: bars, mood, density, sampleHints, patterns, active agents.
Do NOT change bpm -- it is locked for the entire track.
Always rewrite globalNotes to reflect new inter-agent coordination for this section.

CRITICAL -- always include these fields in the mutated sheet:
  bars: target 16 bars per section for a developed, longer composition.
    Outro may use 8 bars to resolve quickly. Bridge may use 8-12 bars for contrast.
  tension (0.0-1.0): section energy driver. verse1≈0.3, instrumental≈0.45, verse2≈0.5,
    bridge≈0.75, verse3≈0.85, outro≈0.2
  harmonic_rhythm: "slow"|"medium"|"fast"|"mixed" -- rate of chord changes.
    Match to tension: slow at low tension, fast/mixed at peak.
  harmonic_map: array of {bar, root_degree, chord_name} for every bar in the section.
    Bass and drums use this to lock onto chord changes structurally.
    root_degree is the scale degree of the chord root (1=tonic, 4=subdominant, 5=dominant, etc.)
  bass_sh101.swing: 0.0-0.5 -- add swing feel appropriate to the genre/mood.
  harmony.synth.buchla / harmony.synth.hybrid: full ADSR envelope available --
    attack_s, decay_s (0=skip), sustain (0.0-1.0), release_s.
    sustain=1.0 + decay_s=0 = held texture pad; sustain=0.0 + short decay_s = pluck/transient.
  globalNotes: MUST include specific beat-level locks between bass and kick,
    register boundaries (bass below C3, pads above C4, etc.), and a 2-bar transition
    tail instruction so sections blend smoothly into each other.

Return ONLY valid JSON. No explanation."""


def evolve(sheet: dict, next_section: str, api_key: str | None = None) -> dict:
    """Mutate a Music Sheet for the next section. BPM is locked."""
    api_key = api_key or config.OPENAI_API_KEY
    current = copy.deepcopy(sheet)
    current["structure"] = next_section

    prompt = (
        f"Current sheet:\n{json.dumps(current, indent=2)}\n\n"
        f"Target section: {next_section}\n\nReturn the mutated sheet JSON."
    )

    try:
        resp = requests.post(
            OPENAI_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": EVOLVE_MODEL,
                "max_tokens": 1400,
                "messages": [
                    {"role": "system", "content": _EVOLVE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=45,
        )
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        evolved = json.loads(text)
        evolved["bpm"] = sheet["bpm"]   # enforce BPM lock
        # Enforce canonical tension so all agents have a consistent energy signal
        canonical_tension = _SECTION_TENSION.get(next_section)
        if canonical_tension is not None:
            evolved["tension"] = canonical_tension
        print(f"  Evolved -> bpm={evolved.get('bpm')} tension={evolved.get('tension', '?')} | mood={evolved.get('mood', '')[:60]}")
        return evolved
    except Exception as e:
        print(f"  [evolve error] {e} -- keeping current sheet")
        return current


# ── SESSION RUNNER ────────────────────────────────────────────────────────

def run_session(
    sheet:         dict,
    out_dir:       str = "output",
    section_label: str = "",
    disable:       list[str] | None = None,
) -> dict[str, AudioSegment]:
    """
    Fire all active agents in parallel.
    Each agent: reads the shared Music Sheet -> LLM sequence call -> synthesize audio.
    Returns { agent_name: AudioSegment }.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prefix   = f"{section_label}_" if section_label else ""
    agents   = sheet.get("agents", {})
    disabled = set(disable or [])
    ant_key  = config.ANTHROPIC_API_KEY

    tasks: dict[str, callable] = {}

    if "sampler" not in disabled and agents.get("sampler", {}).get("active"):
        _p = str(Path(out_dir) / f"{prefix}voice.wav")
        tasks["sampler"] = lambda p=_p: run_voice(sheet, output_path=p)

    if "bass_sh101" not in disabled and agents.get("bass_sh101", {}).get("active"):
        _p = str(Path(out_dir) / f"{prefix}bass_sh101.wav")
        tasks["bass_sh101"] = lambda p=_p: run_sh101(
            sheet, output_path=p, api_key=ant_key,
            vst_path=config.VST_PATHS.get("bass_sh101"))

    if "drums" not in disabled and HAS_DRUMS and agents.get("drums", {}).get("active"):
        _p = str(Path(out_dir) / f"{prefix}drums.wav")
        tasks["drums"] = lambda p=_p: run_drums(
            sheet, output_path=p, api_key=ant_key,
            vst_path=config.VST_PATHS.get("drums"))

    # "harmony" disables both sub-tracks; "buchla"/"hybrid" disable individually.
    # Only run the harmony agent if at least one sub-track is still active.
    _harmony_active = (
        HAS_HARMONY
        and agents.get("harmony", {}).get("active")
        and "harmony" not in disabled
        and not ("buchla" in disabled and "hybrid" in disabled)
    )
    if _harmony_active:
        # base path: agent derives _buchla.wav and _hybrid.wav from it
        _p = str(Path(out_dir) / f"{prefix}harmony.wav")
        tasks["harmony"] = lambda p=_p: run_harmony(
            sheet, output_path=p, api_key=ant_key,
            vst_path=config.VST_PATHS.get("harmony_lead"),
            vst_path_pad=config.VST_PATHS.get("harmony_pad"))

    if "voder" not in disabled and HAS_VODER and agents.get("voder", {}).get("active"):
        _p = str(Path(out_dir) / f"{prefix}voder.wav")
        tasks["voder"] = lambda p=_p: run_voder(
            sheet, output_path=p, api_key=ant_key)

    if not tasks:
        return {}

    # Route agent thread output to per-agent log files
    gui_out    = sys.stdout
    old_stdout = sys.stdout
    sys.stdout = _AgentStream(gui_out)
    gui_out.write(f"  agents -> {', '.join(tasks.keys())}\n")

    def _wrap(name: str, fn):
        log_path = Path(out_dir) / f"{name}.log"
        def _run():
            gui_out.write(f"  [start] {name}\n")
            with open(log_path, "w", encoding="utf-8") as lf:
                _tls.stream = lf
                try:
                    return fn()
                finally:
                    _tls.stream = None
        return _run

    results: dict[str, AudioSegment] = {}

    try:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            future_to_name = {pool.submit(_wrap(n, fn)): n for n, fn in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    audio = future.result()
                    if audio is None:
                        gui_out.write(f"  [empty] {name}\n")
                    elif name == "harmony" and isinstance(audio, dict):
                        # harmony_agent returns {"buchla": seg, "hybrid": seg}
                        # Honour individual sub-track disable flags
                        for sub_name, sub_audio in audio.items():
                            if sub_name in disabled:
                                gui_out.write(f"  [skip]  {sub_name:<12} (disabled)\n")
                            elif sub_audio:
                                results[sub_name] = sub_audio
                                gui_out.write(f"  [done]  {sub_name:<12} {len(sub_audio)/1000:.1f}s\n")
                    elif audio:
                        results[name] = audio
                        gui_out.write(f"  [done]  {name:<12} {len(audio)/1000:.1f}s\n")
                    else:
                        gui_out.write(f"  [empty] {name}\n")
                except Exception as exc:
                    gui_out.write(f"  [error] {name}: {exc}\n")
    finally:
        sys.stdout = old_stdout

    return results


# ── FULL TRACK ────────────────────────────────────────────────────────────

def run_track(
    brief:      str,
    arc:        list[str] | None = None,
    out_dir:    str | Path = "output",
    disable:    list[str] | None = None,
    single_llm: bool = False,
) -> AudioSegment | None:
    """
    Full pipeline:
      1. Chatroom negotiates verse1 -> initial Music Sheet
      2. For each section: evolve() then run_session()
      3. mixer.mix_section() per section
      4. mixer.stitch_and_master() -> full_track.wav
    """
    from mixer.mixer import mix_section, stitch_and_master, stitch_stem

    arc     = arc or DEFAULT_ARC
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  THE CLANKERS 3 -- FULL TRACK")
    print(f"  Brief : {brief}")
    print(f"  Arc   : {' -> '.join(arc)}")
    print(f"{'=' * 62}\n")

    # ── Step 1: chatroom negotiates the opening section ────────────────
    print("Step 1: Chatroom negotiating opening section...\n")
    room  = Chatroom(session_name=arc[0])
    sheet = room.negotiate_section(brief=brief, section_name=arc[0], solo=single_llm)

    # Inject canonical tension for the opening section if chatroom didn't set it
    if "tension" not in sheet:
        sheet["tension"] = _SECTION_TENSION.get(arc[0], 0.35)

    # Save initial sheet
    with open(out_dir / "sheet_initial.json", "w") as f:
        json.dump(sheet, f, indent=2)

    sections: list[AudioSegment] = []

    # Per-stem section lists (all stems, padded with silence when inactive)
    stem_sections:  dict[str, list] = {name: [] for name in ALL_STEMS}
    stem_had_audio: dict[str, bool] = {name: False for name in ALL_STEMS}

    # ── Step 2-4: walk the arc ─────────────────────────────────────────
    for i, section in enumerate(arc):
        import time as _t; _t.sleep(0.05)   # yield for GUI

        if i > 0:
            print(f"\n  Evolving -> {section.upper()}...")
            sheet = evolve(sheet, section)
            with open(out_dir / f"sheet_{section}.json", "w") as f:
                json.dump(sheet, f, indent=2)

        print(f"\n-- {section.upper()} " + "-" * 44)
        tracks = run_session(sheet, out_dir=str(out_dir), section_label=section,
                             disable=disable)

        if not tracks:
            print(f"  [skip] no active agents produced audio for {section}")
            continue

        mixed = mix_section(tracks, sheet)
        if mixed:
            path = out_dir / f"{section}_mix.wav"
            mixed.export(str(path), format="wav")
            print(f"  Mix -> {path}  ({len(mixed)/1000:.1f}s)")
            sections.append(mixed)

            # Collect per-stem audio; pad with silence if agent was silent this section
            section_dur_ms = len(mixed)
            for stem_name in ALL_STEMS:
                if stem_name in tracks:
                    stem_sections[stem_name].append(tracks[stem_name])
                    stem_had_audio[stem_name] = True
                else:
                    stem_sections[stem_name].append(
                        AudioSegment.silent(duration=section_dur_ms, frame_rate=44100)
                    )

    if not sections:
        print("No audio produced.")
        return None

    # ── Step 5a: export individual stems (time-aligned, no master compression)
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    print("\n  Exporting stems...")
    exported_stems = []
    for stem_name in ALL_STEMS:
        if not stem_had_audio.get(stem_name):
            continue   # agent never produced audio -- skip
        segs = stem_sections[stem_name]
        if not segs:
            continue
        try:
            full_stem = stitch_stem(segs)
            stem_path = stems_dir / f"{stem_name}.wav"
            full_stem.export(str(stem_path), format="wav")
            print(f"  Stem -> {stem_name}.wav  ({len(full_stem)/1000:.1f}s)")
            exported_stems.append(stem_name)
        except Exception as e:
            print(f"  [stem error] {stem_name}: {e}")
    if exported_stems:
        print(f"  Stems dir: {stems_dir}")

    # ── Step 5b: stitch + master ────────────────────────────────────────
    print("\n  Assembling + mastering full track...")
    full_track = stitch_and_master(sections)

    track_path = out_dir / "full_track.wav"
    full_track.export(str(track_path), format="wav")
    print(f"\n{'=' * 62}")
    print(f"  DONE: {track_path}  ({len(full_track)/1000:.1f}s)")
    print(f"{'=' * 62}\n")
    return full_track


# ── SOLO COMPANION ────────────────────────────────────────────────────────

def run_solo(
    brief:   str,
    agent:   str,
    section: str = "verse1",
    out_dir: str | Path = "output",
) -> AudioSegment | None:
    """
    Run a single companion in isolation.
    Negotiates a music sheet for the brief, then activates only the target agent.

    Example:
        run_solo("dreamy buchla melodies, slow evolving", "harmony")
        run_solo("heavy kick, sparse snare", "drums")
    """
    from mixer.mixer import mix_section

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  THE CLANKERS 3 — SOLO: {agent.upper()}")
    print(f"  Brief  : {brief}")
    print(f"  Section: {section}")
    print(f"{'=' * 62}\n")

    # Negotiate a full music sheet, then isolate the target agent
    print("Negotiating music sheet...\n")
    room  = Chatroom(session_name=section)
    sheet = room.negotiate_section(brief=brief, section_name=section)

    agents_block = sheet.setdefault("agents", {})
    for name in list(agents_block.keys()):
        agents_block[name]["active"] = (name == agent)
    if agent not in agents_block:
        agents_block[agent] = {"active": True}

    with open(out_dir / f"sheet_solo_{agent}.json", "w") as f:
        json.dump(sheet, f, indent=2)

    print(f"\n-- SOLO: {agent.upper()} " + "-" * 40)
    tracks = run_session(sheet, out_dir=str(out_dir), section_label=f"solo_{agent}")

    if not tracks:
        print(f"  [solo] no audio produced for {agent}")
        return None

    mixed = mix_section(tracks, sheet)
    if mixed:
        path = out_dir / f"solo_{agent}.wav"
        mixed.export(str(path), format="wav")
        print(f"\n  DONE: {path}  ({len(mixed)/1000:.1f}s)")
        print(f"{'=' * 62}\n")
        return mixed

    return None


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The Clankers 3 -- Conductor")
    parser.add_argument("brief", nargs="?",
                        default="dark industrial EBM, cold acid bass, mechanical drums")
    parser.add_argument("--arc", nargs="+", default=None,
                        help=f"Section arc (default: {' '.join(DEFAULT_ARC)})")
    parser.add_argument("--out", default="output")
    parser.add_argument("--disable", nargs="+", default=None)
    args = parser.parse_args()

    run_track(brief=args.brief, arc=args.arc, out_dir=args.out, disable=args.disable)
