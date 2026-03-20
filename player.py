#!/usr/bin/env python3
"""
player.py -- ClankerBoy JSON Renderer for The Clankers 3

Plays ClankerBoy step-sequencer JSON (as defined in CLAUDE.md) through
the Clankers 3 VST band via DawDreamer.

Instrument routing (Clankers 2.0 v2):
  t:1  Buchla Systems   -> Buchla Systems.vst3
  t:2  Pro-One Bass     -> bassYnth Pro-One.vst3
  t:4  Voice            -> sample-based (skipped -- requires samples dir)
  t:5  Voder            -> formant engine (vocal_events array)
  t:6  HybridSynth Pads -> HybridSynth.vst3
  t:10 Drums            -> Antigravity Drums.vst3

CC values (0-127) are normalised to 0-1 and mapped to each VST's
APVTS parameter IDs, then applied as per-note automation arrays.

Usage:
    python player.py composition.json
    python player.py composition.json --out output/my_track.wav
    python player.py composition.json --loops 2
"""

import sys
import json
import argparse
import io
from pathlib import Path

import numpy as np
from pydub import AudioSegment

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

try:
    import dawdreamer as daw
    HAS_DAWDREAMER = True
except ImportError:
    HAS_DAWDREAMER = False

SAMPLE_RATE = 44100


# ── CC → APVTS parameter maps ──────────────────────────────────────────────────
# CC number (as string) -> normalized APVTS param ID for each instrument.
# CC values 0-127 are divided by 127 to get the normalized 0-1 APVTS value.

# t:1 Buchla Systems  (ParameterIDs.h)
BUCHLA_CC = {
    "74": "lpg_1_cutoff",
    "71": "lpg_1_resonance",
    "20": "osc_wavefold_amount",
    "17": "osc_mod_depth_fm",
    "19": "env_1_decay",
    "16": "osc_principal_level",
    "18": "osc_fm_index",
    "10": None,   # Pan -- handled at mix level, not APVTS
    "7":  "mix_master_level",
}

# t:2 bassYnth Pro-One  (PluginProcessor.cpp param IDs)
BASS_CC = {
    "74": "filterCutoff",
    "71": "filterRes",
    "73": "ampAttack",
    "75": "ampDecay",
    "79": "ampSustain",
    "72": "ampRelease",
    "18": "oscBTune",
    "23": "filterDecay",
    "22": "filterAttack",
    "24": "filterSustain",
    "25": "filterRelease",
    "26": "lfoRate",
    "27": "lfoAmount",
    "28": "lfoShape",
    "29": "lfoToFilter",
    "30": "lfoToPitch",
    "5":  "glide",
    "7":  None,   # Volume
    "64": None,   # Sustain pedal
}

# t:6 HybridSynth  (SynthAudioProcessor.cpp param IDs)
HYBRID_CC = {
    "74": "FILTER_CUTOFF",
    "71": "FILTER_RES",
    "73": "ENV1_ATTACK",
    "72": "ENV1_RELEASE",
    "75": "ENV1_DECAY",
    "79": "ENV1_SUSTAIN",
    "26": "LFO1_WAVE",
    "27": "LFO1_RATE",
    "28": "LFO1_AMOUNT",
    "29": "CHORUS_RATE",
    "30": "CHORUS_DEPTH",
    "31": "CHORUS_MIX",
    "85": "DELAY_TIME",
    "86": "DELAY_FEEDBACK",
    "87": "DELAY_MIX",
    "88": "REVERB_SIZE",
    "89": "REVERB_DAMPING",
    "91": "REVERB_MIX",
    "107": "granular_freeze",
    "103": "granular_size",
    "104": "granular_density",
    "105": "granular_texture",
    "106": "granular_spread",
    "102": None,   # Granular Pitch -- no direct param
    "19": "XMOD_FM",
    "20": "XMOD_RING",
    "10": None,   # Pan
    "7":  None,   # Volume
}

# Drums use note numbers only -- no CC map needed
DRUM_CC = {}

CC_MAPS = {
    1:  BUCHLA_CC,
    2:  BASS_CC,
    6:  HYBRID_CC,
    10: DRUM_CC,
}

VST_KEYS = {
    1:  "harmony_lead",   # Buchla Systems
    2:  "bass_sh101",     # bassYnth Pro-One
    6:  "harmony_pad",    # HybridSynth
    10: "drums",          # Antigravity Drums
}

INST_LABELS = {
    1: "buchla",
    2: "bass",
    6: "hybrid",
    10: "drums",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def beats_to_secs(beats: float, bpm: float) -> float:
    return beats * 60.0 / bpm


def _set_param(plugin, name: str, value: float) -> None:
    try:
        plugin.set_parameter(name, float(value))
    except Exception:
        pass


def _build_automation(events: list[tuple[float, float]], total_samples: int,
                      default: float = 0.5) -> np.ndarray:
    """
    Build a per-sample automation array from a list of (sample_offset, value) events.
    Between events the value holds until the next event (zero-order hold).
    """
    arr = np.full(total_samples, default, dtype=np.float32)
    for sample_offset, value in sorted(events):
        idx = int(np.clip(sample_offset, 0, total_samples - 1))
        arr[idx:] = float(value)
    return arr


# ── Step parser ────────────────────────────────────────────────────────────────

def parse_steps(composition: dict) -> dict:
    """
    Walk the ClankerBoy JSON steps and collect per-instrument event lists.

    Returns dict of { t_id: { "notes": [...], "cc_events": {param: [(sample, val), ...]} } }

    Each note: { "midi": int, "vel": int, "start_s": float, "dur_s": float }
    """
    bpm    = float(composition.get("bpm", 120))
    steps  = composition.get("steps", [])
    loops  = int(composition.get("_loops", 1))

    # Total duration of one pass
    total_beats_one = sum(float(s.get("d", 0.5)) for s in steps)
    one_pass_s      = beats_to_secs(total_beats_one, bpm)

    instruments: dict[int, dict] = {}

    def _get_inst(t):
        if t not in instruments:
            instruments[t] = {"notes": [], "cc_events": {}}
        return instruments[t]

    for loop_i in range(loops):
        loop_offset_s = loop_i * one_pass_s
        cursor_beats  = 0.0

        for step in steps:
            d      = float(step.get("d", 0.5))
            tracks = step.get("tracks", [])
            step_start_s = beats_to_secs(cursor_beats, bpm) + loop_offset_s

            for track in tracks:
                t       = int(track.get("t", 0))
                notes   = track.get("n", [])
                vel     = int(track.get("v", 100))
                cc_raw  = track.get("cc", {})
                dur_b   = track.get("dur")

                inst = _get_inst(t)
                cc_map = CC_MAPS.get(t, {})

                # CC automation events
                for cc_str, cc_val in cc_raw.items():
                    param = cc_map.get(str(cc_str))
                    if param is None:
                        continue
                    norm_val = float(cc_val) / 127.0
                    sample   = int(step_start_s * SAMPLE_RATE)
                    inst["cc_events"].setdefault(param, []).append((sample, norm_val))

                # Note events
                dur_s = beats_to_secs(float(dur_b) if dur_b is not None else d, bpm)
                for midi in notes:
                    inst["notes"].append({
                        "midi":    int(midi),
                        "vel":     vel,
                        "start_s": step_start_s,
                        "dur_s":   dur_s,
                    })

            cursor_beats += d

    total_s = one_pass_s * loops
    return instruments, total_s


# ── Per-instrument VST renderer ────────────────────────────────────────────────

def render_vst_instrument(t_id: int, inst_data: dict, total_s: float,
                          bpm: float) -> AudioSegment | None:
    """Render one instrument track through its VST via DawDreamer."""
    if not HAS_DAWDREAMER:
        print(f"  [skip] t:{t_id} -- dawdreamer not installed")
        return None

    vst_key  = VST_KEYS.get(t_id)
    vst_path = config.VST_PATHS.get(vst_key) if vst_key else None
    if not vst_path:
        print(f"  [skip] t:{t_id} -- no VST path configured")
        return None

    label    = INST_LABELS.get(t_id, f"t{t_id}")
    notes    = inst_data["notes"]
    cc_evts  = inst_data["cc_events"]
    total_n  = int(total_s * SAMPLE_RATE) + 1

    engine = daw.RenderEngine(SAMPLE_RATE, 512)
    plugin = engine.make_plugin_processor(label, vst_path)

    # Apply static CC params (first value for each param -- patch setup)
    for param, events in cc_evts.items():
        if events:
            _, first_val = min(events, key=lambda e: e[0])
            _set_param(plugin, param, first_val)

    # Apply per-note CC automation arrays for params that change over time
    for param, events in cc_evts.items():
        if len(events) > 1:
            # Find the default from the earliest event
            _, default_val = min(events, key=lambda e: e[0])
            arr = _build_automation(events, total_n, default=default_val)
            try:
                plugin.set_automation(param, arr)
            except Exception:
                pass   # fall back to static set_parameter already applied

    # Schedule MIDI notes
    for note in notes:
        midi    = int(np.clip(note["midi"], 0, 127))
        vel     = int(np.clip(note["vel"],  1, 127))
        start_s = float(np.clip(note["start_s"], 0.0, total_s - 0.001))
        dur_s   = float(np.clip(note["dur_s"],   0.01, total_s - start_s))
        plugin.add_midi_note(midi, vel, start_s, dur_s)

    engine.load_graph([(plugin, [])])
    engine.render(total_s)

    audio_np = engine.get_audio()
    if audio_np.ndim == 2:
        mono = (audio_np[0] + audio_np[1]) / 2
    else:
        mono = audio_np

    peak = np.max(np.abs(mono))
    if peak < 1e-9:
        print(f"  [empty] t:{t_id} {label} -- no audio produced")
        return None
    mono = mono / peak * 0.85
    pcm  = (mono * 32767).astype(np.int16).tobytes()
    seg  = AudioSegment(pcm, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    print(f"  [done]  t:{t_id:<3} {label:<10} {len(seg)/1000:.1f}s")
    return seg


# ── Mix ────────────────────────────────────────────────────────────────────────

# Per-instrument gain adjustments (dB) for a balanced mix
_INST_GAIN = {
    1:  -3,   # Buchla -- forward but not dominant
    2:  +2,   # Bass -- needs presence
    6:  -6,   # Pads -- sit back
    10:  0,   # Drums -- reference level
}


def mix_tracks(tracks: dict[int, AudioSegment], total_ms: int) -> AudioSegment:
    """Overlay all rendered instrument tracks into a single stereo mix."""
    output = AudioSegment.silent(duration=total_ms, frame_rate=SAMPLE_RATE)
    for t_id, seg in tracks.items():
        gain = _INST_GAIN.get(t_id, 0)
        seg  = seg + gain
        # Pad / trim to total_ms
        if len(seg) < total_ms:
            seg = seg + AudioSegment.silent(duration=total_ms - len(seg))
        else:
            seg = seg[:total_ms]
        output = output.overlay(seg)
    return output


# ── Main renderer ──────────────────────────────────────────────────────────────

def render(composition: dict, loops: int = 1) -> AudioSegment | None:
    """
    Render a ClankerBoy JSON composition dict through the Clankers 3 VSTs.
    Returns a mixed AudioSegment or None if nothing rendered.
    """
    composition["_loops"] = loops
    bpm = float(composition.get("bpm", 120))

    print(f"\n── CLANKERBOY PLAYER ────────────────────────────")
    print(f"BPM    : {bpm}")
    print(f"Steps  : {len(composition.get('steps', []))}  x{loops} loop(s)")
    print(f"DawDreamer: {'yes' if HAS_DAWDREAMER else 'NO -- VST rendering unavailable'}")

    instruments, total_s = parse_steps(composition)
    total_ms = int(total_s * 1000)
    print(f"Total  : {total_s:.2f}s  ({total_ms}ms)")
    print(f"Tracks : {sorted(instruments.keys())}\n")

    rendered: dict[int, AudioSegment] = {}

    for t_id, inst_data in sorted(instruments.items()):
        if t_id in (4, 5):
            print(f"  [skip] t:{t_id} -- voice/voder handled by their own engines")
            continue
        if t_id not in VST_KEYS:
            print(f"  [skip] t:{t_id} -- no routing defined")
            continue

        seg = render_vst_instrument(t_id, inst_data, total_s, bpm)
        if seg is not None:
            rendered[t_id] = seg

    if not rendered:
        print("  No audio produced.")
        return None

    print(f"\n  Mixing {len(rendered)} track(s)...")
    mixed = mix_tracks(rendered, total_ms)
    print(f"  Mix done -- {len(mixed)/1000:.1f}s")
    return mixed


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ClankerBoy JSON Renderer -- The Clankers 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python player.py beat.json
  python player.py beat.json --out output/my_beat.wav
  python player.py beat.json --loops 4 --out output/loop4.wav
        """,
    )
    parser.add_argument("json_file", help="ClankerBoy JSON file to render")
    parser.add_argument("--out",   default=None,  help="Output WAV path (default: <json_name>.wav)")
    parser.add_argument("--loops", default=1, type=int, help="Number of times to loop (default: 1)")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[error] File not found: {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        composition = json.load(f)

    out_path = Path(args.out) if args.out else json_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = render(composition, loops=args.loops)
    if result is None:
        print("[player] No audio rendered.")
        sys.exit(1)

    result.export(str(out_path), format="wav")
    print(f"\n── OUTPUT ───────────────────────────────────────")
    print(f"  {out_path}  ({len(result)/1000:.1f}s)")
    print(f"──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
