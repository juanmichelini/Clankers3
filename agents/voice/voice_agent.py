"""
The Clankers 2.0 -- Voice Agent (SAMPLER)
Reads a Music Sheet JSON, selects samples from the character library,
stitches them into a voice performance, outputs audio.

Library structure (output of voice_slicer.py):
  samples/profiles/
    ranter/
      stop.wav, stop_2.wav, nobody.wav ...   <- word samples
      phrases/
        its_still_there.wav                  <- phrase samples
    witness/
    ghost/
    child/
"""

import os
import re
import json
import random
import requests
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path(__file__).parent.parent.parent / "samples" / "profiles"

CHARACTERS = {
    "ranter":  "ranter",
    "witness": "witness",
    "ghost":   "ghost",
    "child":   "child",
}

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Max words used when building a phrase key (must match voice_slicer.py)
MAX_PHRASE_WORDS = 8


# ─── LIBRARY LOADERS ──────────────────────────────────────────────────────────

def load_library(character: str) -> dict[str, list[Path]]:
    """
    Scan character folder and return dict of word -> [file paths].
    Handles duplicates: stop.wav, stop_2.wav -> all under "stop".
    """
    folder = SAMPLE_DIR / character
    if not folder.exists():
        print(f"[warn] Word library not found: {folder}")
        return {}

    library = {}
    for f in folder.glob("*.wav"):
        base = re.sub(r'_\d+$', '', f.stem)
        library.setdefault(base, []).append(f)

    return library


def load_phrase_library(character: str) -> dict[str, list[Path]]:
    """
    Scan character/phrases/ folder and return dict of phrase_key -> [file paths].
    e.g. 'its_still_there' -> [Path('its_still_there.wav'), Path('its_still_there_2.wav')]
    Returns empty dict if the phrases folder doesn't exist yet.
    """
    folder = SAMPLE_DIR / character / "phrases"
    if not folder.exists():
        return {}

    library = {}
    for f in folder.glob("*.wav"):
        base = re.sub(r'_\d+$', '', f.stem)
        library.setdefault(base, []).append(f)

    return library


def _clean_phrase_key(text: str) -> str:
    """
    Convert a hint string to the same key format used by voice_slicer.py.
    'It's still there!' -> 'its_still_there'
    """
    text = re.sub(r"[^\w\s]", '', text).strip().lower()
    words = text.split()[:MAX_PHRASE_WORDS]
    return '_'.join(words)


# ─── WORD RESOLUTION ──────────────────────────────────────────────────────────

def find_word(word: str, library: dict) -> Path | None:
    """Direct lookup -- pick random variant if multiple exist."""
    clean = re.sub(r'[^\w]', '', word).lower()
    variants = library.get(clean)
    if variants:
        return random.choice(variants)
    return None


def llm_describe_word(word: str, context_phrase: str, available_words: list[str]) -> list[str]:
    """
    Ask the LLM for the closest available words to a missing word.
    Returns ordered list of candidates.
    """
    available_sample = random.sample(available_words, min(200, len(available_words)))

    prompt = f"""A voice sample library is missing the word "{word}".
This word appears in the phrase: "{context_phrase}"

Available words in the library: {', '.join(available_sample)}

Return a JSON array of up to 5 words from the available list that best approximate
the meaning or function of "{word}" IN THE CONTEXT of that phrase. Closest match first.
Return ONLY the JSON array, nothing else. Example: ["afraid", "alone", "dark"]"""

    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        text = response.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        candidates = json.loads(text)
        return [c.lower() for c in candidates if isinstance(c, str)]
    except Exception as e:
        print(f"  [llm error] {e}")
        return []


def resolve_word(word: str, context_phrase: str, library: dict) -> tuple[Path | None, str]:
    """
    Try direct match, then LLM fallback.
    Returns (path, method) where method is 'direct', 'llm', or 'missing'.
    """
    path = find_word(word, library)
    if path:
        return path, "direct"

    print(f"  [missing] '{word}' (in '{context_phrase}') -- asking LLM...")
    available = list(library.keys())
    candidates = llm_describe_word(word, context_phrase, available)

    for candidate in candidates:
        path = find_word(candidate, library)
        if path:
            print(f"  [llm match] '{word}' -> '{candidate}'")
            return path, "llm"

    print(f"  [unresolved] '{word}' -- no match found")
    return None, "missing"


# ─── SEGMENT LOADER ───────────────────────────────────────────────────────────

def _load_seg(path: Path) -> AudioSegment:
    """Load an audio file, apply soft fades and normalize."""
    seg = AudioSegment.from_file(path)
    fade_ms = min(15, len(seg) // 4)
    if fade_ms > 0:
        seg = seg.fade_in(fade_ms).fade_out(fade_ms)
    return normalize(seg)


# ─── SEQUENCE BUILDER ─────────────────────────────────────────────────────────

def hints_to_words(hints: list[str]) -> list[str]:
    """Flatten hint phrases into individual words (used for stats reporting)."""
    words = []
    for hint in hints:
        words.extend(hint.lower().split())
    return words


def build_sequence(
    hints: list[str],
    word_library: dict,
    phrase_library: dict,
) -> list[tuple[AudioSegment, str]]:
    """
    Build a playback sequence from the Music Sheet hints.

    Resolution priority per hint:
      1. Direct phrase match  -- the whole hint matches a phrase file
      2. Word-by-word         -- fall back to individual word samples (with LLM assist)

    Returns a list of (AudioSegment, label) tuples.
    """
    sequence = []

    for hint in hints:
        # 1. Try whole-hint phrase match
        if phrase_library:
            key = _clean_phrase_key(hint)
            variants = phrase_library.get(key)
            if variants:
                path = random.choice(variants)
                seg = _load_seg(path)
                sequence.append((seg, key))
                print(f"  [phrase] '{hint}' -> {path.name}")
                continue

        # 2. Word-by-word fallback
        words = hint.lower().split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if not clean:
                continue
            path, method = resolve_word(clean, hint, word_library)
            if path:
                seg = _load_seg(path)
                sequence.append((seg, clean))

    return sequence


# ─── CHARACTER SELECTION ──────────────────────────────────────────────────────

def pick_character(sheet: dict) -> str:
    mood = sheet.get("mood", "").lower()
    instruction = sheet["agents"]["sampler"].get("instruction", "").lower()
    combined = mood + " " + instruction

    if any(w in combined for w in ["paranoid", "rant", "dark mental", "intrusive", "obsessive"]):
        return "ranter"
    elif any(w in combined for w in ["cold", "detach", "observe", "clinical"]):
        return "witness"
    elif any(w in combined for w in ["ghost", "drift", "fragment", "memory", "distant"]):
        return "ghost"
    elif any(w in combined for w in ["child", "simple", "innocent", "tender"]):
        return "child"
    return "ranter"


# ─── TIMING ───────────────────────────────────────────────────────────────────

def bars_to_ms(bars: int, bpm: int) -> int:
    return int(bars * 4 * (60000 / bpm))


CROSSFADE_MS = 10


def random_fill(
    word_library: dict,
    phrase_library: dict,
    count: int,
) -> list[tuple[AudioSegment, str]]:
    """
    Pick random clips to fill the timeline when no hints resolved.
    Prefers phrase library if available; mixes in words otherwise.
    """
    result = []

    if phrase_library:
        keys = list(phrase_library.keys())
        chosen = random.choices(keys, k=count)
        for key in chosen:
            path = random.choice(phrase_library[key])
            seg = _load_seg(path)
            result.append((seg, key))
    elif word_library:
        keys = list(word_library.keys())
        chosen = random.choices(keys, k=count)
        for key in chosen:
            path = random.choice(word_library[key])
            seg = _load_seg(path)
            result.append((seg, key))

    return result


def build_performance(
    sequence: list[tuple[AudioSegment, str]],
    hints: list[str],
    density: str,
    total_ms: int,
    bpm: int = 120,
    swing: float = 0.0,
    humanize: bool = True,
) -> AudioSegment:
    """
    Place clips across the timeline.
    Timing is BPM-aware, with optional swing and humanization.
    Cycles through the sequence to fill the full duration.
    """
    output = AudioSegment.silent(duration=total_ms)
    if not sequence:
        return output

    beat_ms   = 60000 / bpm
    sixteenth = beat_ms / 4

    phrase_gap_bars = {
        "sparse":   random.uniform(1.0, 2.0),
        "medium":   random.uniform(0.5, 1.0),
        "dense":    random.uniform(0.25, 0.5),
        "high":     random.uniform(0.15, 0.35),
        "very low": random.uniform(2.0, 4.0),
    }.get(density, 1.0)
    phrase_gap_ms = phrase_gap_bars * 4 * beat_ms

    cursor = beat_ms * 4
    if humanize:
        cursor += random.uniform(-beat_ms * 0.5, beat_ms * 0.5)
    cursor = max(0, cursor)

    sixteenth_counter = 0
    seq_index = 0
    import time

    while cursor < total_ms:
        if seq_index > 0 and seq_index % 24 == 0:
            time.sleep(0)  # yield GIL so GUI stays responsive
        seg, label = sequence[seq_index % len(sequence)]
        seq_index += 1

        seg = seg.fade_in(CROSSFADE_MS).fade_out(CROSSFADE_MS)

        position = cursor
        if swing > 0 and sixteenth_counter % 2 == 1:
            position += sixteenth * swing
        if humanize:
            position += random.uniform(-beat_ms * 0.05, beat_ms * 0.05)

        position = max(0, min(int(position), total_ms - 1))
        output = output.overlay(seg, position=position)

        cursor += len(seg) + phrase_gap_ms
        sixteenth_counter += 1

    return output


# ─── EFFECTS ──────────────────────────────────────────────────────────────────

def apply_effects(audio: AudioSegment, instruction: str) -> AudioSegment:
    instruction = instruction.lower()

    if "pitch" in instruction or "shift" in instruction:
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.85)
        }).set_frame_rate(audio.frame_rate)

    if "glitch" in instruction or "stutter" in instruction:
        chunk = audio[:250]
        audio = chunk * 2 + audio

    if "slow" in instruction:
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.75)
        }).set_frame_rate(audio.frame_rate)

    if "reverse" in instruction:
        audio = audio.reverse()

    return normalize(audio)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(sheet: dict, output_path: str = "voice_output.wav") -> AudioSegment | None:
    sampler = sheet["agents"]["sampler"]

    if not sampler.get("active", False):
        print("Sampler inactive on this sheet.")
        return None

    bpm         = sheet.get("bpm", 120)
    bars        = sheet.get("bars", 8)
    density     = sampler.get("density", "sparse")
    hints       = sampler.get("sampleHints", [])
    instruction = sampler.get("instruction", "")
    swing       = sampler.get("swing", 0.0)
    humanize    = sampler.get("humanize", True)

    total_ms  = bars_to_ms(bars, bpm)
    character = pick_character(sheet)

    print(f"\n── VOICE AGENT ──────────────────────")
    print(f"Character : {CHARACTERS.get(character, character)}")
    print(f"Duration  : {total_ms}ms  ({bars} bars @ {bpm} bpm)")
    print(f"Density   : {density}  swing={swing}  humanize={humanize}")
    print(f"Hints     : {hints}")

    word_library   = load_library(character)
    phrase_library = load_phrase_library(character)

    print(f"Words     : {len(word_library)} unique  |  "
          f"Phrases: {len(phrase_library)} unique  ('{character}/')")

    sequence = build_sequence(hints, word_library, phrase_library)
    print(f"Resolved  : {len(sequence)} clips from {len(hints)} hints")

    if not sequence:
        fill_count = {"sparse": 6, "medium": 12, "dense": 24,
                      "high": 32, "very low": 3}.get(density, 8)
        sequence = random_fill(word_library, phrase_library, fill_count)
        print(f"  [fill] No hints resolved -- using {len(sequence)} random clips")

    performance = build_performance(
        sequence, hints, density, total_ms,
        bpm=bpm, swing=swing, humanize=humanize,
    )
    performance = apply_effects(performance, instruction)

    performance.export(output_path, format="wav")
    print(f"Output    : {output_path}")
    print(f"────────────────────────────────────\n")

    return performance


# ─── EXAMPLE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_sheet = {
        "title": "Observation Alpha",
        "bpm": 100,
        "key": "A minor",
        "bars": 16,
        "mood": "cold, clinical, static, surveillance",
        "structure": "intro",
        "timeSignature": "4/4",
        "agents": {
            "sampler": {
                "active": True,
                "density": "sparse",
                "instruction": "clean, unpitched, slightly delayed, isolated",
                "sampleHints": [
                    "subject is stationary",
                    "temperature dropping",
                    "two figures approaching",
                    "perimeter secured",
                    "anomaly recorded"
                ]
            }
        }
    }

    run(example_sheet, output_path="voice_output.wav")
