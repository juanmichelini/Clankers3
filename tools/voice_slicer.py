import os
import re
import argparse
from pathlib import Path
import whisperx
import torch
from pydub import AudioSegment
from pydub.effects import normalize

SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

TARGET_DBFS  = -20.0  # target RMS loudness for each phrase sample
FADE_MS      = 5      # fade-in/out length to eliminate click artifacts
MAX_WORDS    = 6      # hard cap: flush phrase if it reaches this many words
TARGET_WORDS = 4      # soft target when splitting an oversized phrase
PUNCT_FLUSH  = set('.,;:!?')  # punctuation characters that end a phrase


def normalize_word(seg: AudioSegment) -> AudioSegment:
    """Normalize RMS to TARGET_DBFS and apply short fade envelopes."""
    diff = TARGET_DBFS - seg.dBFS
    seg = seg.apply_gain(diff)
    return seg.fade_in(FADE_MS).fade_out(FADE_MS)


def clean_token(text):
    """Return (clean_text, has_punct) -- strip punctuation and numbers, lowercase."""
    has_punct = any(c in PUNCT_FLUSH for c in text)
    clean = re.sub(r'[^a-z]', '', text.lower()).strip()
    return clean, has_punct


def flush_phrase(words, audio, padding_ms, phrase_counts, output_dir):
    """
    Export one phrase (list of WhisperX word_info dicts) as a single .wav file.
    Returns the filename used, or None if skipped.
    """
    if not words:
        return None

    phrase_text = '_'.join(
        re.sub(r'[^a-z]', '', w.get("word", "").lower()).strip()
        for w in words
        if re.sub(r'[^a-z]', '', w.get("word", "").lower()).strip()
    )
    if not phrase_text:
        return None

    start_ms = int(words[0]["start"] * 1000)
    end_ms   = int(words[-1]["end"]   * 1000)
    phrase_audio = audio[max(0, start_ms - padding_ms): end_ms + padding_ms]
    phrase_audio = normalize_word(phrase_audio)

    phrase_counts[phrase_text] = phrase_counts.get(phrase_text, 0) + 1
    idx = phrase_counts[phrase_text]
    filename = f"{phrase_text}.wav" if idx == 1 else f"{phrase_text}_{idx}.wav"

    phrase_audio.export(os.path.join(output_dir, filename), format="wav")
    return filename


def slice_audio(input_file, output_dir, model_name="base", padding_ms=30):
    """
    Transcribe and slice sentence-like phrases from an audio file using WhisperX.

    Slicing rules (Option A):
      1. Flush the current phrase at any punctuation boundary (. , ; : ! ?).
      2. If a phrase reaches MAX_WORDS without hitting punctuation, flush at
         the nearest word boundary targeting TARGET_WORDS words.
      3. Skip phrases with fewer than 2 clean words.
    """
    print(f"\nProcessing '{input_file}'...")
    print(f"Device: {DEVICE} | Compute: {COMPUTE_TYPE}")

    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"Error loading {input_file} with pydub: {e}")
        return

    # Step 1: Load WhisperX model and transcribe
    print(f"Loading WhisperX model '{model_name}'...")
    model = whisperx.load_model(model_name, DEVICE, compute_type=COMPUTE_TYPE, language="en")

    print("Transcribing... (this may take a moment)")
    audio_whisperx = whisperx.load_audio(str(input_file))
    result = model.transcribe(audio_whisperx, batch_size=16 if DEVICE == "cuda" else 4)

    # Step 2: Force-align with wav2vec2 for precise word boundaries
    print("Force-aligning words with wav2vec2...")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    result = whisperx.align(result["segments"], align_model, align_metadata, audio_whisperx, DEVICE, return_char_alignments=False)

    # Free GPU memory
    del model
    del align_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)

    phrase_counts  = {}
    exported_count = 0
    buffer         = []   # accumulated word_info dicts for current phrase

    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            raw_text = word_info.get("word", "")
            clean, has_punct = clean_token(raw_text)

            if not clean:
                continue  # skip noise / empty tokens

            # Skip words without timing info (can't slice without boundaries)
            if word_info.get("start") is None or word_info.get("end") is None:
                continue

            buffer.append(word_info)

            # Rule 1 -- punctuation boundary: flush immediately
            if has_punct:
                if len(buffer) >= 2:
                    if flush_phrase(buffer, audio, padding_ms, phrase_counts, output_dir):
                        exported_count += 1
                buffer = []
                continue

            # Rule 2 -- hard word cap reached: split at TARGET_WORDS, keep rest
            if len(buffer) >= MAX_WORDS:
                to_flush = buffer[:TARGET_WORDS]
                buffer   = buffer[TARGET_WORDS:]
                if len(to_flush) >= 2:
                    if flush_phrase(to_flush, audio, padding_ms, phrase_counts, output_dir):
                        exported_count += 1

    # Flush any remaining words at end of file
    if len(buffer) >= 2:
        if flush_phrase(buffer, audio, padding_ms, phrase_counts, output_dir):
            exported_count += 1

    print(f"\nDone! Exported {exported_count} phrase clips.")


def process_directory(directory, output_dir, model_name, padding_ms):
    """Process all supported audio files in a directory."""
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED_FORMATS:
                slice_audio(os.path.join(root, fname), output_dir, model_name, padding_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice Slicer: Cut audio into sentence phrases using WhisperX forced alignment."
    )
    parser.add_argument("input_path", help="Audio file or directory to process")
    parser.add_argument("--output_dir", default="samples/raw", help="Output directory for .wav phrase files")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--padding", type=int, default=30, help="Padding in ms around each phrase (default: 30)")
    args = parser.parse_args()

    path = Path(args.input_path)
    if path.is_file():
        slice_audio(str(path), args.output_dir, args.model, args.padding)
    elif path.is_dir():
        process_directory(str(path), args.output_dir, args.model, args.padding)
    else:
        print(f"Error: Invalid input path '{args.input_path}'")
