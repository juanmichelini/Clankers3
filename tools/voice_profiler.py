import os
import re
import shutil
import json
import argparse
import requests

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# The Clankers 2.0 -- default character profiles
DEFAULT_PROFILES = ["ranter", "witness", "ghost", "child"]

PROFILE_DESCRIPTIONS = {
    "ranter":  "paranoid inner monologue, intrusive thoughts, obsessive, self-interrupting, dark mental state",
    "witness": "cold, detached, clinical, observational, descriptive without emotion",
    "ghost":   "fragmented, drifting, melancholic, nostalgic, from another time or place",
    "child":   "simple language, innocent, tender, naive -- context makes it eerie or warm",
}

def get_word_categorizations(word_list, profiles, api_key=None):
    """
    Sends a batch of words to the OpenAI API for semantic categorization.
    """
    descriptions = "\n".join(
        f"- {p}: {PROFILE_DESCRIPTIONS.get(p, p)}" for p in profiles
    )

    prompt = f"""You are building vocabulary databases for AI voice characters in an electronic music project.

Categorize each word into one of these character profiles based on thematic tone and semantic meaning:
{descriptions}

Return ONLY a JSON dictionary where keys are the exact words and values are exact profile names.
If a word fits multiple profiles, pick the strongest match.
If a word is neutral/functional (e.g. "the", "and"), assign the closest thematic fit.

Words: {word_list}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        OPENAI_API_URL,
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    text = response.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print(f"RAW TEXT RESPONSE: {text}")
        raise ValueError("No JSON object in response")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as decode_error:
        print(f"RAW TEXT RESPONSE: {text}")
        raise ValueError(f"JSON decode failed: {decode_error}")


def profile_voices_by_theme(input_dir, output_dir, profiles, batch_size=50, api_key=None):
    """
    Reads audio files, batches their names, gets categorizations from the LLM, and moves the files.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    if not audio_files:
        print(f"No .wav files found in '{input_dir}'.")
        return

    # Extract base word from filename (stop_2.wav -> stop)
    words_to_files = {}
    for file in audio_files:
        word = re.sub(r'_\d+$', '', file.split(".")[0])
        words_to_files.setdefault(word, []).append(file)

    unique_words = list(words_to_files.keys())

    print(f"Found {len(audio_files)} files | {len(unique_words)} unique words")
    print(f"Profiles: {', '.join(profiles)}")
    print("-" * 40)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(unique_words), batch_size):
        batch = unique_words[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(unique_words) + batch_size - 1) // batch_size

        print(f"Batch {batch_num}/{total_batches} ({len(batch)} words)...")

        try:
            categorizations = get_word_categorizations(batch, profiles, api_key)

            for word, profile in categorizations.items():
                # Normalize profile name to lowercase
                profile = profile.lower()
                if profile not in profiles:
                    print(f"  [invalid profile] '{profile}' for '{word}' -> defaulting to '{profiles[-1]}'")
                    profile = profiles[-1]

                profile_dir = os.path.join(output_dir, profile)
                os.makedirs(profile_dir, exist_ok=True)

                for file in words_to_files.get(word, []):
                    src = os.path.join(input_dir, file)
                    dst = os.path.join(profile_dir, file)
                    shutil.copy2(src, dst)

        except Exception as e:
            print(f"  [error] batch {batch_num}: {e}")

    print("-" * 40)
    print(f"Done. Profiles saved to '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice Profiler: Sort voice database into Clankers character profiles using Claude."
    )

    parser.add_argument("--input_dir",  type=str, default="voice_database",          help="Sliced .wav files from voice_slicer.py (default: voice_database)")
    parser.add_argument("--output_dir", type=str, default="voice_database/profiles", help="Output directory for profile folders (default: voice_database/profiles)")
    parser.add_argument("--profiles",   type=str, nargs='+', default=DEFAULT_PROFILES, help="Character profiles (default: ranter witness ghost child)")
    parser.add_argument("--batch_size", type=int, default=50,                         help="Words per API call (default: 50)")
    parser.add_argument("--api_key",    type=str, default=None,                       help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Warning: No OpenAI API key found. Set OPENAI_API_KEY or pass --api_key.")

    profile_voices_by_theme(args.input_dir, args.output_dir, args.profiles, args.batch_size, key)
