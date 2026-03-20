#!/usr/bin/env python3
# session.py — The Clankers 3
# End-to-end runner. Entry point for the full pipeline.
#
# Usage:
#   python session.py "dark industrial EBM, cold acid bass, mechanical drums"
#   python session.py "paranoid breakdown, lush pads" --arc verse1 bridge outro
#   python session.py "acid rave" --disable voder --out my_output
#
# Pipeline:
#   brief → Chatroom (verse1 negotiation) → Music Sheet JSON
#         → Conductor (evolve per section) → Agent Swarm (parallel)
#         → Mixer (EQ + master compression) → full_track.wav

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conductor.conductor import run_track, run_solo, DEFAULT_ARC


def main():
    parser = argparse.ArgumentParser(
        description="The Clankers 3 — Full Session Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python session.py "dark industrial EBM, cold acid bass"
  python session.py "acid rave, euphoric" --arc verse1 verse2 outro
  python session.py "ambient drone" --disable drums bass303 --out ambient_out
  python session.py "paranoid breakdown" --arc verse1 bridge verse3 --disable voder

Solo companion mode (invoke a single agent in isolation):
  python session.py "dreamy buchla melodies" --solo harmony
  python session.py "heavy kick, sparse snare" --solo drums
  python session.py "cold acid bass" --solo bass_sh101
  python session.py "eerie formant whispers" --solo voder --section bridge
        """,
    )
    parser.add_argument(
        "brief",
        nargs="?",
        default="dark introspective industrial EBM, cold acid bass, mechanical drums",
        help="Client creative brief (free text describing the track)",
    )
    parser.add_argument(
        "--arc",
        nargs="+",
        default=None,
        metavar="SECTION",
        help=f"Section arc. Default: {' '.join(DEFAULT_ARC)}",
    )
    parser.add_argument(
        "--out",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--disable",
        nargs="+",
        default=None,
        metavar="AGENT",
        help="Agent names to disable: sampler bass303 bass_sh101 drums harmony voder",
    )
    parser.add_argument(
        "--solo",
        default=None,
        metavar="AGENT",
        help="Invoke a single companion in isolation: sampler bass_sh101 drums harmony voder",
    )
    parser.add_argument(
        "--section",
        default="verse1",
        help="Section name for --solo mode (default: verse1)",
    )

    args = parser.parse_args()

    try:
        if args.solo:
            result = run_solo(
                brief   = args.brief,
                agent   = args.solo,
                section = args.section,
                out_dir = args.out,
            )
        else:
            result = run_track(
                brief   = args.brief,
                arc     = args.arc,
                out_dir = args.out,
                disable = args.disable,
            )
        if result is None:
            print("[session] No audio produced — check agent logs in the output directory.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[session] Aborted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[session] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
