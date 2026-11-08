#!/usr/bin/env python3
"""Speak English text with Underdog Woof, after our own preprocessing (see preprocess.py).

    uv run speak.py "Hello there. This is Juno." -o out/hello.wav
    uv run speak.py --file input-docs/story.txt          # writes out/story.wav
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from underdog_woof import SR, VOICES, Woof
from underdog_woof.textsplit import split_text

from preprocess import preprocess

HERE = Path(__file__).resolve().parent


def _mmss(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak (or use --file)")
    ap.add_argument("--file", help="read the text from this file")
    ap.add_argument("-o", "--out", help="output WAV (default: out/<input file name>.wav with --file, else out/woof.wav)")
    ap.add_argument("--voice", default="juno", choices=VOICES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.5)
    a = ap.parse_args()

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = a.text
    else:
        ap.error("give the text or --file")

    units = preprocess(text)
    if not units:
        ap.error("no text to speak")
    # the pieces Underdog's engine will synthesize, split the same way Woof.__call__ splits them
    pieces = [p for unit in units for p in (split_text(unit) or [unit])]
    total_chars = sum(len(p) for p in pieces)
    print(f"Speaking {len(pieces)} pieces ({total_chars} characters) with {a.voice}...", file=sys.stderr, flush=True)

    tts = Woof(voice=a.voice, temp=a.temp, normalize=False)    # already normalized by preprocess()

    # Underdog's engine has no progress hook, so we wrap this instance's per-piece method (Underdog's files are
    # untouched) to print one line as each piece finishes. Plain lines, not a redrawn bar, so test.sh's log stays clean.
    engine_piece = tts._piece
    done = {"pieces": 0, "chars": 0}

    def piece_with_progress(piece):
        wav = engine_piece(piece)
        done["pieces"] += 1
        done["chars"] += len(piece)
        spent = time.perf_counter() - t
        left = spent * (total_chars - done["chars"]) / max(1, done["chars"])   # time tracks text length
        pct = 100 * done["chars"] // max(1, total_chars)
        print(f"[{done['pieces']:>{len(str(len(pieces)))}}/{len(pieces)} {pct:3d}%  {_mmss(spent)} elapsed  "
              f"~{_mmss(left)} left] {piece}", file=sys.stderr, flush=True)
        return wav

    tts._piece = piece_with_progress

    t = time.perf_counter()
    # one engine call per line; the seed is set once, so the whole run is reproducible
    audio = np.concatenate([tts(unit, seed=a.seed if k == 0 else None) for k, unit in enumerate(units)])
    elapsed = time.perf_counter() - t

    if a.out:
        out = Path(a.out)
    else:
        out = HERE / "out" / ((Path(a.file).stem if a.file else "woof") + ".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    tts.save(str(out), audio)
    print(f"{out}: {len(audio) / SR:.2f} s of audio in {elapsed:.2f} s ({len(audio) / SR / elapsed:.1f}x real time)")


if __name__ == "__main__":
    main()
