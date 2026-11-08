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
    for unit in units:
        for piece in split_text(unit):          # the pieces Underdog's engine will synthesize, for the log
            print(f"  piece: {piece}", file=sys.stderr)

    tts = Woof(voice=a.voice, temp=a.temp, normalize=False)    # already normalized by preprocess()
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
