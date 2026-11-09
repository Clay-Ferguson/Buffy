#!/usr/bin/env python3
"""Speak English text with Underdog Woof, after our own preprocessing (see preprocess.py).

    uv run speak.py "Hello there. This is Juno." -o out/hello.wav
    uv run speak.py --file input-docs/story.txt          # writes out/story.wav
    uv run speak.py --batch jobs.tsv                     # many files, loading the model once

A batch file has one job per line, three tab-separated fields:

    <input.txt> TAB <output.wav> TAB <seed, or empty for --seed>

The model is loaded once for the whole batch, and after writing each WAV the line "KOCREATOR_DONE <n>" (n = the
1-based job number) is printed to stdout. That is the batch contract of kocreator's TTS plugins (tts.sh in this
folder), but nothing here depends on kocreator.
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
DONE_MARKER = "KOCREATOR_DONE"


def _mmss(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


class Speaker:
    """Underdog's engine, loaded once, plus our preprocessing and per-piece progress lines."""

    def __init__(self, voice, temp):
        self.voice = voice
        self.tts = Woof(voice=voice, temp=temp, normalize=False)    # already normalized by preprocess()
        # Underdog's engine has no progress hook, so we wrap this instance's per-piece method (Underdog's files are
        # untouched) to print one line as each piece finishes. Plain lines, not a redrawn bar, so logs stay clean.
        self._engine_piece = self.tts._piece
        self.tts._piece = self._piece_with_progress
        self._progress = None

    def _piece_with_progress(self, piece):
        wav = self._engine_piece(piece)
        p = self._progress
        p["pieces"] += 1
        p["chars"] += len(piece)
        spent = time.perf_counter() - p["start"]
        left = spent * (p["total_chars"] - p["chars"]) / max(1, p["chars"])   # time tracks text length
        pct = 100 * p["chars"] // max(1, p["total_chars"])
        n = p["total_pieces"]
        print(f"[{p['pieces']:>{len(str(n))}}/{n} {pct:3d}%  {_mmss(spent)} elapsed  "
              f"~{_mmss(left)} left] {piece}", file=sys.stderr, flush=True)
        return wav

    def speak(self, text, seed):
        """Return (audio, seconds spent) for *text*, or raise ValueError if there is nothing to speak."""
        units = preprocess(text)
        if not units:
            raise ValueError("no text to speak")
        # the pieces Underdog's engine will synthesize, split the same way Woof.__call__ splits them
        pieces = [p for unit in units for p in (split_text(unit) or [unit])]
        total_chars = sum(len(p) for p in pieces)
        print(f"Speaking {len(pieces)} pieces ({total_chars} characters) with {self.voice}...",
              file=sys.stderr, flush=True)

        start = time.perf_counter()
        self._progress = {"pieces": 0, "chars": 0, "start": start,
                          "total_pieces": len(pieces), "total_chars": total_chars}
        # one engine call per line; the seed is set once, at the start of each text, so every text is reproducible
        # on its own: the same text and seed give the same audio whether spoken alone or anywhere in a batch
        audio = np.concatenate([self.tts(unit, seed=seed if k == 0 else None) for k, unit in enumerate(units)])
        return audio, time.perf_counter() - start

    def save(self, out, audio, elapsed):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.tts.save(str(out), audio)
        print(f"{out}: {len(audio) / SR:.2f} s of audio in {elapsed:.2f} s "
              f"({len(audio) / SR / elapsed:.1f}x real time)", flush=True)


def read_batch(path):
    """Parse a batch file into (input, output, seed or None) tuples; exit with a message if it's malformed."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        sys.exit(f"error: batch file not found: {path}")
    jobs = []
    for number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0] or not fields[1]:
            sys.exit(f"error: {path} line {number}: expected input<TAB>output<TAB>seed, got {line!r}")
        seed = fields[2].strip()
        if seed and not seed.lstrip("-").isdigit():
            sys.exit(f"error: {path} line {number}: seed must be a whole number, got {seed!r}")
        jobs.append((fields[0], fields[1], int(seed) if seed else None))
    if not jobs:
        sys.exit(f"error: batch file is empty: {path}")
    return jobs


def run_batch(speaker, jobs, default_seed):
    for number, (input_path, output_path, seed) in enumerate(jobs, start=1):
        print(f"[job {number}/{len(jobs)}] {input_path}", flush=True)
        try:
            text = Path(input_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            sys.exit(f"error: input file not found: {input_path}")
        try:
            audio, elapsed = speaker.speak(text, default_seed if seed is None else seed)
        except ValueError as e:
            sys.exit(f"error: {input_path}: {e}")
        speaker.save(output_path, audio, elapsed)
        print(f"{DONE_MARKER} {number}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak (or use --file or --batch)")
    ap.add_argument("--file", help="read the text from this file")
    ap.add_argument("--batch", help="speak every job in this batch file (see above), loading the model once")
    ap.add_argument("-o", "--out", help="output WAV (default: out/<input file name>.wav with --file, else out/woof.wav)")
    ap.add_argument("--voice", default="juno", choices=VOICES)
    ap.add_argument("--seed", type=int, default=0, help="the take; in a batch, for jobs whose seed field is empty")
    ap.add_argument("--temp", type=float, default=0.5)
    a = ap.parse_args()

    if a.batch:
        if a.text or a.file or a.out:
            ap.error("--batch can't be combined with text, --file or --out")
        jobs = read_batch(a.batch)
        started = time.perf_counter()
        speaker = Speaker(a.voice, a.temp)
        print(f"Loaded Underdog Woof ({a.voice}) in {time.perf_counter() - started:.2f} s; "
              f"speaking {len(jobs)} job(s)", flush=True)
        run_batch(speaker, jobs, a.seed)
        return

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = a.text
    else:
        ap.error("give the text, --file or --batch")

    if not preprocess(text):
        ap.error("no text to speak")
    speaker = Speaker(a.voice, a.temp)
    audio, elapsed = speaker.speak(text, a.seed)
    if a.out:
        out = Path(a.out)
    else:
        out = HERE / "out" / ((Path(a.file).stem if a.file else "woof") + ".wav")
    speaker.save(out, audio, elapsed)


if __name__ == "__main__":
    main()
