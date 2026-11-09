# Buffy

Runs the [Underdog Woof](https://huggingface.co/ConwayResearch/Underdog-Woof) text-to-speech model (Juno voice) with
Underdog's own Python code from huggingface, plus our own text preprocessor that prevents repeated phrases (see "Why preprocess" below).
Underdog's rule-based text normalizer reads numbers, dates, times, money, units, codes, URLs and abbreviations aloud the way
people say them, and its lexicon respells about 25,000 hard names and foreign words. See also: [Underdog Website](https://underdog.ai)

### Sample Audio: [Juno Voice](<Sample - Juno Voice - Gettysburg Address.wav>)

## Files

| File | What it is |
|---|---|
| `speak.py` | Command line entry point: our preprocessor, then Underdog's engine, one engine call per line. Speaks one text, or a batch of files with the model loaded once |
| `tts.sh` | Makes Buffy a [kocreator](https://github.com/Clay-Ferguson/kocreator) TTS plugin (see "Use with kocreator" below) |
| `preprocess.py` | Our text preprocessor (see "Why preprocess" below) |
| `setup.sh` | One-step setup: fetches Underdog's package and weights from the audited commit, then `uv sync` |
| `test.sh` | Lists the documents in `input-docs/` by number, speaks the one you pick to `out/<name>.wav`, logs to `out/run_log.txt` |
| `input-docs/` | Test documents (plain text); add as many as you like |
| `pyproject.toml`, `uv.lock` | Python project for [uv](https://docs.astral.sh/uv/); the only dependency is numpy |
| `underdog_woof/` | Underdog's package, with its `LICENSE` and `NOTICE` (fetched by `setup.sh`, not in git) |
| `woof_int8.npz` | Underdog's weights, next to the package where it expects them (fetched by `setup.sh`, not in git) |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and curl. On a clean machine:

```bash
./setup.sh
```

It downloads each Underdog file that isn't already present from commit `be70296` of the Hugging Face repo, the
commit that was audited. A commit ID can't later point to different files, so newer Underdog releases are never picked
up by accident; if that commit ever disappears, the download fails and the script stops. Underdog doesn't publish a
pip-installable package, so `setup.sh` lists the files it needs. It then runs `uv sync --locked`, which creates `.venv`
with the numpy pinned in `uv.lock`. It's safe to run again.

## Underdog updates

Since `setup.sh` always fetches the same commit, Underdog's updates never arrive on their own. To see whether there
are any, open [Underdog Woof's commit history](https://huggingface.co/ConwayResearch/Underdog-Woof/commits/main):
if the top commit's ID starts with `be70296` (the start of `COMMIT` in `setup.sh`), nothing has changed. Anything above
it is new; click a commit to see which files it changed.

Moving to a newer version means auditing it, then setting `COMMIT` in `setup.sh` to the new commit (and adding or
removing entries in `FILES` if Underdog added or removed files) and running `./setup.sh`. `setup.sh` records the commit
it downloaded from in `underdog_woof/.commit`, and downloads every file again when `COMMIT` no longer matches.

## Use

To try a document, run `./test.sh`. It lists the files in `input-docs/` with numbers and asks which one to speak
(or give the number directly: `./test.sh 2`). The audio goes to `out/` under the document's name, so
`input-docs/Gettysburg Address.txt` becomes `out/Gettysburg Address.wav`. The run's output (including every piece
spoken) and system details go to `out/run_log.txt`. It doesn't play anything.

```bash
uv run speak.py "Your order ships on March 3rd. Call 415-555-0199 with questions." -o out/hello.wav
uv run speak.py --file input-docs/story.txt --seed 3        # writes out/story.wav
```

Juno is the default voice (`--voice juno`); `--seed` picks the take (same seed + same text = same audio). The log
lists every piece the engine speaks. Prefer `speak.py` over Underdog's own `python -m underdog_woof`, which skips the
preprocessing below.

### Batches

To speak many files, give `speak.py` a batch file instead of `--file`. It loads the model once and speaks every job in
order. The batch file has one job per line, three tab-separated fields: the input text file, the output WAV, and the
seed (leave it empty to use `--seed`):

```bash
printf '%s\t%s\t\n%s\t%s\t3\n' in/a.txt out/a.wav in/b.txt out/b.wav > jobs.tsv
uv run speak.py --batch jobs.tsv            # a.wav with --seed (0), b.wav with seed 3
```

After writing each WAV it prints `KOCREATOR_DONE <n>` (the 1-based job number) on its own line, which is how kocreator
follows progress. The seed is set at the start of each text, so a text spoken in a batch gives exactly the same audio
as the same text and seed spoken alone. `--voice` and `--temp` apply to the whole batch.

## Use with kocreator

Buffy is also a TTS plugin for [kocreator](https://github.com/Clay-Ferguson/kocreator), which builds narrated demo
videos from screenshots and text. With Buffy checked out next to kocreator and set up, put this in kocreator's
`kocreator.toml`:

```toml
tts_plugin = "../Buffy"
```

kocreator then calls `tts.sh`, which implements kocreator's plugin contract:

- `./tts.sh check` exits 0 when Buffy is set up (Underdog's files are present and uv is installed).
- `./tts.sh settings` prints the voice settings as JSON. kocreator keys its cache on them.
- `./tts.sh speak <jobs.tsv>` runs `speak.py --batch` once for all of kocreator's narrations.

The voice settings kocreator uses (`VOICE`, `SEED`, `TEMP`) are variables at the top of `tts.sh`. kocreator's
per-clip `.seed` files fill the seed field of the batch, so one glitchy clip can be re-rolled without touching the
others. `tts.sh` drops `UV_PROJECT_ENVIRONMENT` and `VIRTUAL_ENV`, so Buffy always runs in its own `.venv`, and Buffy
honours the `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` thread limits kocreator sets. The cache key doesn't include
Buffy's version, so after changing `preprocess.py` or moving to a newer Underdog commit, delete kocreator's
`generated-wav/` folders to regenerate every clip.

## Tip about Voices
This model supports two voices, Mira and Juno. Juno is a great sounding voice, but Mira sounded quite bad with lots of
very audible static and other auditory artifacts, so I highly recommend sticking with Juno.

## Why preprocess

Underdog's engine speaks text in pieces. Its splitter (`underdog_woof/textsplit.py`) cuts every sentence longer than
120 characters, preferring commas, semicolons, colons and dashes. When a stretch between those marks is itself over
120 characters, it falls back to cutting before a word like "who" or "and", which leaves a piece ending mid-phrase
("...a final resting place for those"). The model was trained on speech that ends at punctuation, and on a piece like
that it often doesn't stop cleanly: it repeats its last words. Measured over several random seeds, such pieces run
70–80 ms per character against about 60–70 for normal ones.

`preprocess.py` makes sure that fallback never happens, without changing any Underdog code:

1. **Lines.** Each non-blank line goes to the engine in its own call, so a line break is always a break. A line that
   doesn't end in punctuation gets a period, so titles and headings are spoken as their own sentence.
2. **US states.** State abbreviations right after a place name are written out: "Gettysburg, Pa." becomes
   "Gettysburg, Pennsylvania.", "Boston, MA 02115" becomes "Boston, Massachusetts 02115". Underdog's normalizer
   doesn't do this, and the model can't guess what "Pa." means: in testing it said nothing at all for it in 3 of 4
   takes. Postal codes that are also everyday words (OK, IN, OR, ME, OH, HI, LA...) are only expanded with a ZIP code
   after them or a word like "in" or "from" before the place ("born in Dayton, OH"), so "Fine, OK." is left alone.
   Any other abbreviation the normalizer doesn't know reaches the model as raw letters, so write it out if it matters.
3. **Normalize first.** Underdog's own normalizer runs before measuring, since it makes text longer ("Nov. 19th 1863"
   becomes "November nineteenth eighteen sixty three").
4. **Commas.** Any stretch between punctuation marks longer than 110 characters gets a comma before the joining word
   nearest its middle ("and", "who", "which", "because"...), or before a preposition, or at the middle space as a last
   resort. Commas never go inside a spoken number ("two hundred and thirty four dollars"), never after "a"/"the" or a
   preposition, and "for which" / "to whom" stay together.

So every piece the engine speaks ends on punctuation. On the Gettysburg Address and about 9,600 pieces of real-world
text (the license texts and package READMEs on a Linux machine), no piece ended without punctuation.

What this can't promise: the model is small, and Underdog's own measurement for Juno is 1.53% of words wrong. This
removes the known cause of repeated phrases, but not every possible slip.

## License

Woof's code and weights are under the Underdog Non-Commercial License 1.0 (`underdog_woof/LICENSE`): free for
personal use with credit ("Underdog Woof"); commercial use needs a license from Underdog.
