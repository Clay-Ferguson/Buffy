#!/usr/bin/env bash
# Speak one of the documents in input-docs/ with the Juno voice (speak.py: our preprocessor, then Underdog's engine).
# Lists the files with numbers and asks which one to run; or pass the number directly: ./test.sh 2
# The audio goes to out/<file name without extension>.wav, and the run's output, including every piece the engine
# speaks, to out/run_log.txt. Run ./setup.sh first.
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

DOCS=input-docs
LOG=out/run_log.txt

if [[ ! -f woof_int8.npz || ! -f underdog_woof/__main__.py ]]; then
    echo "error: Underdog files missing; run ./setup.sh first" >&2
    exit 1
fi

files=()
for f in "$DOCS"/*; do
    [[ -f "$f" ]] && files+=("$f")
done
if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no files in $DOCS/" >&2
    exit 1
fi

for i in "${!files[@]}"; do
    printf "%3d) %s\n" $((i + 1)) "$(basename "${files[$i]}")"
done

choice="${1:-}"
if [[ -z "$choice" ]]; then
    read -r -p "Number of the file to speak: " choice
fi
if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#files[@]} )); then
    echo "error: '$choice' is not a number from 1 to ${#files[@]}" >&2
    exit 1
fi

TEXT_FILE="${files[$((choice - 1))]}"
name="$(basename "$TEXT_FILE")"
WAV="out/${name%.*}.wav"
if [[ ! -s "$TEXT_FILE" ]]; then
    echo "error: '$TEXT_FILE' is empty" >&2
    exit 1
fi

mkdir -p out
set +e
{
    echo "date: $(date -Is)"
    echo "host: $(uname -srm)"
    echo "python: $(uv run --locked python -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')"
    echo "cpus: $(nproc 2>/dev/null || echo unknown)"
    echo "input: $TEXT_FILE"
    uv run --locked speak.py --file "$TEXT_FILE" -o "$WAV" 2>&1
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
if [[ $status -ne 0 ]]; then
    echo "TTS run FAILED (exit $status); see $LOG" >&2
    exit "$status"
fi

echo "Done: $WAV (log in $LOG)"
