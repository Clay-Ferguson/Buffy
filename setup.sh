#!/usr/bin/env bash
# One-step setup for Buffy on a clean machine. Safe to run again: it skips whatever is already done.
# To see whether Underdog has published something newer, open https://huggingface.co/ConwayResearch/Underdog-Woof/commits/main
# and compare the top commit's ID with COMMIT below.
#   1. downloads Underdog Woof's package (underdog_woof/), its license files and its weights from one pinned commit of
#      the Hugging Face repo: the commit whose code was security-audited. A commit ID can't point to different files
#      later, so newer Underdog releases are never picked up by accident.
#   2. uv sync: creates .venv with the dependencies pinned in uv.lock (just numpy)
# Needs: uv (https://docs.astral.sh/uv/) and curl.
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# The audited commit. Moving to a newer Underdog release means auditing it first, then changing this line.
COMMIT=be702960539777fc3f698e54335276a8391b00f9
REPO_URL=https://huggingface.co/ConwayResearch/Underdog-Woof/resolve/$COMMIT

# Underdog doesn't publish a pip-installable package, so the files are listed here.
# local path                    path in the repo
FILES=(
    "underdog_woof/__init__.py   underdog_woof/__init__.py"
    "underdog_woof/__main__.py   underdog_woof/__main__.py"
    "underdog_woof/normalize.py  underdog_woof/normalize.py"
    "underdog_woof/respell.py    underdog_woof/respell.py"
    "underdog_woof/respell.tsv   underdog_woof/respell.tsv"
    "underdog_woof/runtime.py    underdog_woof/runtime.py"
    "underdog_woof/textsplit.py  underdog_woof/textsplit.py"
    "underdog_woof/LICENSE       LICENSE"
    "underdog_woof/NOTICE        NOTICE"
    "woof_int8.npz               woof_int8.npz"
)

# ---- Underdog files
# underdog_woof/.commit records which commit the files came from; when COMMIT changes, every file is downloaded again.
STAMP=underdog_woof/.commit
mkdir -p underdog_woof out
fresh=0
if [[ "$(cat "$STAMP" 2>/dev/null)" != "$COMMIT" ]]; then
    fresh=1
    rm -f "$STAMP"
    echo "getting every file from commit ${COMMIT:0:7}"
fi
for entry in "${FILES[@]}"; do
    read -r local remote <<< "$entry"
    if [[ $fresh -eq 0 && -f "$local" ]]; then
        echo "present     $local"
        continue
    fi
    echo "downloading $local"
    curl -fsSL -o "$local.part" "$REPO_URL/$remote"
    mv "$local.part" "$local"
done
echo "$COMMIT" > "$STAMP"

# ---- python environment
uv sync --locked
uv run python -c 'import numpy; print("python: numpy", numpy.__version__, "ready")'

echo
echo "Setup complete. Try:  ./test.sh"
