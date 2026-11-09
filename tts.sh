#!/usr/bin/env bash
# tts.sh — makes Buffy a kocreator TTS plugin (https://github.com/Clay-Ferguson/kocreator).
# (If you don't know what kocreator is, and just want the pure TTS capability you can ignore this file)
#
# kocreator finds this script through the plugin folder set in its kocreator.toml
# (tts_plugin = "../Buffy"). It implements kocreator's batch-only plugin
# contract (see "Writing a TTS Plugin" in kocreator's README.md):
#
#   tts.sh check              exit 0 when Buffy is set up
#   tts.sh settings           print the audio settings as JSON
#   tts.sh speak <jobs.tsv>   speak every job (input<TAB>output<TAB>seed per line)
#                             with one speak.py --batch run, loading the model once
set -euo pipefail

# ── Voice settings ─────────────────────────────────────────────────
# Always passed to speak.py explicitly, rather than left to its defaults,
# because kocreator caches each clip against the `settings` output: change one
# here and every narration is regenerated on the next run.
VOICE="juno"      # juno or mira (mira has audible artifacts; see README.md)
SEED=0            # the take used for jobs with an empty seed field
TEMP="0.5"        # sampling temperature (speak.py's default)

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

usage() {
    echo "Usage: $0 check | settings | speak <jobs.tsv>" >&2
    exit 2
}

cmd_check() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required but was not found in PATH." >&2
        exit 1
    fi
    local missing=0
    for f in speak.py woof_int8.npz underdog_woof/__init__.py; do
        if [ ! -f "$HERE/$f" ]; then
            echo "  missing: $HERE/$f" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "Buffy is not set up. Run it once with:  cd $HERE && ./setup.sh" >&2
        exit 1
    fi
}

cmd_settings() {
    printf '{"engine": "underdog-woof (Buffy)", "voice": "%s", "default_seed": %s, "temp": %s}\n' \
        "$VOICE" "$SEED" "$TEMP"
}

cmd_speak() {
    [ $# -eq 1 ] || usage
    local jobs="$1"
    # Buffy runs in its own .venv; never let a caller's uv environment leak in.
    unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
    # The jobs file holds absolute paths; its own path is made absolute too
    # because --directory runs from this folder.
    exec uv run --locked --directory "$HERE" speak.py --batch "$(realpath -m "$jobs")" \
        --voice "$VOICE" --seed "$SEED" --temp "$TEMP"
}

[ $# -ge 1 ] || usage
command="$1"
shift
case "$command" in
    check)    cmd_check ;;
    settings) cmd_settings ;;
    speak)    cmd_speak "$@" ;;
    *)        usage ;;
esac
