#!/usr/bin/env bash
# One-click launcher for EdgeTTS-Arena (Linux / macOS).
#
# Sources every dedicated worker env script that exists (Qwen3 / CosyVoice /
# MeloTTS), reports which worker interpreters are configured, then starts the
# Arena. Missing env scripts are reported, not fatal.
#
# Usage (from the repo root):
#   ./start.sh                 # serve with the Gradio Arena UI (default)
#   ./start.sh --no-ui         # serve the JSON API only
#   ./start.sh --doctor        # run the deployment doctor and exit
#   ./start.sh --suite piper kokoro   # run the standard suite for given models
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

VENV_PYTHON=".venv/bin/python"
MODE="ui"
SUITE_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-ui) MODE="api"; shift ;;
        --doctor) MODE="doctor"; shift ;;
        --suite) MODE="suite"; shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUITE_MODELS+=("$1"); shift; done ;;
        *) echo "[start] unknown option: $1" >&2; exit 2 ;;
    esac
done

# 1) Source dedicated worker environments if they have been bootstrapped.
for script in \
    "exports/bootstrap/qwen3/env.sh" \
    "exports/bootstrap/cosyvoice/env.sh" \
    "exports/bootstrap/melotts/env.sh"; do
    if [[ -f "$script" ]]; then
        echo "[start] sourcing $script"
        # shellcheck disable=SC1090
        source "$script"
    else
        echo "[start] skip $script (not bootstrapped yet)"
    fi
done

# 2) Report which dedicated worker interpreters are now visible.
report_worker() {
    local var="$1" name="$2"
    if [[ -n "${!var:-}" ]]; then
        echo "[start] $name worker -> ${!var}"
    else
        echo "[start] $name worker NOT configured (that model will error until bootstrapped)"
    fi
}
report_worker EDGETTS_ARENA_QWEN3_PYTHON "Qwen3-TTS 0.6B (PyTorch)"
report_worker EDGETTS_ARENA_COSYVOICE_PYTHON "CosyVoice 300M SFT"
report_worker EDGETTS_ARENA_MELOTTS_PYTHON "MeloTTS Chinese"

# 3) Launch using the main venv interpreter.
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[start] Main venv Python not found: $VENV_PYTHON. Create it first (see README)." >&2
    exit 1
fi

case "$MODE" in
    doctor) "$VENV_PYTHON" -m edgetts_arena doctor --workers --exports-root exports/doctor ;;
    suite) "$VENV_PYTHON" -m edgetts_arena suite --models "${SUITE_MODELS[@]}" ;;
    api) echo "[start] launching API: http://127.0.0.1:8000"; "$VENV_PYTHON" -m edgetts_arena serve ;;
    ui) echo "[start] launching Arena UI: http://127.0.0.1:8000/arena/"; "$VENV_PYTHON" -m edgetts_arena serve --ui ;;
esac
