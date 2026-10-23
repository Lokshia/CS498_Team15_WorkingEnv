#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cu130}"
if [[ $# -gt 1 || ( "$backend" != "cu130" && "$backend" != "cpu" ) ]]; then
    printf '%s\n' 'Usage: bash scripts/bootstrap.sh [cu130|cpu]' >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    printf '%s\n' 'This installer targets Linux x86-64. Use bootstrap.ps1 on Windows x86-64.' >&2
    exit 2
fi
command -v uv >/dev/null || {
    printf '%s\n' 'Install uv >= 0.11.21 first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
}
command -v git >/dev/null || {
    printf '%s\n' 'Git must be on PATH to fetch the pinned course and mjlab dependencies.' >&2
    exit 1
}
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
if [[ ! -f uv.lock ]]; then
    printf '%s\n' 'uv.lock is missing. Obtain the complete team release; do not resolve new versions locally.' >&2
    exit 1
fi
# Reuse the exact interpreter or let sync download it; do not separately
# rewrite global Python links. Both overrides are scoped to this command.
UV_PROJECT_ENVIRONMENT="$project_root/.venv" UV_PYTHON_DOWNLOADS=automatic \
    uv sync --locked --no-dev --python 3.13.14 --extra "$backend"
device=cuda
[[ "$backend" != "cpu" ]] || device=cpu
"$project_root/.venv/bin/python" -m lifting_lab doctor --device "$device"
printf 'Environment ready for %s. Interpreter: %s/.venv/bin/python\n' "$device" "$project_root"
printf '%s\n' 'No automatic CPU fallback is performed. Do not update dependencies during a training run.'
