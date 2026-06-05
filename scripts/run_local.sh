#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv311/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
fi

python -m src.data.stream_ff_dataset \
  --out-root data/processed \
  --metadata-csv data/metadata.csv \
  --num-real "${NUM_REAL:-200}" \
  --num-fake "${NUM_FAKE:-200}"
