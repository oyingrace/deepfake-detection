#!/usr/bin/env bash
# Copy runtime files into deploy/ so this folder is the full Docker build context.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing backend bundle into ${DEPLOY}..."

if [[ ! -f "${ROOT}/outputs/best.pt" ]]; then
  echo "ERROR: outputs/best.pt not found. Train epoch 1 first."
  exit 1
fi

rm -rf "${DEPLOY}/api" "${DEPLOY}/src" "${DEPLOY}/configs" "${DEPLOY}/outputs"
mkdir -p "${DEPLOY}/outputs"

cp -R "${ROOT}/api" "${DEPLOY}/api"
cp -R "${ROOT}/src" "${DEPLOY}/src"
cp -R "${ROOT}/configs" "${DEPLOY}/configs"
cp "${ROOT}/outputs/best.pt" "${DEPLOY}/outputs/best.pt"

# Drop training-only / heavy modules from deploy copy
rm -f "${DEPLOY}/src/data/stream_ff_dataset.py" \
      "${DEPLOY}/src/data/build_metadata.py" \
      "${DEPLOY}/src/data/extract_frames.py" \
      "${DEPLOY}/src/data/extract_eye_sequences.py" \
      "${DEPLOY}/src/train/train.py" \
      "${DEPLOY}/src/train/adversarial.py" \
      "${DEPLOY}/src/eval/"*.py 2>/dev/null || true
rm -rf "${DEPLOY}/src/eval" "${DEPLOY}/src/train" "${DEPLOY}/src/viz" 2>/dev/null || true

echo "Done. Bundle ready:"
du -sh "${DEPLOY}/outputs/best.pt"
echo "files: $(find "${DEPLOY}" -type f | wc -l | tr -d ' ')"
