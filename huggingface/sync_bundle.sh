#!/usr/bin/env bash
# Copy runtime files into huggingface/ when using Space root directory = huggingface/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing Hugging Face bundle into ${HF}..."

if [[ ! -f "${ROOT}/outputs/best.pt" ]]; then
  echo "ERROR: outputs/best.pt not found. Train epoch 1 first."
  exit 1
fi

rm -rf "${HF}/api" "${HF}/src" "${HF}/configs" "${HF}/outputs"
mkdir -p "${HF}/outputs"

cp -R "${ROOT}/api" "${HF}/api"
cp -R "${ROOT}/src" "${HF}/src"
cp -R "${ROOT}/configs" "${HF}/configs"
cp "${ROOT}/outputs/best.pt" "${HF}/outputs/best.pt"

rm -f "${HF}/src/data/stream_ff_dataset.py" \
      "${HF}/src/data/build_metadata.py" \
      "${HF}/src/data/extract_frames.py" \
      "${HF}/src/data/extract_eye_sequences.py" \
      "${HF}/src/train/train.py" \
      "${HF}/src/train/adversarial.py" 2>/dev/null || true
rm -rf "${HF}/src/eval" "${HF}/src/train" "${HF}/src/viz" 2>/dev/null || true

echo "Done. Commit huggingface/api, src, configs, outputs if using subfolder deploy."
