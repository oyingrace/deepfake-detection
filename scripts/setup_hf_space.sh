#!/usr/bin/env bash
# Link this repo to a Hugging Face Docker Space (run once after creating the Space on huggingface.co).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-username>/deepfake-detection"
  echo "Example: $0 oyingrace/deepfake-detection"
  exit 1
fi

SPACE_ID="$1"

if ! command -v git >/dev/null; then
  echo "ERROR: git is required."
  exit 1
fi

if ! command -v huggingface-cli >/dev/null; then
  echo "Install the Hugging Face CLI first:"
  echo "  pip install huggingface_hub[cli]"
  echo "  huggingface-cli login"
  exit 1
fi

echo "Adding Hugging Face Space remote: ${SPACE_ID}"
git remote remove huggingface 2>/dev/null || true
git remote add huggingface "https://huggingface.co/spaces/${SPACE_ID}"

echo ""
echo "Remote added. Push with:"
echo "  git push huggingface main"
echo ""
echo "Or connect GitHub in Space Settings → Repository (recommended — auto-deploy on push)."
