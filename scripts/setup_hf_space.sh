#!/usr/bin/env bash
# One-time push of this repo to your Hugging Face Docker Space.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPACE="${1:-DevQueen/deepfake-server}"

if ! git remote get-url huggingface &>/dev/null; then
  git remote add huggingface "https://huggingface.co/spaces/${SPACE}"
  echo "Added remote: huggingface → https://huggingface.co/spaces/${SPACE}"
else
  echo "Remote 'huggingface' already exists."
fi

echo ""
echo "Next steps:"
echo ""
echo "1. Create a write token: https://huggingface.co/settings/tokens"
echo ""
echo "2. Login (one time):"
echo "   pip install huggingface_hub[cli]"
echo "   huggingface-cli login"
echo ""
echo "3. Push:"
echo "   git push huggingface main"
echo ""
echo "Or use GitHub Actions (recommended): add HF_TOKEN secret on GitHub,"
echo "then every 'git push origin main' auto-deploys the Space."
echo "See huggingface/README.md"
