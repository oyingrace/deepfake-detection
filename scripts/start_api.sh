#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv311/bin/activate
pip install -q -r api/requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
