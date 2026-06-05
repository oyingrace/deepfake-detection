# Enhanced Deepfake Detector

Research-oriented deepfake detection pipeline based on an LRCN + ViT hybrid with blink-aware temporal modeling and attentive adversarial training.

## Features
- Eye-region sequence extraction from videos (MediaPipe + OpenCV)
- LRCN-style temporal model with ViT/CNN backbone
- Adversarial training hooks (FGSM/PGD)
- Evaluation suite (Accuracy, Precision, Recall, F1, AUC)
- Ablation runner and explainability utilities

## Project Structure
- `configs/` experiment and model configs
- `src/data/` dataset preparation and preprocessing
- `src/models/` backbones and model architecture
- `src/train/` training and adversarial routines
- `src/eval/` evaluation and ablation utilities
- `src/viz/` attention map visualization helpers
- `scripts/` local/cloud convenience runners

## Quick Start

### 1. Environment (Python 3.11 only)

```bash
PYENV_VERSION=3.11.11 python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-phase1.txt
pip install mediapipe==0.10.21 --no-deps
pip install absl-py attrs flatbuffers "protobuf>=4.25,<5" jax jaxlib matplotlib sounddevice
```

### 2. HuggingFace login (one-time)

```bash
hf auth login
```

### 3. Phase 1 — Stream dataset and build `.npz` files

Uses HuggingFace **streaming** (`load_dataset(..., streaming=True)` + `decode(False)`): one video at a time in RAM, no 17GB zip copied into `data/processed/`. Only `.npz` files are saved locally.

```bash
source .venv311/bin/activate
python -m src.data.stream_ff_dataset \
  --out-root data/processed \
  --metadata-csv data/metadata.csv \
  --num-real 200 \
  --num-fake 200
```

Smoke test (2+2 videos):

```bash
NUM_REAL=2 NUM_FAKE=2 bash scripts/run_local.sh
```

### 4. Phase 2 — Train (after Phase 1 completes)

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn
python -m src.train.train --config configs/train/aat_pgd.yaml
python -m src.eval.evaluate --checkpoint outputs/best.pt --config configs/train/aat_pgd.yaml
```

## Datasets
- FaceForensics++
- Celeb-DF

Use identity-disjoint splits to avoid leakage.

## Notes
- Local machine: preprocessing/debug
- Cloud GPU: full training and ablation sweeps
