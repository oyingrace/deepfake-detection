# DeepFake Eye-Blink Detection — Cursor AI Build Plan

## Project Overview

An Enhanced Eye-Blinking LRCN (Long-term Recurrent ConvNet) for DeepFake detection using Attentive Adversarial Training (AAT) with a Vision Transformer (ViT) backbone. The research is by **Alina Chikwado Godsaves** under supervision of **Mr. Akanji**.

The goal is to detect deepfake videos by analyzing unnatural eye-blinking patterns and fine-grained ocular artifacts (eyelid dynamics, pupil reflections) using a hybrid CNN/LSTM + ViT model hardened with adversarial training.

**Python version: 3.11** (PyTorch does not install on 3.13)

---

## What Is Already Scaffolded (Do NOT Recreate)

All files below exist but need to be **verified, completed, and wired together**:

| Area | Files | Status |
|------|-------|--------|
| Config system | `configs/base.yaml`, `configs/model/lrcn_vit.yaml`, `configs/train/aat_pgd.yaml` | ✅ Exists |
| Data pipeline | `src/data/build_metadata.py`, `src/data/extract_frames.py`, `src/data/extract_eye_sequences.py`, `src/data/dataset.py` | ✅ Exists, verify |
| Model | `src/models/backbones.py`, `src/models/lrcn_vit.py` | ✅ Exists, verify |
| Training | `src/train/train.py`, `src/train/adversarial.py` | ✅ Exists, verify |
| Evaluation | `src/eval/evaluate.py`, `src/eval/ablation.py`, `src/eval/plots.py` | ✅ Exists, verify |
| Explainability | `src/viz/attention_maps.py` | ✅ Exists |
| Scripts | `scripts/run_local.sh`, `scripts/run_cloud.sh` | ✅ Exists |
| Docs | `docs/reproducibility_checklist.md`, `docs/results_template.md` | ✅ Exists |

---

## Phase 0 — Environment & Dependency Fix (FIRST PRIORITY)

**Goal:** Get a working Python 3.11 venv with all ML/CV deps installed.

### Tasks

- [ ] **0.1** Confirm `python3.11` is available, or install via `pyenv` / system package manager
- [ ] **0.2** Create venv: `python3.11 -m venv .venv311 && source .venv311/bin/activate`
- [ ] **0.3** Pin exact working versions in `requirements.txt`:
  ```
  torch==2.2.2
  torchvision==0.17.2
  timm==0.9.16
  opencv-python-headless==4.9.0.80
  mediapipe==0.10.11
  pandas==2.2.2
  numpy==1.26.4
  scikit-learn==1.4.2
  matplotlib==3.8.4
  seaborn==0.13.2
  tqdm==4.66.4
  pytorch-grad-cam==1.5.0
  Pillow==10.3.0
  pyyaml==6.0.1
  albumentations==1.4.3
  einops==0.7.0
  wandb==0.17.0
  datasets==2.19.0
  huggingface_hub==0.23.0
  av==12.0.0
  ```
- [ ] **0.4** Run `pip install -r requirements.txt` inside venv and confirm zero errors
- [ ] **0.5** Smoke-test: `python -c "import torch; import timm; import mediapipe; import datasets; print('OK')"`
- [ ] **0.6** Update `scripts/run_local.sh` to activate `.venv311` before any python calls
- [ ] **0.7** One-time HuggingFace login (only needed once per machine):
  ```bash
  huggingface-cli login
  # Paste your token from https://huggingface.co/settings/tokens
  # Token needs Read access only
  ```

---

## Phase 1 — Dataset via HuggingFace Streaming (NO DOWNLOAD NEEDED)

**Goal:** Stream FaceForensics++ c23 videos directly from HuggingFace one at a time, extract eye sequences into tiny `.npz` files, and discard each video. No raw videos are ever stored on disk.

### How Streaming Works

```
HuggingFace server
    → sends video #1 to RAM (temp, ~5MB)
    → MediaPipe extracts eye crops + EAR signal
    → saves tiny .npz file (~50KB) to data/processed/
    → video is gone from memory
    → repeat for video #2, #3 ... #200
```

At the end: ~200 `.npz` files totalling ~100–300MB. Zero raw videos on disk.

### Dataset

**Source:** `bitmind/FaceForensicsC23` on HuggingFace  
**URL:** https://huggingface.co/datasets/bitmind/FaceForensicsC23  
**Contents:** 7,000 MP4 videos — 1,000 real + 6,000 deepfakes (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, DeepFakeDetection), c23 compression  
**We use:** 200 real (`/Real/`) + 200 fake (`/Deepfakes/`) = 400 videos total

### Tasks

- [ ] **1.1** Create `src/data/stream_ff_dataset.py` — a NEW script that replaces the old download-based `build_metadata.py` + `extract_frames.py` flow:

```python
"""
Stream FaceForensics++ c23 from HuggingFace.
Downloads one video at a time into RAM, extracts eye sequences,
saves .npz files, discards the video. No raw videos stored on disk.

Usage:
    python -m src.data.stream_ff_dataset \
        --out-root data/processed \
        --num-real 200 \
        --num-fake 200
"""
import io, tempfile, os, csv
import numpy as np
import cv2
import mediapipe as mp
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

HF_DATASET = "bitmind/FaceForensicsC23"
REAL_PATH_MARKER = "/Real/"
FAKE_PATH_MARKER = "/Deepfakes/"   # use only Deepfakes subfolder, not all 6

def compute_ear(landmarks, eye_indices):
    """Compute Eye Aspect Ratio from MediaPipe landmarks."""
    # eye_indices: [p1, p2, p3, p4, p5, p6]
    p = [landmarks[i] for i in eye_indices]
    A = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    B = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    C = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    return (A + B) / (2.0 * C + 1e-6)

def extract_sequences_from_video_bytes(video_bytes, label, video_id, seq_len=16):
    """
    Given raw video bytes, extract overlapping eye-region sequences.
    Returns list of dicts: {'frames': (T,H,W,3), 'ear': (T,), 'label': int, 'video_id': str}
    """
    # Write to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    sequences = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    # MediaPipe eye landmark indices (left eye outer→inner, right eye similar)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / 10))  # sample at ~10fps

    all_frames, all_ears = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                # Compute EAR (average of both eyes)
                left_ear = compute_ear(lm, LEFT_EYE)
                right_ear = compute_ear(lm, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # Crop eye region: bounding box around both eyes
                eye_pts = [lm[i] for i in LEFT_EYE + RIGHT_EYE]
                xs = [int(p.x * w) for p in eye_pts]
                ys = [int(p.y * h) for p in eye_pts]
                x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
                y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)
                crop = rgb[y1:y2, x1:x2]

                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224))  # ViT input size
                    all_frames.append(crop)
                    all_ears.append(ear)

        frame_idx += 1

    cap.release()
    face_mesh.close()
    os.unlink(tmp_path)  # delete temp file immediately

    # Slice into non-overlapping sequences of length seq_len
    for i in range(0, len(all_frames) - seq_len + 1, seq_len):
        frames = np.stack(all_frames[i:i+seq_len]).astype(np.uint8)
        ears = np.array(all_ears[i:i+seq_len], dtype=np.float32)
        sequences.append({
            'frames': frames,
            'ear': ears,
            'label': label,
            'video_id': f"{video_id}_seq{i}"
        })

    return sequences


def stream_and_extract(out_root, num_real=200, num_fake=200, seq_len=16):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Stream dataset — never downloads the full zip
    ds = load_dataset(HF_DATASET, streaming=True, split="train")

    real_count, fake_count = 0, 0
    metadata_rows = []

    pbar = tqdm(total=num_real + num_fake, desc="Streaming videos")

    for item in ds:
        video_path_str = str(item.get('video', ''))

        is_real = REAL_PATH_MARKER in video_path_str and real_count < num_real
        is_fake = FAKE_PATH_MARKER in video_path_str and fake_count < num_fake

        if not is_real and not is_fake:
            continue

        label = 0 if is_real else 1
        video_id = Path(video_path_str).stem

        # item['video'] is a dict with 'bytes' key when streaming
        video_bytes = item['video']['bytes'] if isinstance(item['video'], dict) else None
        if video_bytes is None:
            continue

        sequences = extract_sequences_from_video_bytes(
            video_bytes, label, video_id, seq_len
        )

        for seq in sequences:
            npz_name = f"{seq['video_id']}.npz"
            npz_path = out_root / npz_name
            np.savez_compressed(
                npz_path,
                frames=seq['frames'],
                ear=seq['ear'],
                label=np.array(seq['label']),
                video_id=np.array(seq['video_id'])
            )
            metadata_rows.append({
                'npz_path': str(npz_path),
                'label': label,
                'video_id': video_id,
                'split': 'train'  # will be reassigned below
            })

        if is_real:
            real_count += 1
        else:
            fake_count += 1

        pbar.update(1)

        if real_count >= num_real and fake_count >= num_fake:
            break

    pbar.close()

    # Assign splits: 70% train, 15% val, 15% test (by video_id, not sequence)
    unique_ids = list({r['video_id'] for r in metadata_rows})
    np.random.shuffle(unique_ids)
    n = len(unique_ids)
    train_ids = set(unique_ids[:int(0.7 * n)])
    val_ids = set(unique_ids[int(0.7 * n):int(0.85 * n)])

    for row in metadata_rows:
        if row['video_id'] in train_ids:
            row['split'] = 'train'
        elif row['video_id'] in val_ids:
            row['split'] = 'val'
        else:
            row['split'] = 'test'

    # Write metadata CSV
    csv_path = Path('data/metadata.csv')
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['npz_path', 'label', 'video_id', 'split'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDone! {real_count} real + {fake_count} fake videos processed.")
    print(f"Total sequences: {len(metadata_rows)}")
    print(f"Metadata written to: {csv_path}")
    print(f"Sequences saved to: {out_root}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default='data/processed')
    parser.add_argument('--num-real', type=int, default=200)
    parser.add_argument('--num-fake', type=int, default=200)
    parser.add_argument('--seq-len', type=int, default=16)
    args = parser.parse_args()
    stream_and_extract(args.out_root, args.num_real, args.num_fake, args.seq_len)
```

- [ ] **1.2** Run the streaming script:
  ```bash
  source .venv311/bin/activate
  python -m src.data.stream_ff_dataset \
    --out-root data/processed \
    --num-real 200 \
    --num-fake 200
  ```
  This will run for ~20–60 minutes depending on internet speed. It streams each video, processes it, saves a tiny `.npz`, and moves on. Your terminal will show a progress bar.

- [ ] **1.3** When done, verify output:
  ```bash
  ls data/processed/ | wc -l      # should be several hundred .npz files
  du -sh data/processed/           # should be ~100-300MB total
  python -c "
  import numpy as np
  d = np.load('data/processed/' + __import__('os').listdir('data/processed')[0], allow_pickle=True)
  print('frames:', d['frames'].shape)   # expect (16, 224, 224, 3)
  print('ear:', d['ear'].shape)         # expect (16,)
  print('label:', d['label'])           # expect 0 or 1
  "
  ```

- [ ] **1.4** Verify `data/metadata.csv` has rows with `npz_path`, `label`, `video_id`, `split` columns and a healthy mix of train/val/test rows

- [ ] **1.5** Update `src/data/dataset.py` to read from `data/metadata.csv` (pointing to `.npz` files) instead of from raw video paths. The `__getitem__` contract remains unchanged:
  ```python
  {'frames': Tensor[T,3,224,224], 'ear': Tensor[T], 'label': int}
  ```

- [ ] **1.6** Update `configs/base.yaml`:
  ```yaml
  data:
    metadata_csv: data/metadata.csv
    processed_root: data/processed
    seq_len: 16
    img_size: 224
  ```

- [ ] **1.7** Add to `.gitignore`:
  ```
  data/processed/
  data/raw/
  data/metadata.csv
  outputs/
  *.npz
  *.pt
  .venv311/
  ```

---

## Phase 2 — Dataset Loader Verification

**Goal:** Confirm `src/data/dataset.py` correctly reads the `.npz` files produced by streaming.

### Tasks

- [ ] **2.1** Open `src/data/dataset.py` — update it to read from `metadata.csv` instead of raw video paths. Each row's `npz_path` points directly to a processed sequence file.
- [ ] **2.2** Add `albumentations` augmentations for training split only:
  ```python
  import albumentations as A
  from albumentations.pytorch import ToTensorV2

  train_transform = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
      A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
      A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2(),
  ])
  ```
- [ ] **2.3** Smoke-test the DataLoader:
  ```python
  from src.data.dataset import EyeBlinkDataset
  ds = EyeBlinkDataset('data/metadata.csv', split='train')
  sample = ds[0]
  assert sample['frames'].shape == (16, 3, 224, 224)
  assert sample['ear'].shape == (16,)
  assert sample['label'] in [0, 1]
  print("DataLoader OK")
  ```

---

## Phase 3 — Model Architecture Verification & Fix

**Goal:** Ensure the LRCN + ViT hybrid model is correctly implemented and matches the research proposal.

### Architecture Spec (from proposal)

```
Input: eye-region sequence (T=16 frames, each 224×224 RGB) + EAR signal (T floats)
  ↓
ViT Backbone (timm: vit_small_patch16_224, pretrained=True)
  → Per-frame [CLS] token → shape (T, 384)
  ↓
LSTM Temporal Encoder
  → Hidden size: 256, Num layers: 2, Dropout: 0.3
  ↓
Blink Dynamics Head
  → Concatenate LSTM output + EAR
  → FC(257, 128) → ReLU
  → Blink timing constraint (0.1–0.4s window)
  ↓
Classifier Head
  → FC(256, 128) → ReLU → Dropout(0.5) → FC(128, 2)
  → Output: [real_logit, fake_logit]
```

### Tasks

- [ ] **3.1** Open `src/models/backbones.py` — verify `build_backbone(config)` returns a timm ViT. For `vit_small_patch16_224` embed dim = 384.
- [ ] **3.2** Open `src/models/lrcn_vit.py` — verify forward pass. Frames arrive as `(B, T, 3, 224, 224)`. Reshape to `(B*T, 3, 224, 224)` before ViT, then reshape back to `(B, T, embed_dim)` before LSTM.
- [ ] **3.3** Add **attention consistency loss**: KL-divergence between adjacent frame ViT attention maps, weighted by `lambda_attn`.
- [ ] **3.4** Add **blink timing regularizer**: penalize uncertain predictions when EAR < 0.2 but blink duration is outside 0.1–0.4s. Weight: `lambda_blink`.
- [ ] **3.5** Add unit test in `tests/test_model.py`:
  ```python
  model = LRCNViT(config)
  dummy = {'frames': torch.randn(2, 16, 3, 224, 224), 'ear': torch.randn(2, 16)}
  out = model(dummy)
  assert out['logits'].shape == (2, 2)
  ```

---

## Phase 4 — Training Loop Fix & Wire-Up

**Goal:** Get the full training loop running end-to-end with adversarial training and all loss components.

### Tasks

- [ ] **4.1** Open `src/train/train.py` — verify it loads config, DataLoader, model, AdamW, LR scheduler, and saves `outputs/best.pt` on val AUC improvement.
- [ ] **4.2** **Wire in `wandb`**: if `config.wandb.enabled: true`, call `wandb.init()` and log metrics each epoch.
- [ ] **4.3** Total loss formula:
  ```
  L_total = L_ce(clean)
          + alpha     * L_ce(adversarial)
          + lambda_attn  * L_attn_consistency
          + lambda_blink * L_blink_regularizer
  ```
- [ ] **4.4** Open `src/train/adversarial.py` — verify PGD: `eps=8/255`, `steps=10`, applied only to eye-region frames.
- [ ] **4.5** Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- [ ] **4.6** Update `configs/train/aat_pgd.yaml`:
  ```yaml
  epochs: 30
  batch_size: 16
  lr: 3e-4
  weight_decay: 1e-4
  alpha: 0.5
  lambda_attn: 0.1
  lambda_blink: 0.05
  pgd_eps: 0.031
  pgd_steps: 10
  wandb:
    enabled: false
    project: "deepfake-eye-blink"
  ```
- [ ] **4.7** Smoke-train: 2 epochs on 50 samples — confirm zero errors.
- [ ] **4.8** Full training: `python -m src.train.train --config configs/train/aat_pgd.yaml`

---

## Phase 5 — Evaluation & Ablation

**Goal:** Produce evaluation numbers and ablation table for the thesis.

### Tasks

- [ ] **5.1** Open `src/eval/evaluate.py` — verify it outputs Accuracy, Precision, Recall, F1, AUC.
- [ ] **5.2** Run: `python -m src.eval.evaluate --checkpoint outputs/best.pt --config configs/train/aat_pgd.yaml`
- [ ] **5.3** Open `src/eval/ablation.py` — confirm 4 configs: Full / No AAT / No ViT / No blink regularizer.
- [ ] **5.4** Run ablation: `python -m src.eval.ablation --config configs/train/aat_pgd.yaml`
- [ ] **5.5** Open `src/eval/plots.py` — confirm it generates `confusion_matrix.png` and `roc_curve.png`.
- [ ] **5.6** Fill in `docs/results_template.md` with actual numbers.

---

## Phase 6 — Inference API

**Goal:** FastAPI server that accepts an uploaded video and returns a prediction.

### New files
```
api/
  main.py
  inference.py    # reuses the same eye extraction logic from stream_ff_dataset.py
  schemas.py
  requirements.txt
```

### Tasks

- [ ] **6.1** `api/inference.py` — reuse `extract_sequences_from_video_bytes()` from `stream_ff_dataset.py`. Load model once, run forward pass on all sequences, average predictions across sequences.
- [ ] **6.2** `api/main.py` — `/predict` endpoint (POST, multipart file upload) + `/health` endpoint.
- [ ] **6.3** Load model at startup via FastAPI `lifespan`, not per-request.
- [ ] **6.4** Add CORS for `http://localhost:5173`.
- [ ] **6.5** `api/requirements.txt`: `fastapi>=0.111.0`, `uvicorn[standard]`, `python-multipart>=0.0.9`
- [ ] **6.6** Test: `curl -X POST http://localhost:8000/predict -F "file=@test_video.mp4"`
- [ ] **6.7** `scripts/start_api.sh`:
  ```bash
  source .venv311/bin/activate
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```

---

## Phase 7 — Demo Frontend

**Goal:** React web UI for the defence demonstration.

### Stack: React + Vite + Tailwind + Recharts

```
frontend/
  src/
    App.jsx
    components/
      VideoUploader.jsx
      ResultCard.jsx
      FrameChart.jsx
      AttentionViewer.jsx
  index.html
  package.json
  vite.config.js
```

### Tasks

- [ ] **7.1** `cd frontend && npm create vite@latest . -- --template react && npm install`
- [ ] **7.2** `npm install tailwindcss recharts axios`
- [ ] **7.3** `VideoUploader.jsx`: drag-and-drop or file picker for `.mp4/.avi/.mov`, video preview, "Analyse Video" button, loading spinner.
- [ ] **7.4** `ResultCard.jsx`: REAL (green) / FAKE (red) verdict badge, confidence %, blink rate stat.
- [ ] **7.5** `FrameChart.jsx`: Recharts line chart of per-frame fake probability, frames above 0.5 highlighted red.
- [ ] **7.6** `AttentionViewer.jsx`: Grad-CAM attention overlay image from API response.
- [ ] **7.7** Proxy in `vite.config.js`: `/predict` → `http://localhost:8000/predict`
- [ ] **7.8** `frontend/.env`: `VITE_API_URL=http://localhost:8000`
- [ ] **7.9** `scripts/start_frontend.sh`:
  ```bash
  cd frontend && npm run dev
  ```

---

## Phase 8 — Integration & Final QA

- [ ] **8.1** Run API + frontend together. Upload one of the `.npz` source videos as a test.
- [ ] **8.2** Test with a real webcam recording — should return REAL.
- [ ] **8.3** Fix any CORS issues.
- [ ] **8.4** Create `docs/README_DEMO.md`:
  ```
  1. source .venv311/bin/activate
  2. ./scripts/start_api.sh         (Terminal 1)
  3. ./scripts/start_frontend.sh    (Terminal 2)
  4. Open http://localhost:5173
  ```
- [ ] **8.5** Document exact setup commands for a fresh machine.

---

## Project Directory Structure (Final)

```
deepfake-detector/
├── configs/
│   ├── base.yaml
│   ├── model/lrcn_vit.yaml
│   └── train/aat_pgd.yaml
├── src/
│   ├── data/
│   │   ├── stream_ff_dataset.py   ← NEW (replaces download-based flow)
│   │   ├── extract_eye_sequences.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── backbones.py
│   │   └── lrcn_vit.py
│   ├── train/
│   │   ├── train.py
│   │   └── adversarial.py
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── ablation.py
│   │   └── plots.py
│   ├── viz/
│   │   └── attention_maps.py
│   └── utils.py
├── api/
│   ├── main.py
│   ├── inference.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── processed/     ← .npz files only (~200MB), gitignored
│   └── metadata.csv   ← generated, gitignored
├── outputs/
│   ├── best.pt
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── scripts/
│   ├── run_local.sh
│   ├── run_cloud.sh
│   ├── start_api.sh
│   └── start_frontend.sh
├── tests/
│   └── test_model.py
├── docs/
│   ├── reproducibility_checklist.md
│   ├── results_template.md
│   └── README_DEMO.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Suggestions & Overrides

### ⚠️ Old files to DEPRECATE (keep but do not use)
`src/data/build_metadata.py` and `src/data/extract_frames.py` were written for a local download workflow. They are superseded by `stream_ff_dataset.py`. Keep them in the repo for reference but do not run them.

### ⚠️ ViT Input Resolution
Frames are extracted at 224×224 directly in the streaming script. No resizing needed elsewhere.

### ⚠️ Internet Required for Phase 1
The streaming script needs internet during the ~20–60 min preprocessing run. After that, everything runs offline from the `.npz` files.

### ⚠️ Pre-trained Checkpoint Option
Use `timm`'s pretrained ViT weights (ImageNet). Fine-tuning for 5–10 epochs on 400 videos is sufficient for a compelling defence demo.

### ✅ Frontend: Keep it Simple
Single-page upload → result. No auth, no database needed.