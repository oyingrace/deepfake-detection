# Cursor AI Skill: DeepFake Eye-Blink Detector

## What This Project Is

An Enhanced Eye-Blinking LRCN (Long-term Recurrent ConvNet) for DeepFake video detection using Attentive Adversarial Training (AAT). The model detects deepfakes by analysing unnatural blink patterns and fine-grained ocular artifacts using a hybrid ViT + LSTM architecture hardened with PGD adversarial training.

**Research by:** Alina Chikwado Godsaves (BSc Computer Science)
**Supervisor:** Mr. Akanji

---

## Critical Constraints — Read Before Writing Any Code

| Constraint | Rule |
|-----------|------|
| **Python version** | Always use Python 3.11. Do NOT use 3.12 or 3.13. |
| **Virtual environment** | `.venv311/` at project root. All scripts activate it first. |
| **ViT input size** | Frames are already 224×224 from the streaming script. Never resize again inside the model. |
| **Face detection** | Use **MediaPipe** only. Not Dlib. |
| **No raw videos on disk** | The streaming script processes videos in RAM and saves only `.npz` files. Never write `.mp4` files permanently. |
| **Dataset source** | `bitmind/FaceForensicsC23` on HuggingFace, streamed — NOT downloaded. |
| **No localhost hardcoding** | API base URL comes from `VITE_API_URL` env var in the frontend. |
| **No `<form>` tags in React** | Use `onClick` + `FormData` + `axios.post()` instead. |

---

## Dataset: How Streaming Works

The project does NOT download the 17.9GB dataset zip. Instead:

```
HuggingFace → one video at a time (in RAM, ~5MB)
    → MediaPipe extracts eye crops + EAR signal
    → saves .npz file (~50KB) to data/processed/
    → video discarded from memory
    → repeat
```

**Key file:** `src/data/stream_ff_dataset.py`
- Run once to produce all `.npz` files
- Takes ~20–60 min depending on internet
- Requires `hf auth login` (one-time setup) - Done 
- After it finishes, everything runs fully offline

**HuggingFace login (one-time):**
```bash
pip install huggingface_hub
hf auth login
# Paste token from https://huggingface.co/settings/tokens (Read access)
```

---

## Architecture Reference

### Model Pipeline

```
Input: eye sequence (T=16 frames, 224×224 RGB) + EAR signal (T floats)
  ↓
Reshape: (B, T, 3, 224, 224) → (B*T, 3, 224, 224)
  ↓
ViT Backbone (timm: vit_small_patch16_224, pretrained=True)
  → Extract [CLS] token per frame → shape: (B*T, 384)
  → Reshape back: (B, T, 384)
  ↓
LSTM Temporal Encoder
  → input: (B, T, 384), hidden_size=256, num_layers=2, dropout=0.3
  → output: (B, T, 256) + final hidden
  ↓
Blink Dynamics Head
  → Concatenate LSTM output[:, -1, :] + EAR[:, -1:] → (B, 257)
  → FC(257, 128) → ReLU
  → Blink timing constraint (0.1–0.4s window)
  ↓
Classifier Head
  → FC(256, 128) → ReLU → Dropout(0.5) → FC(128, 2)
  → Output: logits (B, 2)
```

### Loss Function

```
L_total = L_ce(clean)
        + alpha        * L_ce(adversarial)
        + lambda_attn  * L_attn_consistency   # KL div between adjacent frame attentions
        + lambda_blink * L_blink_regularizer  # penalise uncertain preds during blinks
```

Default values: `alpha=0.5`, `lambda_attn=0.1`, `lambda_blink=0.05`

---

## File Contracts (Do Not Break These)

### `src/data/stream_ff_dataset.py`
Produces `.npz` files and `data/metadata.csv`. Each `.npz` contains:
```python
{
  'frames':   np.uint8,   # shape (16, 224, 224, 3)
  'ear':      np.float32, # shape (16,)
  'label':    np.int,     # 0 = real, 1 = fake
  'video_id': np.str_,    # unique identifier
}
```

### `src/data/dataset.py` — `__getitem__` must return:
```python
{
    'frames': torch.Tensor,  # shape (T, 3, 224, 224), float32, normalised [0,1]
    'ear':    torch.Tensor,  # shape (T,), float32
    'label':  int,           # 0 = real, 1 = fake
}
```

### `src/models/lrcn_vit.py` — `forward()` must return:
```python
{
    'logits':       torch.Tensor,  # shape (B, 2)
    'attn_maps':    torch.Tensor,  # shape (B, T, num_heads, num_patches, num_patches)
    'blink_logits': torch.Tensor,  # shape (B, T, 2)
}
```

### `api/inference.py` — `predict_video()` must return:
```python
{
    'label':              str,         # 'REAL' or 'FAKE'
    'confidence':         float,       # probability of fake (0.0–1.0)
    'blink_rate':         float,       # detected blinks per second
    'frame_scores':       list[float], # per-frame fake probability
    'attention_map_path': str | None,  # path to Grad-CAM PNG or None
}
```

---

## Key Source Files & Their Purpose

| File | Purpose |
|------|---------|
| `src/data/stream_ff_dataset.py` | ONE-TIME script: streams FF++ from HuggingFace, saves .npz files |
| `src/data/dataset.py` | PyTorch Dataset reading .npz files + albumentations for train |
| `src/models/backbones.py` | `build_backbone(config)` → timm ViT |
| `src/models/lrcn_vit.py` | Full LRCN-ViT hybrid model |
| `src/train/train.py` | Training loop: AdamW + cosine LR + optional wandb |
| `src/train/adversarial.py` | FGSM and PGD attack implementations |
| `src/eval/evaluate.py` | Metrics: Accuracy, Precision, Recall, F1, AUC |
| `src/eval/ablation.py` | 4 ablation configs compared |
| `src/viz/attention_maps.py` | Grad-CAM on eye-region patches |
| `api/main.py` | FastAPI: `/predict` + `/health` |
| `api/inference.py` | Stateless inference: video bytes → prediction dict |
| `frontend/src/App.jsx` | React UI: upload → result view |

---

## Eye Aspect Ratio (EAR)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

MediaPipe left eye landmark indices: `[33, 160, 158, 133, 153, 144]`
MediaPipe right eye landmark indices: `[362, 385, 387, 263, 373, 380]`
Use average of both eyes.

**Blink threshold:** EAR < 0.2 for ≥ 2 consecutive frames
**Normal blink duration:** 0.1–0.4 seconds (3–12 frames at 30fps)

---

## Config System

```python
from src.utils import load_config
config = load_config('configs/train/aat_pgd.yaml')
# inherits from configs/base.yaml automatically
```

Never hardcode hyperparameters in model/training code — always read from config.

`configs/base.yaml` must include:
```yaml
data:
  metadata_csv: data/metadata.csv
  processed_root: data/processed
  seq_len: 16
  img_size: 224
```

---

## Albumentations (Training Only)

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

Do NOT apply to val or test splits.

---

## FastAPI Setup

### CORS (required — frontend runs on port 5173)
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Model loaded once at startup
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(
        checkpoint="outputs/best.pt",
        config="configs/train/aat_pgd.yaml"
    )
    yield

app = FastAPI(lifespan=lifespan)
```

### Start command
```bash
source .venv311/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend Environment Variables

`frontend/.env`:
```
VITE_API_URL=http://localhost:8000
```

In React:
```js
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
```

---

## Common Bugs to Avoid

1. **Streaming item format**: When streaming HuggingFace datasets, `item['video']` may be a dict `{'bytes': b'...', 'path': '...'}` not a raw path. Always check `isinstance(item['video'], dict)` and read `.bytes`.

2. **Temp file cleanup**: The streaming script writes each video to a temp file for OpenCV. Always delete it with `os.unlink(tmp_path)` after processing — even if an exception occurs. Use try/finally.

3. **EAR not passed to model**: `dataset.py` returns `ear` but it's easy to forget to pass it through the training loop. The blink dynamics head silently fails with all-zero EAR.

4. **ViT reshape bug**: Frames enter as `(B, T, 3, 224, 224)`. Before ViT, reshape to `(B*T, 3, 224, 224)`. After ViT, reshape back to `(B, T, embed_dim)`. Forgetting this causes a shape mismatch in the LSTM.

5. **PGD on full frame**: Apply PGD only to eye-region frames tensor, not to any full-frame input.

6. **Model loaded per-request**: Use FastAPI lifespan to load model once. Loading per request causes 10–30s timeouts.

7. **Split leakage**: Splits in `stream_ff_dataset.py` are assigned by `video_id`, not by sequence. One video's sequences must all go to the same split.

---

## Demo Day Checklist

- [ ] `python -m src.eval.evaluate --checkpoint outputs/best.pt` runs cleanly
- [ ] `/health` returns `{"status": "ok"}`
- [ ] Upload known fake video → returns FAKE with >70% confidence
- [ ] Upload real webcam video → returns REAL
- [ ] Frame chart renders
- [ ] App shows friendly error if API is not running
- [ ] Short (<3 second) video doesn't crash the app

---

## Evaluation Targets

| Metric | Baseline (Gazi 2021 LRCN) | Target (AAT-ViT) |
|--------|--------------------------|-----------------|
| Accuracy | ~88% | >91% |
| AUC | ~0.90 | >0.93 |
| F1 | ~0.87 | >0.90 |

On FaceForensics++ c23, video-disjoint test split.