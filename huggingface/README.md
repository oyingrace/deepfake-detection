---
title: Deepfake Detector API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Deepfake Detector API

FastAPI inference service for the [Enhanced Deepfake Detector](https://github.com/oyingrace/deepfake-detection) project (LRCN + ViT, blink-aware temporal modeling).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and model load state |
| `POST` | `/predict` | Upload a video (multipart form field `file`) |

## Deploy this Space from GitHub

### Option A — Repo root (recommended)

1. Create a **Docker** Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space Settings → Repository** → connect `oyingrace/deepfake-detection` (branch `main`).
3. Leave **Root directory** empty. Hugging Face uses the root `Dockerfile`.
4. Continue at step 4 below.

### Option B — `huggingface/` subfolder (uses this README as the Space card)

1. From the repo root: `./huggingface/sync_bundle.sh` then commit the synced files.
2. Create a Docker Space and connect GitHub with **Root directory** = `huggingface`.
3. Continue at step 4 below.

### Configure & test

4. **Space Settings → Variables** (runtime):

   | Variable | Example |
   |----------|---------|
   | `ALLOWED_ORIGINS` | `https://your-app.vercel.app,http://localhost:5173` |

5. Wait for the build (~5–10 min). Your API URL:

   `https://<your-username>-deepfake-detection.hf.space`

6. Test: `curl https://<your-username>-deepfake-detection.hf.space/health`

## Connect the Vercel frontend

After the Space is live, update `frontend/vercel.json` rewrites to your HF URL:

```json
"destination": "https://<your-username>-deepfake-detection.hf.space/predict"
```

Push to GitHub → Vercel redeploys.

Or set `VITE_API_URL` on Vercel to the HF Space URL (and set `ALLOWED_ORIGINS` on the Space).

## Notes

- Free CPU tier: **16 GB RAM** (enough for PyTorch + MediaPipe inference).
- Space sleeps after ~48 hours of inactivity; first request wakes it (may take 1–2 min).
- Model weights: `outputs/best.pt` (~22 MB) are copied into the image at build time.
