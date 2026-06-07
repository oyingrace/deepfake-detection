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

FastAPI inference service for the Enhanced Deepfake Detector (LRCN + ViT, blink-aware temporal modeling).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and model load state |
| `POST` | `/predict` | Upload a video (multipart form field `file`) |

---

## Deploy (GitHub → Space auto-sync) — recommended

Hugging Face Spaces **do not** have a “connect GitHub” button like Render.  
Instead, use a **GitHub Action** that mirrors your repo to the Space on every push.

### One-time setup

1. **Create a Hugging Face token** (write access):  
   https://huggingface.co/settings/tokens

2. **Add it to GitHub** as a secret:  
   Repo → **Settings → Secrets and variables → Actions → New repository secret**  
   - Name: `HF_TOKEN`  
   - Value: your HF token

3. **Push this repo to GitHub** (includes `.github/workflows/sync-to-hf-space.yml`).

4. Watch the Action run: GitHub → **Actions** → “Sync to Hugging Face Space”.

5. When the build finishes, your API is at:

   **https://devqueen-deepfake-server.hf.space**

   (Check the exact URL on your Space page — it may differ slightly.)

6. **Space Settings → Variables** (runtime):

   | Variable | Example |
   |----------|---------|
   | `ALLOWED_ORIGINS` | `https://your-app.vercel.app,http://localhost:5173` |

7. Test:

   ```bash
   curl https://devqueen-deepfake-server.hf.space/health
   ```

---

## Deploy (manual git push)

If you prefer not to use GitHub Actions:

```bash
./scripts/setup_hf_space.sh DevQueen/deepfake-server
huggingface-cli login          # paste write token
git push huggingface main
```

Use a [write token](https://huggingface.co/settings/tokens) when prompted for a password.

---

## Connect the Vercel frontend

Update `frontend/vercel.json` rewrites to your HF Space URL:

```json
"destination": "https://devqueen-deepfake-server.hf.space/predict"
```

Push to GitHub → Vercel redeploys.

---

## Notes

- Free CPU tier: **16 GB RAM** (enough for PyTorch + MediaPipe).
- Space sleeps after ~48h idle; first request may take 1–2 min to wake.
- `outputs/best.pt` (~22 MB) is included in the Docker image at build time.
- Render uses `deploy/Dockerfile`; Hugging Face uses the root `Dockerfile`.
