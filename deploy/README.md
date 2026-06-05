# Render backend deploy bundle

Self-contained Docker build for the FastAPI inference API.

## What's in this folder

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the API image |
| `.dockerignore` | Keeps the image smaller |
| `requirements.txt` | Inference-only Python deps (no Hugging Face) |
| `sync_bundle.sh` | Copies `api/`, `src/`, `configs/`, `best.pt` from the repo root |
| `api/`, `src/`, `configs/`, `outputs/` | Created by `sync_bundle.sh` (not always in git) |

## Before first deploy

From the **repo root**:

```bash
./deploy/sync_bundle.sh
```

Re-run after you change API code, model weights, or configs.

## Test locally

```bash
cd deploy
docker build -t deepfake-api .
docker run -p 8000:8000 deepfake-api
curl http://localhost:8000/health
```

## Render setup

1. Push repo to GitHub (include `outputs/best.pt` or run sync in CI before build).
2. Render → **New Web Service** → connect repo.
3. Settings:
   - **Root Directory:** `deploy`
   - **Environment:** Docker
   - **Dockerfile Path:** `Dockerfile` (default)
4. **Environment variables:**

   | Key | Example |
   |-----|---------|
   | `ALLOWED_ORIGINS` | `https://your-frontend.onrender.com,http://localhost:5173` |
   | `DEVICE` | `cpu` |

5. Deploy. API URL: `https://<service>.onrender.com`

## Notes

- Free tier: 512MB RAM, cold starts, request timeouts — use **short videos** for demos.
- `best.pt` is ~22MB; commit it or ensure sync runs before Docker build.
- Frontend: deploy separately on Render **Static Site** with `VITE_API_URL` pointing to this service.
