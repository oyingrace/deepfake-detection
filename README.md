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

FastAPI inference service for the [Enhanced Deepfake Detector](https://github.com/oyingrace/deepfake-detection) (LRCN + ViT, blink-aware temporal modeling).

> **GitHub project docs:** see [.github/README.md](.github/README.md)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and model load state |
| `POST` | `/predict` | Upload a video (multipart form field `file`) |

## Example

```bash
curl https://devqueen-deepfake-server.hf.space/health
curl -X POST https://devqueen-deepfake-server.hf.space/predict -F "file=@video.mp4"
```

## Runtime variables

| Variable | Description |
|----------|-------------|
| `ALLOWED_ORIGINS` | Comma-separated frontend URLs for CORS (e.g. your Vercel app) |
| `DEVICE` | `cpu` (default) |

## Deploy notes

- Synced from GitHub via `.github/workflows/sync-to-hf-space.yml`
- Full setup guide: [huggingface/README.md](huggingface/README.md)
