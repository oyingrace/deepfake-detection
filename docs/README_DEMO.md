# Demo — Running the Deepfake Detector

## Prerequisites

- Python 3.11 virtual environment at `.venv311` (already set up)
- Node.js ≥ 18 (for the frontend)
- A trained model checkpoint at `outputs/best.pt`

## Quick Start (two terminals)

### Terminal 1 — Inference API

```bash
cd deepfake-detection
./scripts/start_api.sh
```

The API starts at **http://localhost:8000**  
Check it's alive: `curl http://localhost:8000/health`

### Terminal 2 — Web Frontend

```bash
cd deepfake-detection
./scripts/start_frontend.sh
```

Open **http://localhost:5173** in your browser.

---

## Using the Demo

1. Open **http://localhost:5173**
2. Drag & drop a video (MP4, MOV, AVI, MKV, WebM) or click to browse
3. Click **Analyse Video**
4. The result panel shows:
   - **REAL** (green) or **FAKE** (red) verdict
   - Confidence score and blink rate
   - Per-sequence probability chart

---

## No checkpoint yet?

The API runs in **mock mode** until `outputs/best.pt` exists.  
Mock mode returns a hard-coded FAKE prediction so you can test the UI  
before training finishes.

Once training completes (or after epoch 1), the checkpoint is saved automatically at `outputs/best.pt`.

---

## API Reference

### `POST /predict`

Upload a video file (multipart form-data):

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_video.mp4"
```

Response:

```json
{
  "label": "FAKE",
  "confidence": 0.82,
  "blink_rate": 0.3,
  "frame_scores": [0.75, 0.84, 0.87, 0.80],
  "attention_map_url": null
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Fresh Machine Setup

```bash
# 1. Clone repo and cd into it
# 2. Create Python 3.11 venv
pyenv local 3.11.11
python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
pip install -r api/requirements.txt

# 3. Install frontend deps
cd frontend && npm install && cd ..

# 4. Start API + frontend (two terminals, as above)
```
