from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.inference import load_model, predict_video
from api.schemas import HealthResponse, PredictionResponse

CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "outputs/best.pt")
CONFIG = os.getenv("MODEL_CONFIG", "configs/train/cpu_fast.yaml")
DEVICE = os.getenv("DEVICE", "cpu")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(CHECKPOINT)
    if checkpoint.exists():
        print(f"Loading model from {checkpoint}...")
        _state["model"] = load_model(str(checkpoint), CONFIG, DEVICE)
        _state["model_loaded"] = True
        print("Model loaded.")
    else:
        print(f"WARNING: checkpoint not found at {checkpoint}. Running in mock mode.")
        _state["model"] = None
        _state["model_loaded"] = False
    yield
    _state.clear()


app = FastAPI(title="Deepfake Detector API", lifespan=lifespan)

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_cors_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.get("/")
def root():
    return {"service": "Deepfake Detector API", "health": "/health", "predict": "POST /predict"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_state.get("model_loaded", False))


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("video/"):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise HTTPException(status_code=400, detail="Please upload a video file.")

    video_bytes = await file.read()
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model = _state.get("model")

    if model is None:
        # Mock response when no checkpoint is available yet
        return PredictionResponse(
            label="FAKE",
            confidence=0.82,
            blink_rate=0.3,
            frame_scores=[0.75, 0.84, 0.87, 0.80],
            attention_map_url=None,
        )

    result = predict_video(video_bytes, model, device=DEVICE)
    return PredictionResponse(**result)
