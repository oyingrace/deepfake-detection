from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str                      # "REAL" or "FAKE"
    confidence: float               # probability of fake (0.0–1.0)
    blink_rate: float               # detected blinks per second
    frame_scores: list[float]       # per-sequence fake probability
    attention_map_url: str | None   # URL to Grad-CAM image, or None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
