"""Eye sequence extraction for inference (no Hugging Face dependencies)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(landmarks, eye_indices: list[int]) -> float:
    p = [landmarks[i] for i in eye_indices]
    a = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    b = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    c = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    return float((a + b) / (2.0 * c + 1e-6))


def _extract_sequences_from_capture(
    cap: cv2.VideoCapture,
    label: int,
    video_id: str,
    seq_len: int,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> list[dict]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / 10))

    all_frames: list[np.ndarray] = []
    all_ears: list[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                left_ear = compute_ear(lm, LEFT_EYE)
                right_ear = compute_ear(lm, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                eye_pts = [lm[i] for i in LEFT_EYE + RIGHT_EYE]
                xs = [int(p.x * w) for p in eye_pts]
                ys = [int(p.y * h) for p in eye_pts]
                x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
                y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224))
                    all_frames.append(crop.astype(np.uint8))
                    all_ears.append(ear)
        frame_idx += 1

    sequences: list[dict] = []
    for i in range(0, len(all_frames) - seq_len + 1, seq_len):
        frames = np.stack(all_frames[i : i + seq_len]).astype(np.uint8)
        ears = np.array(all_ears[i : i + seq_len], dtype=np.float32)
        sequences.append(
            {
                "frames": frames,
                "ear": ears,
                "label": label,
                "video_id": f"{video_id}_seq{i // seq_len:03d}",
            }
        )
    return sequences


def extract_sequences_from_video_path(
    video_path: str | Path,
    label: int,
    video_id: str,
    seq_len: int = 16,
) -> list[dict]:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video_path))
    try:
        return _extract_sequences_from_capture(cap, label, video_id, seq_len, face_mesh)
    finally:
        cap.release()
        face_mesh.close()


def extract_sequences_from_video_bytes(
    video_bytes: bytes,
    label: int,
    video_id: str,
    seq_len: int = 16,
) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        return extract_sequences_from_video_path(tmp_path, label, video_id, seq_len=seq_len)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
