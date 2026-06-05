from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]


def eye_bbox(landmarks, image_w: int, image_h: int, indices: List[int], pad: int = 8) -> Tuple[int, int, int, int]:
    pts = np.array([(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in indices])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return max(0, x1 - pad), max(0, y1 - pad), min(image_w, x2 + pad), min(image_h, y2 + pad)


def process_frame(frame: np.ndarray, face_mesh) -> Tuple[np.ndarray, float]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, 0.0

    h, w = frame.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    lx1, ly1, lx2, ly2 = eye_bbox(lm, w, h, LEFT_EYE_IDX)
    rx1, ry1, rx2, ry2 = eye_bbox(lm, w, h, RIGHT_EYE_IDX)

    left = frame[ly1:ly2, lx1:lx2]
    right = frame[ry1:ry2, rx1:rx2]
    if left.size == 0 or right.size == 0:
        return None, 0.0

    left = cv2.resize(left, (112, 112))
    right = cv2.resize(right, (112, 112))
    eye_pair = np.concatenate([left, right], axis=1)

    left_open = abs(lm[LEFT_EYE_IDX[2]].y - lm[LEFT_EYE_IDX[3]].y)
    right_open = abs(lm[RIGHT_EYE_IDX[2]].y - lm[RIGHT_EYE_IDX[3]].y)
    ear_proxy = float((left_open + right_open) / 2.0)
    return eye_pair, ear_proxy


def sample_sequences(frames: List[np.ndarray], ear: List[float], seq_len: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x, b = [], []
    for i in range(0, len(frames) - seq_len + 1, seq_len):
        x.append(np.stack(frames[i : i + seq_len], axis=0))
        b.append(np.array(ear[i : i + seq_len], dtype=np.float32))
    return x, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    records = []
    for row in tqdm(df.to_dict(orient="records"), desc="Extracting eye sequences"):
        frame_dir = Path(row["frame_dir"])
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        eye_frames: List[np.ndarray] = []
        ear_values: List[float] = []
        for fp in frame_paths:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            eye_img, ear = process_frame(frame, mesh)
            if eye_img is None:
                continue
            eye_frames.append(eye_img)
            ear_values.append(ear)

        sequences, blink = sample_sequences(eye_frames, ear_values, args.sequence_length)
        video_id = Path(row["video_path"]).stem
        saved = 0
        for idx, (seq, blink_seq) in enumerate(zip(sequences, blink)):
            out_dir = args.out_root / "sequences" / row["dataset"] / row["split"] / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{idx:03d}.npz"
            np.savez_compressed(out_path, frames=seq, blink=blink_seq, label=row["label"])
            saved += 1
        if saved > 0:
            records.append(
                {
                    "dataset": row["dataset"],
                    "video_path": row["video_path"],
                    "identity": row["identity"],
                    "split": row["split"],
                    "label": row["label"],
                    "sequence_count": saved,
                    "sequence_dir": str((args.out_root / "sequences" / row["dataset"] / row["split"] / video_id).resolve()),
                }
            )

    out_csv = args.out_root / "metadata_sequences.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved sequence metadata: {out_csv}")


if __name__ == "__main__":
    main()
