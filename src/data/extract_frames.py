from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def extract_video_frames(video_path: Path, out_dir: Path, every_n: int = 2) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            out_path = out_dir / f"{written:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            written += 1
        idx += 1
    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--every-n", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    records = []
    for row in tqdm(df.to_dict(orient="records"), desc="Extracting frames"):
        vpath = Path(row["video_path"])
        video_id = vpath.stem
        out_dir = args.out_root / "frames" / row["dataset"] / video_id
        count = extract_video_frames(vpath, out_dir, every_n=args.every_n)
        row["frame_dir"] = str(out_dir)
        row["frame_count"] = count
        records.append(row)

    out_csv = args.out_root / "metadata_frames.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved frame metadata: {out_csv}")


if __name__ == "__main__":
    main()
