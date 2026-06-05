from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import zipfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.data.eye_extract import (
    extract_sequences_from_video_bytes,
    extract_sequences_from_video_path,
)

HF_DATASET = "bitmind/FaceForensicsC23"
HF_ZIP_FILE = "FaceForensics++_C23.zip"
REAL_PATH_MARKER = "/real/"
FAKE_PATH_MARKER = "/fake/deepfakes/"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _video_id_from_path(repo_path: str) -> str:
    """Extract stable video id from zip:// or filesystem paths."""
    normalized = repo_path.replace("\\", "/")
    if "://" in normalized:
        inner = normalized.split("://", 1)[1].split("::")[0]
        return Path(inner).stem
    return Path(normalized).stem


def _is_fake_path(path_lower: str) -> bool:
    return FAKE_PATH_MARKER in path_lower and "deepfakedetection" not in path_lower


def _is_real_path(path_lower: str) -> bool:
    return REAL_PATH_MARKER in path_lower


def _video_path_from_item(item: dict) -> str:
    video = item.get("video")
    if isinstance(video, dict):
        return str(video.get("path", ""))
    return str(video or "")


def _video_bytes_from_item(item: dict) -> bytes | None:
    import fsspec

    video = item.get("video")
    if not isinstance(video, dict):
        return None

    raw = video.get("bytes")
    if raw is not None:
        return raw

    path = video.get("path")
    if not path:
        return None

    if str(path).startswith("zip://"):
        with fsspec.open(path, "rb") as handle:
            return handle.read()

    if os.path.exists(path):
        return Path(path).read_bytes()

    return None


def iter_hf_stream(num_real: int, num_fake: int):
    """
    Stream one HF dataset row at a time (no torchcodec decode).
    Each video is loaded into RAM, processed, then discarded.
    """
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.decode(False)

    if num_real == 0 and num_fake > 0:
        print("Skipping first ~1000 rows (DeepFakeDetection) to reach /fake/Deepfakes/ ...")
        ds = ds.skip(1000)
    elif num_fake == 0 and num_real > 0:
        print("Skipping first ~6000 rows to reach /real/ videos ...")
        ds = ds.skip(6000)

    real_count = 0
    fake_count = 0
    scanned = 0

    for item in ds:
        scanned += 1
        if scanned % 250 == 0 and (num_real > 0 or num_fake > 0):
            tqdm.write(f"  scanned {scanned} rows (real={real_count}, fake={fake_count})")

        video_path_str = _video_path_from_item(item)
        path_lower = video_path_str.lower()
        is_real = _is_real_path(path_lower) and real_count < num_real
        is_fake = _is_fake_path(path_lower) and fake_count < num_fake
        if not is_real and not is_fake:
            continue

        label = 0 if is_real else 1
        video_bytes = _video_bytes_from_item(item)
        if video_bytes is None:
            continue

        yield video_path_str, video_bytes, label

        if is_real:
            real_count += 1
        else:
            fake_count += 1
        if real_count >= num_real and fake_count >= num_fake:
            break


def _resolve_hub_zip() -> Path:
    zip_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=HF_ZIP_FILE,
        repo_type="dataset",
    )
    return Path(zip_path)


def iter_zip_members(num_real: int, num_fake: int, zip_path: Path):
    """Read one video at a time from the HF-hosted zip (cached under ~/.cache/huggingface)."""
    real_count = 0
    fake_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            path = member.replace("\\", "/")
            if Path(path).suffix.lower() not in VIDEO_SUFFIXES:
                continue

            path_lower = path.lower()
            is_real = _is_real_path(path_lower) and real_count < num_real
            is_fake = _is_fake_path(path_lower) and fake_count < num_fake
            if not is_real and not is_fake:
                continue

            label = 0 if is_real else 1
            with zf.open(member) as handle:
                video_bytes = handle.read()

            yield path, video_bytes, label

            if is_real:
                real_count += 1
            else:
                fake_count += 1
            if real_count >= num_real and fake_count >= num_fake:
                break


def iter_videos(num_real: int, num_fake: int):
    """Stream via HF datasets API; fall back to zip cache if needed."""
    try:
        yield from iter_hf_stream(num_real, num_fake)
        return
    except Exception as exc:
        print(f"HF streaming failed ({exc}). Trying zip cache (may be slow first time).")
        zip_path = _resolve_hub_zip()
        yield from iter_zip_members(num_real, num_fake, zip_path)


def _append_metadata_row(metadata_path: Path, row: dict) -> None:
    write_header = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["npz_path", "label", "video_id", "split"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def stream_and_extract(
    out_root: str | Path,
    num_real: int = 200,
    num_fake: int = 200,
    seq_len: int = 16,
    metadata_csv: str | Path = "data/metadata.csv",
    seed: int = 42,
    append_metadata: bool = False,
) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Streaming from HuggingFace ({HF_DATASET}) — one video at a time, "
        "no full zip download into project folder."
    )
    print("Videos are processed in RAM; only .npz files are saved under data/processed/.")

    metadata_rows: list[dict] = []
    real_count = 0
    fake_count = 0

    pbar = tqdm(total=num_real + num_fake, desc="Streaming videos")
    for repo_path, video_bytes, label in iter_videos(num_real, num_fake):
        video_id = _video_id_from_path(repo_path)
        sequences = extract_sequences_from_video_bytes(
            video_bytes, label, video_id, seq_len=seq_len
        )
        if not sequences:
            print(f"Warning: no face/eye sequences in {repo_path}, skipping.")
            continue

        for seq in sequences:
            npz_path = out_root / f"{seq['video_id']}.npz"
            np.savez_compressed(
                npz_path,
                frames=seq["frames"],
                ear=seq["ear"],
                label=np.array(seq["label"]),
                video_id=np.array(seq["video_id"]),
            )
            row = {
                "npz_path": str(npz_path.resolve()),
                "label": label,
                "video_id": video_id,
                "split": "train",
            }
            metadata_rows.append(row)
            # Write progressively so Ctrl+C doesn't lose completed videos
            _append_metadata_row(metadata_path, row)

        if label == 0:
            real_count += 1
        else:
            fake_count += 1
        pbar.update(1)

    pbar.close()

    if append_metadata and metadata_path.exists():
        import pandas as pd

        existing = pd.read_csv(metadata_path).to_dict(orient="records")
        metadata_rows = existing + metadata_rows

    if not metadata_rows:
        raise RuntimeError(
            "No sequences extracted. Ensure `hf auth login` is done and paths contain "
            "/real/ or /fake/Deepfakes/. For fake-only runs, wait 1–2 min or use --append "
            "if adding to an existing metadata.csv."
        )

    rng = np.random.default_rng(seed)
    unique_ids = sorted({row["video_id"] for row in metadata_rows})
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    train_ids = set(unique_ids[: int(0.7 * n)])
    val_ids = set(unique_ids[int(0.7 * n) : int(0.85 * n)])
    for row in metadata_rows:
        if row["video_id"] in train_ids:
            row["split"] = "train"
        elif row["video_id"] in val_ids:
            row["split"] = "val"
        else:
            row["split"] = "test"

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["npz_path", "label", "video_id", "split"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDone! {real_count} real + {fake_count} fake videos processed.")
    print(f"Total sequences: {len(metadata_rows)}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Sequences saved to: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--metadata-csv", default="data/metadata.csv")
    parser.add_argument("--num-real", type=int, default=200)
    parser.add_argument("--num-fake", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new sequences to existing metadata.csv instead of overwriting.",
    )
    args = parser.parse_args()
    stream_and_extract(
        out_root=args.out_root,
        num_real=args.num_real,
        num_fake=args.num_fake,
        seq_len=args.seq_len,
        metadata_csv=args.metadata_csv,
        seed=args.seed,
        append_metadata=args.append,
    )


if __name__ == "__main__":
    main()
