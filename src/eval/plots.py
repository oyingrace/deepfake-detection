from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-true", required=True, help="NumPy .npy path for ground truth labels")
    parser.add_argument("--y-pred", required=True, help="NumPy .npy path for predicted labels")
    parser.add_argument("--y-prob", required=True, help="NumPy .npy path for predicted probabilities")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    save_evaluation_plots(
        np.load(args.y_true),
        np.load(args.y_pred),
        np.load(args.y_prob),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
