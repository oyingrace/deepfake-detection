# Reproducibility Checklist

- Record commit hash for every run.
- Log full config files for training and evaluation.
- Store random seed values for Python, NumPy, and PyTorch.
- Keep dataset split manifests identity-disjoint and immutable.
- Save model checkpoints with timestamp and metric tags.
- Track software versions (`python3 --version`, `pip freeze`).
- Archive key outputs: metrics JSON, confusion matrix, ROC, attention maps.
- Document hardware profile (GPU model, VRAM, CUDA/cuDNN versions).
