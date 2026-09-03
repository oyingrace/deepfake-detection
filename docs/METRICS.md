# Evaluation Metrics

This project uses [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) to measure how well the deepfake detector performs on a **labeled test set**.

## Labels

| Value | Meaning |
|-------|---------|
| `0` | Real video |
| `1` | Fake (deepfake) video |

Precision, recall, and F1 are reported for the **fake** class (`1`) by default.

## What each metric means

| Metric | What it measures | Good when… |
|--------|------------------|------------|
| **Accuracy** | Fraction of correct predictions (real or fake) | Higher is better |
| **Precision** | Of videos called fake, how many were actually fake | Fewer false alarms |
| **Recall** | Of all real fakes, how many the model caught | Fewer missed fakes |
| **F1** | Balance between precision and recall | Higher is better |
| **AUC** | How well the model ranks fakes above reals (uses probability scores, not just hard labels) | Closer to 1.0 is better |

**Example:** High recall but low precision means the model catches most fakes but also flags many real videos as fake.

## How metrics are computed

1. Load the **test split** from `data/metadata.csv` (via `EyeSequenceDataset`).
2. Run the trained model on every test sample.
3. Collect three arrays:
   - `y_true` — ground-truth labels from the dataset
   - `y_pred` — predicted class (`0` or `1`)
   - `y_prob` — probability the video is fake (`softmax` score for class `1`)
4. Pass those arrays to scikit-learn in `src/eval/evaluate.py`.

During **training**, only validation **accuracy** is logged each epoch (`src/train/train.py`). Full metrics are run after training with the evaluate script below.

## Run evaluation (numbers only)

```bash
source .venv311/bin/activate
python -m src.eval.evaluate \
  --checkpoint outputs/best.pt \
  --config configs/train/aat_pgd.yaml
```

Example output:

```json
{
  "accuracy": 0.91,
  "precision": 0.89,
  "recall": 0.93,
  "f1": 0.91,
  "auc": 0.96
}
```

Results are also saved to `outputs/eval/metrics.json`.

## Visual representation (plots)

Add `--plots` to generate PNG charts from the same run:

```bash
python -m src.eval.evaluate \
  --checkpoint outputs/best.pt \
  --config configs/train/aat_pgd.yaml \
  --plots
```

This writes:

| File | What you see |
|------|----------------|
| `outputs/eval/confusion_matrix.png` | Grid of true vs predicted counts (correct top-left / bottom-right, errors off-diagonal) |
| `outputs/eval/roc_curve.png` | Trade-off between true-positive and false-positive rate; curve closer to the top-left is better |
| `outputs/eval/metrics.json` | Numeric metrics |
| `outputs/eval/y_*.npy` | Raw label arrays (for re-plotting or custom analysis) |


### Re-generate plots without re-running the model

If you already have saved arrays:

```bash
python -m src.eval.plots \
  --y-true outputs/eval/y_true.npy \
  --y-pred outputs/eval/y_pred.npy \
  --y-prob outputs/eval/y_prob.npy \
  --out-dir outputs/eval
```

### Custom output folder

```bash
python -m src.eval.evaluate \
  --checkpoint outputs/best.pt \
  --config configs/train/aat_pgd.yaml \
  --out-dir outputs/my_run \
  --plots
```

## Prerequisites

- A trained checkpoint at `outputs/best.pt` (or pass another path).
- Test data: `data/metadata.csv` with a `test` split and `.npz` eye-sequence files on disk.
- Dependencies: `scikit-learn`, `matplotlib`, `seaborn` (see `requirements.txt`).

## Where the code lives

| File | Role |
|------|------|
| `src/eval/evaluate.py` | Runs model on test set, computes metrics, saves JSON/arrays |
| `src/eval/plots.py` | Builds confusion matrix and ROC curve images |
| `src/train/train.py` | Training-time validation accuracy only |

## Further reading

- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- `docs/results_template.md` — table template for thesis/report numbers
