# Results Table Template

## Main Metrics

| Model Variant | Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Baseline LRCN | Celeb-DF |  |  |  |  |  |
| LRCN + ViT | Celeb-DF |  |  |  |  |  |
| LRCN + ViT + PGD | Celeb-DF |  |  |  |  |  |
| LRCN + ViT + AAT | Celeb-DF |  |  |  |  |  |

## Ablation

| Ablation | Accuracy | F1 | AUC | Notes |
|---|---:|---:|---:|---|
| Full Model |  |  |  |  |
| No AAT |  |  |  |  |
| No ViT |  |  |  |  |
| No Blink Regularizer |  |  |  |  |

## Robustness Sweep (FGSM/PGD)

| Attack | Epsilon | Accuracy | F1 | AUC |
|---|---:|---:|---:|---:|
| FGSM | 0.01 |  |  |  |
| FGSM | 0.02 |  |  |  |
| PGD | 0.01 |  |  |  |
| PGD | 0.03 |  |  |  |
