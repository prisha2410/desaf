# DeSAF: Results Summary

All results are on the official PAN 2025 splits (`pan2025_full`). Test-set numbers are touched exactly once per checkpoint reported.

## Dataset

| Tier | Split | Documents | Boundaries | Changes | Change rate |
|---|---|---|---|---|---|
| Easy | Train | 4,200 | 48,402 | 10,224 | 21.1% |
| Easy | Validation | 900 | 10,247 | 2,201 | 21.5% |
| Easy | Test | 900 | 10,328 | 2,238 | 21.7% |
| Medium | Train | 4,200 | 58,817 | 12,503 | 21.3% |
| Medium | Validation | 900 | 12,759 | 2,750 | 21.6% |
| Medium | Test | 900 | 12,966 | 2,797 | 21.6% |
| Hard | Train | 4,200 | 51,062 | 8,815 | 17.3% |
| Hard | Validation | 900 | 10,648 | 1,906 | 17.9% |
| Hard | Test | 900 | 11,080 | 1,911 | 17.2% |
| **Total** | | **18,000** | **226,309** | **45,345** | **20.0%** |

## Headline Hard-tier results (official test, single seed)

| Model | Hard-tier test F1 |
|---|---|
| Plain model (no contrastive loss) | 0.7067 |
| Contrastive model (SupCon) | 0.7097 |
| Three-way ensemble (plain + contrastive + LightGBM) | **0.7350** |

## Full per-tier breakdown, seed 43 (official test)

| Tier | F1 | WindowDiff | Precision | Recall |
|---|---|---|---|---|
| Easy | 0.9286 | 0.0732 | 0.9421 | 0.9155 |
| Medium | 0.7198 | 0.2381 | 0.7806 | 0.6679 |
| Hard | 0.7198 | 0.2688 | 0.7080 | 0.7321 |
| **Macro** | **0.7894** | | | |

## Tier B architectural ablation (validation, single seed 42)

| Technique | Easy F1 | Medium F1 | Hard F1 | Notes |
|---|---|---|---|---|
| XBM cross-batch memory | 0.9168 | 0.7495 | 0.6726 | best Medium among first group |
| R-Drop consistency regularization | 0.9104 | **0.7560** | 0.6825 | best Medium result overall |
| Self-adjusting Dice loss | 0.9274 | 0.7382 | 0.6741 | |
| FiLM affine fusion | 0.9173 | 0.7271 | **0.7091** | best single-model Hard result |
| Transformer layer on BiLSTM | broken | broken | broken | negative result, did not train |

## Comparison to published PAN 2025 systems

| System | Easy F1 | Medium F1 | Hard F1 |
|---|---|---|---|
| Team wqd | ~0.92-0.95 | ~0.82-0.83 | ~0.83 |
| SCL-DeBERTa | ~0.92-0.95 | ~0.82-0.83 | ~0.83 |
| better_call_claude | ~0.92-0.95 | ~0.82-0.83 | ~0.72 |
| **DeSAF (this project)** | **0.9286** | **0.7198** | **0.7350 (ensemble)** |

## Interpretability: gate reliance, correct vs incorrect predictions

| Group | N boundaries | Mean gate value |
|---|---|---|
| Correctly classified | 30,495 | 0.2246 |
| Incorrectly classified | 3,159 | 0.1959 |
| Difference | | +0.0287 (not yet significance-tested) |

## Status

- Multi-seed (seeds 43, 44), curriculum schedule ablation, factorial ablation, and FiLM+R-Drop combination: in progress.
- Significance testing: not started.
- Full interpretability writeup: in progress.
