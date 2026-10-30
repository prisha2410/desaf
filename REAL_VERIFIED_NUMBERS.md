# DeSAF — The Actual Verified Numbers (pan2025_full ONLY)

Every number below was produced live this session, via real terminal output pasted
into this conversation, from `train_deberta_fusion.py` / `eval_official_test_2025.py`,
reading from `data/raw/pan2025_full/{tier}/{split}/`. Nothing here comes from the old
2024+2025-combined pipeline, the stale GitHub mirror, or any search-tool paraphrase.
If a number isn't in this document, it is NOT verified for use in the paper — go back
to the real terminal output or rerun the script before trusting anything else.

---

## 1. Original paper headline (already in the submitted/rejected DELCON draft)

Single run, seed43, 2-2-8 curriculum, contrastive on, `--no-char-ngrams` (193-dim),
threshold 0.50:

| Tier | F1 | WindowDiff | Precision | Recall |
|---|---|---|---|---|
| Easy | 0.9286 | 0.0732 | 0.9421 | 0.9155 |
| Medium | 0.7198 | 0.2381 | 0.7806 | 0.6679 |
| Hard | 0.7198 | 0.2688 | 0.7080 | 0.7321 |
| **Macro** | **0.7894** | | | |

### Hard-tier ensemble table (also in the submitted draft)

| Configuration | Hard F1 |
|---|---|
| Base fusion model | 0.7067 |
| + Supervised contrastive loss | 0.7097 |
| Three-way late-fusion ensemble (base + contrastive + LightGBM on surface-similarity features) | **0.7350** |

---

## 2. Multi-seed results (this session, task #2)

Same config as seed43 (2-2-8, contrastive on, `--no-char-ngrams`), only the seed changed:

| Seed | Easy | Medium | Hard | Threshold |
|---|---|---|---|---|
| 43 | 0.9286 | 0.7198 | 0.7198 | 0.50 |
| 44 | 0.9315 | 0.7229 | 0.7019 | 0.52 |
| 45 | 0.9181 | 0.7155 | 0.7182 | 0.64 |
| **Mean ± SD** | **0.9261 ± 0.0058** | **0.7194 ± 0.0031** | **0.7133 ± 0.0080** | |

---

## 3. Curriculum schedule ablation (task #4)

Same architecture, 12 total epochs, different phase splits:

| Schedule | Seed | Easy | Medium | Hard |
|---|---|---|---|---|
| 2-2-8 (original) | 44 | 0.9315 | 0.7229 | 0.7019 |
| 4-4-4 | 42 | 0.9349 | 0.7155 | 0.6993 |
| 6-3-3 | 42 | 0.9344 | 0.7146 | 0.6765 |

---

## 4. Factorial ablation: stylometric fusion × contrastive loss (task #5)

Seed 42 held fixed across all four cells, 2-2-8 curriculum:

| Style | Contrastive | Easy | Medium | Hard | Threshold |
|---|---|---|---|---|---|
| ✓ | ✓ | 0.9242 | 0.7155 | 0.7065 | 0.50 |
| ✓ | ✗ | 0.9238 | 0.7081 | 0.7165 | 0.66 |
| ✗ | ✓ | 0.9173 | 0.7125 | 0.7049 | 0.60 |
| ✗ | ✗ | 0.9136 | 0.7183 | 0.7054 | — |

(Note: the cell originally mislabeled "factorial_nofusion_nocontrastive" is actually
style-ON/contrastive-OFF — the training command never had `--no-style` in it. The
true no-style/no-contrastive cell is the separate `true_nostyle_nocontrastive` run.)

---

## 4b. Isolated gate ablation: gate vs. raw concat (official test)

Same config as seed43 headline (2-2-8, contrastive on, `--no-char-ngrams`, style ON), only
`--attention-fusion` removed, so fusion is raw `[h ; delta]` (no gate, and no LayerNorm on delta).
Checkpoint `checkpoints/ablation_concat_contrastive_seed43.pt`, val-tuned threshold 0.44.

| Fusion | Seed | Easy | Medium | Hard | Macro | Threshold |
|---|---|---|---|---|---|---|
| Gate (headline) | 43 | 0.9286 | 0.7198 | 0.7198 | 0.7894 | 0.50 |
| Raw concat | 43 | 0.9351 | 0.7140 | 0.7209 | 0.7900 | 0.44 |
| Delta (concat - gate) | | +0.0065 | -0.0058 | +0.0011 | +0.0006 | |

Validation (same run): overall 0.7948; Easy 0.9311 / Medium 0.7432 / Hard 0.7126.
Predictions saved: `checkpoints/concat_seed43/test_predictions.npz`.
Seeds 44/45 for concat not yet run.

Significance, gate_seed43 vs concat_seed43 (`compute_significance.py`, 10,000 doc-level
bootstrap resamples + McNemar). gate_seed43 F1s reproduced exactly (0.9286/0.7198/0.7198),
confirming `checkpoints/seed43/test_predictions.npz` is valid.

| Tier | Delta (gate - concat) | 95% CI | Bootstrap p | McNemar p |
|---|---|---|---|---|
| Easy | -0.0065 | [-0.0138, 0.0011] | 0.0902 | 0.0750 |
| Medium | +0.0057 | [-0.0039, 0.0152] | 0.2346 | 0.1309 |
| Hard | -0.0010 | [-0.0135, 0.0118] | 0.8702 | 0.4425 |

No tier significant at 0.05. Gate and raw concat are statistically indistinguishable on seed 43.

Seed-noise calibration, gate_seed43 vs gate_seed44 (identical config, only the seed differs):

| Tier | Delta (seed43 - seed44) | 95% CI | Bootstrap p | McNemar p |
|---|---|---|---|---|
| Easy | -0.0029 | [-0.0093, 0.0035] | 0.3722 | 0.4523 |
| Medium | -0.0031 | [-0.0120, 0.0059] | 0.5098 | 0.6180 |
| Hard | +0.0179 | [0.0066, 0.0293] | 0.0026 | 0.0008 |

Changing only the seed gives a SIGNIFICANT Hard-tier difference (0.018); removing the gate
gives a non-significant one (0.001). Implication: bootstrap/McNemar between two single runs
capture test-sampling noise, not training randomness, so single-seed "significant" gaps of
~0.02 on Hard are not evidence of an architectural effect. Architecture claims need
multi-seed comparisons.

---

## 5. Tier B architectural sweep (validation set, NOT official test — seed 42, single seed)

| Technique | Easy | Medium | Hard | Note |
|---|---|---|---|---|
| XBM cross-batch memory | 0.9168 | 0.7495 | 0.6726 | |
| R-Drop | 0.9104 | **0.7560** | 0.6825 | best Medium result overall |
| Self-adjusting Dice loss | 0.9274 | 0.7382 | 0.6741 | |
| FiLM affine fusion | 0.9173 | 0.7271 | **0.7091** | best single-model Hard result |
| FiLM + R-Drop combined (official test, seed42) | 0.9245 | 0.7245 | 0.7011 | |
| Transformer-seq on BiLSTM | — | — | — | negative result, flatlined, did not train |

---

## 6. Threshold-robustness checks (official test, fixed threshold 0.5)

| Checkpoint | Easy | Medium | Hard |
|---|---|---|---|
| Plain, no contrastive (seed42, 193-dim) | 0.9089 | 0.7215 | 0.6983 |
| Contrastive, seed42 (same as factorial row 1 above, different threshold) | 0.9242 | 0.7155 | 0.7065 |

---

## 7. Dataset statistics (confirmed via real data-loading logs, matches official PAN release)

| Tier | Train | Val | Test |
|---|---|---|---|
| Easy | 4200 | 900 | 900 |
| Medium | 4200 | 900 | 900 |
| Hard | 4200 | 900 | 900 |

---

## 8. Interpretability (gate analysis, validation set, all tiers)

- Mean gate reliance, correct predictions: 0.2246 (n=30,495)
- Mean gate reliance, incorrect predictions: 0.1959 (n=3,159)
- Difference: +0.0287

Top features by |gate delta| (change vs no-change): gunning_fog (+0.4254, trusted MORE
at change), several function-word features trusted LESS at change (fw_themselves,
fw_yet, fw_quite, etc., deltas around -0.28 to -0.31), word_count (-0.17).

---

## 9. New diagnostic findings (this session, the construction-artifact / continuum work)

**(A) Hard-tier inversion, full 6,000-doc official set:**
Change boundaries mean stylometric distance 46.86 vs no-change 74.81, Mann-Whitney
p<0.000001.

**(B) Construction-leakage diagnostic:**
Same-author pairs near a splice point (≤1 away): mean 79.33. Same-author pairs deep in
a stable run (≥3 away): mean 72.19. p<0.000001 — significant difference, some evidence
of construction leakage into nearby text.

**(C) Continuous topic-overlap trend, all 18,000 PAN docs pooled:**
Smooth, monotonic transition from "normal" direction (low overlap) to "inverted" (high
overlap), crossing zero around measured overlap ≈ 0.07–0.08, nearly every bin
significant.

**(D) Within-tier effect-size stratification — the Medium-instability finding:**
- Easy: effect sizes [-110.24, -211.20, -251.39] — consistently normal, large magnitude.
- Hard: effect sizes [18.91, 32.74, 27.02, 18.58] — consistently inverted.
- Medium: effect sizes [+8.85, +2.34, +3.20, -11.16] — near zero, **sign flips**.

**(E) C50/Reuters-50-50 cross-corpus validation (zero splicing, real naturally-written text):**
Same-author mean distance 73.28 vs different-author mean distance 87.43, p<0.000001 —
normal direction, does NOT replicate the inversion at C50's overall average overlap.

**(F) C50 overlap-binned follow-up:**
Across bins covering ~88% of C50's overlap range, consistently significant normal
direction (p<0.0001 every bin). Only the single highest-overlap bin (~12% of data)
nominally flips sign, but NOT significant (p=0.149), and that bin is confounded (same-
author pairs have much higher natural overlap than random cross-author pairs).

---

## 10. Official PAN 2025 leaderboard (from the PAN 2025 overview paper, paper_279.pdf)

Table 1 of the overview paper, all 12 teams, official test F1, in the paper's order:

| Team name | Authors | Easy | Medium | Hard |
|---|---|---|---|---|
| wqd | Lin et al. [16] | 0.958 | 0.823 | **0.830** |
| xxsu-team (the "SCL-DeBERTa" system) | Lin et al. [26] | 0.955 | **0.825** | 0.829 |
| stylospies | Boriceanu and Băltoiu [18] | **0.959** | 0.786 | 0.791 |
| team-tmu | Hosseinbeigi and Mehrani [20] | 0.950 | 0.792 | 0.792 |
| better-call-claude | Schmidt et al. [22] | 0.929 | 0.815 | 0.731 |
| openfact | Księżniak et al. [24] | 0.919 | 0.771 | 0.752 |
| cornell-1 | Bölöni-Turgut et al. [25] | 0.909 | 0.793 | 0.698 |
| batatavada-pict | Rohra et al. [27] | 0.823 | 0.766 | 0.667 |
| hhu | Meier et al. [19] | 0.761 | 0.666 | 0.642 |
| ksu | Alsheddi and Menai [17] | 0.507 | 0.747 | 0.467 |
| hellojie | Chen et al. [21] | 0.461 | 0.583 | 0.484 |
| team-of-bf | Liu et al. [23] | 0.486 | 0.443 | 0.473 |
| Baseline: predict 1 | | 0.178 | 0.177 | 0.147 |
| Baseline: predict 0 | | 0.439 | 0.440 | 0.453 |

Note: "SCL-DeBERTa" is our label; the official team name is xxsu-team. Cite it by
team name or as Lin et al. [26] (not to be confused with wqd, which is also Lin et al. [16]).

Citation corrections needed in the paper:
- better_call_claude: use the official numbers above (0.929/0.815/0.731), NOT their
  self-reported arXiv numbers (0.929/0.828/0.724).
- wqd: describe as DeBERTa + Bayesian hyperparameter optimization, NOT a
  DeBERTa+RoBERTa+BGE-M3 fusion.
- Construction methodology: quote the overview paper directly ("applying different
  similarity thresholds for pairs of consecutive sentences"; relatively high / moderate /
  low for easy / medium / hard). No numeric threshold is published anywhere, so there is
  nothing to compare our measured ~0.07-0.08 crossover against.

---

## Done since the last version

- Isolated gate-vs-concat ablation, seed 43: run, evaluated on official test, significance
  tested (section 4b). Result: no significant difference on any tier.
- Significance testing pipeline confirmed working on real predictions: gate_seed43 F1s
  reproduce the headline numbers exactly.
- Seed-noise calibration (seed43 vs seed44): Hard differs significantly from seed alone.

## Still open / not yet done

- Concat ablation seeds 44 and 45 (running), then a 3-seed gate vs 3-seed concat
  comparison (mean +/- SD), plus per-seed significance tests.
- Check whether the Hard ensemble gain (0.7097 -> 0.7350) holds on seeds 44/45; the
  original number is single-seed and the seed noise on Hard is ~0.018.
- Do NOT present the contrastive Hard gain (+0.003) as an improvement; it is far below
  seed noise.
- Primary-source verification of Koppel et al. (2007), Mikros & Argiri (2007), and
  Wegmann et al. (2022); needs you to upload the PDFs.
- `true_nostyle_nocontrastive` has no saved predictions file yet, so style-vs-no-style
  significance can't be tested until `save_test_predictions.py` is run on it.
