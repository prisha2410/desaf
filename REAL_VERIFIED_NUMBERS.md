# DeSAF: The Actual Verified Numbers (pan2025_full ONLY)

Every number below was produced live this session, via real terminal output pasted
into this conversation, from `train_deberta_fusion.py` / `eval_official_test_2025.py`,
reading from `data/raw/pan2025_full/{tier}/{split}/`. Nothing here comes from the old
2024+2025-combined pipeline, the stale GitHub mirror, or any search-tool paraphrase.
If a number isn't in this document, it is NOT verified for use in the paper. Go back
to the real terminal output or rerun the script before trusting anything else.

> **METRIC CONFIRMED 2026-11-01 (scripts/rescore_official.py, logs\rescore_official.log): the PAN 2025
> leaderboard metric is POOLED two-class macro-F1. On our test labels it reproduces both baseline rows
> exactly (predict 1: 0.178/0.177/0.147; predict 0: 0.439/0.440/0.453); per-document macro does not
> (0.177/0.182/0.157 and 0.435/0.432/0.443). Official-metric results: section 11. The warning below is
> kept for the record; its estimate (0.955/0.825/0.830 for seed43) matched the measured 0.9546/0.8249/0.8301.**
>
> **METRIC WARNING (found 2026-11-01).** Every DeSAF F1 in this file is POSITIVE-CLASS F1
> (F1 of the "change" class): metrics.py uses sklearn's default average='binary', and gate_lab /
> compute_significance use only tp/fp/fn (seed43 Easy P 0.9421, R 0.9155 -> 0.9286 exactly).
> The PAN 2025 leaderboard (section 10) uses TWO-CLASS MACRO-F1, pooled over all boundaries of a tier:
> its "predict 0" row 0.439/0.440/0.453 equals (1-p)/(2-p) and its "predict 1" row 0.178/0.177/0.147
> equals p/(1+p) at change rates ~0.217/0.214/0.172; positive-class F1 of "predict 0" would be 0.
> So EVERY DeSAF-vs-leaderboard comparison made so far (DELCON draft included) compared different
> metrics. ESTIMATE ONLY (from section 1's P/R and those change rates): seed43 = about
> 0.955 / 0.825 / 0.830 in the leaderboard's metric. Not verified until scripts/rescore_official.py
> reproduces the baseline rows from the real test labels and re-scores the saved predictions.
> Until then: call the mean over tiers "tier-average (positive-class) F1", not "macro", and make no
> leaderboard comparison.

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
| ✗ | ✗ | 0.9136 | 0.7183 | 0.7054 | n/a |

(Note: the cell originally mislabeled "factorial_nofusion_nocontrastive" is actually
style-ON/contrastive-OFF: the training command never had `--no-style` in it. The
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
| Gate (headline) | 44 | 0.9315 | 0.7229 | 0.7019 | 0.7854 | 0.52 |
| Raw concat | 44 | 0.9319 | 0.7201 | 0.7125 | 0.7882 | (from log) |
| Delta (concat - gate) | | +0.0004 | -0.0028 | +0.0106 | +0.0028 | |

Seed43 concat validation: overall 0.7948; Easy 0.9311 / Medium 0.7432 / Hard 0.7126.
Predictions saved: `checkpoints/concat_seed43/test_predictions.npz`,
`checkpoints/concat_seed44/test_predictions.npz` (seed44 predictions log present; verify file).
| Gate (headline) | 45 | 0.9181 | 0.7155 | 0.7182 | 0.7839 | 0.64 |
| Raw concat | 45 | 0.9280 | 0.7114 | 0.7236 | 0.7877 | 0.56 |
| Delta (concat - gate) | | +0.0099 | -0.0041 | +0.0054 | +0.0038 | |

Seed45 concat: rerun completed 2026-10-30 (first attempt was cut off by a shutdown); predictions
saved to `checkpoints/concat_seed45/test_predictions.npz`; eval and predictions F1 agree exactly.

**3-seed summary (official test, mean +/- sample SD, ddof=1):**

| Fusion | Easy | Medium | Hard | Macro |
|---|---|---|---|---|
| Gate | 0.9261 +/- 0.0071 | 0.7194 +/- 0.0037 | 0.7133 +/- 0.0099 | 0.7863 +/- 0.0028 |
| Raw concat | 0.9317 +/- 0.0036 | 0.7152 +/- 0.0045 | 0.7190 +/- 0.0058 | 0.7886 +/- 0.0012 |
| Mean paired delta (concat - gate) | +0.0056 | -0.0042 | +0.0057 | +0.0023 |
| Paired t-test p (3 seeds, uncorrected) | 0.181 | 0.040 | 0.174 | |

NOTE: section 2 reports the gate SDs as population SD (ddof=0: 0.0058/0.0031/0.0080). Use the
sample SD above (ddof=1: 0.0071/0.0037/0.0099) in the paper, and say so.
Reading: no tier differs significantly once you correct for testing 3 tiers (Bonferroni 0.040*3 =
0.12). The only consistent direction is a small Medium advantage for the gate (same sign on all 3
seeds); concat is slightly ahead on Easy, Hard and macro. Accuracy does not depend on the gate.

Significance, gate_seed43 vs concat_seed43 (`compute_significance.py`, 10,000 doc-level
bootstrap resamples + McNemar). gate_seed43 F1s reproduced exactly (0.9286/0.7198/0.7198),
confirming `checkpoints/seed43/test_predictions.npz` is valid.

| Tier | Delta (gate - concat) | 95% CI | Bootstrap p | McNemar p |
|---|---|---|---|---|
| Easy | -0.0065 | [-0.0138, 0.0011] | 0.0902 | 0.0750 |
| Medium | +0.0057 | [-0.0039, 0.0152] | 0.2346 | 0.1309 |
| Hard | -0.0010 | [-0.0135, 0.0118] | 0.8702 | 0.4425 |

No tier significant at 0.05. Gate and raw concat are statistically indistinguishable on seed 43.

Significance, gate_seed44 vs concat_seed44 (gate_seed44 F1s reproduced exactly: 0.9315/0.7229/0.7019):

| Tier | Delta (gate - concat) | 95% CI | Bootstrap p | McNemar p |
|---|---|---|---|---|
| Easy | -0.0004 | [-0.0072, 0.0064] | 0.9112 | 0.9462 |
| Medium | +0.0029 | [-0.0064, 0.0120] | 0.5540 | 0.6010 |
| Hard | -0.0105 | [-0.0234, 0.0022] | 0.1132 | 0.0698 |

No tier significant at 0.05 on seed 44 either.

Significance, gate_seed45 vs concat_seed45 (gate_seed45 F1s reproduced exactly: 0.9181/0.7155/0.7182):

| Tier | Delta (gate - concat) | 95% CI | Bootstrap p | McNemar p |
|---|---|---|---|---|
| Easy | -0.0099 | [-0.0194, -0.0008] | 0.0352 | 0.0350 |
| Medium | +0.0041 | [-0.0067, 0.0148] | 0.4588 | 0.6514 |
| Hard | -0.0054 | [-0.0190, 0.0081] | 0.4324 | 0.3803 |

Across all 9 per-seed tier tests, 1 is significant at 0.05 (Easy, seed 45, favouring concat).
With 9 tests about 0.45 false positives are expected by chance, and these tests ignore training
randomness, so this is not evidence of a real gate effect.

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

## 4c. Gate lab round 1 (SCREENING ONLY: validation 5-fold CV, heads on cached seed43 outputs)

Not official test numbers. Heads train on 3/5 of validation, so absolute levels are below the
end-to-end model; use only for comparing variants. Macro F1, mean +/- SD over 3 head seeds:

| Variant | Easy | Medium | Hard | Macro |
|---|---|---|---|---|
| original (checkpoint, optimistic) | 0.9208 | 0.7316 | 0.7104 | 0.7876 |
| cls_only | 0.9010 | 0.7215 | 0.6712 | 0.7645 |
| concat (raw) | 0.9231 | 0.7201 | 0.6827 | 0.7753 |
| concat_ln | 0.9148 | 0.7236 | 0.6685 | 0.7690 |
| gate (DeSAF copy) | 0.9147 | 0.7211 | 0.6692 | 0.7683 |
| scaled_overlap | 0.9197 | 0.7170 | 0.6660 | 0.7676 |
| overlap_gate | 0.9125 | 0.7060 | 0.6535 | 0.7573 |
| concat_overlap_tier | 0.9290 | 0.7272 | 0.7007 | 0.7857 |
| overlap_tier_gate | 0.9377 | 0.7325 | 0.7138 | 0.7946 |

Findings: gate = concat_ln again (replicates the end-to-end null). Overlap-only gate does NOT
help (loses to scaled_overlap on all seeds, -0.010 macro); its gate curve shows no clean flip.
Tier flag gives the biggest jump (concat_overlap_tier vs concat_overlap: +0.030 macro, +0.049
Hard). overlap_tier_gate is best (+0.013 Hard over concat_overlap_tier, same sign on 3 seeds),
but round 1 lacked its strict control and a per-tier-threshold (calibration) check; round 2 adds
concat_tier, scaled_overlap_tier, tier_gate and the threshold check. Round-1 gate display for tier
variants normalised each tier separately (bug, fixed in round 2), so its cross-tier signs are not
interpretable.

**Round 2 (same setup, 2026-10-31/11-01):**

| Variant | Easy | Medium | Hard | Macro |
|---|---|---|---|---|
| concat_tier (raw concat + tier flag) | 0.9436 | 0.7389 | 0.7304 | 0.8043 +/- 0.0028 |
| scaled_overlap_tier | 0.9367 | 0.7372 | 0.7125 | 0.7955 |
| tier_gate | 0.9379 | 0.7286 | 0.7096 | 0.7920 |
| overlap_tier_gate (retrained) | 0.9377 | 0.7325 | 0.7138 | 0.7946 |

- Gate verdict: overlap_tier_gate vs scaled_overlap_tier = -0.0008 macro (mixed signs), vs
  tier_gate = +0.0026 (mixed). Overlap-conditioned gating adds nothing. Hypothesis not supported.
- Tier flag: concat_tier vs concat = +0.0290 macro, +0.0477 Hard, +0.0188 Medium, same sign on all
  3 seeds. concat_tier beats overlap_tier_gate (-0.0096 macro, same sign all seeds) and the
  checkpoint's own (optimistic) validation score 0.7876.
- Threshold check: per-tier thresholds change macro by -0.003 to +0.002 for every variant,
  including the checkpoint. The tier gain is NOT calibration; the flag changes feature use.
  [REVISED 2026-11-09: that check used positive-class F1 and seed43 only; re-tested under the official metric by
  tier_ranking_check.py. Until then do not write "NOT calibration".]
- Gate curves (shared reference = Easy at overlap 0), overlap_tier_gate and tier_gate agree:
  Hard reverses trigram_dissim (about -0.7 to -0.86) and punct_dist (-0.28 to -0.62); Medium
  reverses fw_dist (-0.46 to -0.85); Easy uses all three normally. Curves are flat across overlap
  within each tier: the reversal is tied to the tier, not to measured pair overlap. Consistent
  with within-tier finding (D). Single head (seed 1, fold 0) per variant.
- Pre-registered next step: --final original,concat,concat_tier,overlap_tier_gate; primary claim
  concat_tier vs concat on test. Tier flag is legitimate in PAN (separate per-tier test sets) but
  must be disclosed; it does not apply where the tier is unknown.

**OFFICIAL TEST, pre-registered final run (2026-11-01). Encoder = gated seed43 checkpoint (frozen);
sequence heads trained on validation (85/15 split), 3 head seeds, test touched once:**

| Model | Easy | Medium | Hard | Macro |
|---|---|---|---|---|
| original (seed43 checkpoint, sanity: reproduced exactly) | 0.9286 | 0.7198 | 0.7198 | 0.7894 |
| concat head (no tier) | 0.9364 +/- 0.0041 | 0.7022 +/- 0.0013 | 0.6944 +/- 0.0048 | 0.7777 +/- 0.0021 |
| **concat_tier head** | **0.9538 +/- 0.0012** | **0.7331 +/- 0.0046** | **0.7398 +/- 0.0074** | **0.8089 +/- 0.0036** |
| overlap_tier_gate head | 0.9451 +/- 0.0028 | 0.7261 +/- 0.0046 | 0.7272 +/- 0.0040 | 0.7995 +/- 0.0034 |

(sample SD over 3 head seeds; one encoder only)
- Primary: concat_tier - concat = +0.0174 / +0.0308 / +0.0455, macro +0.0312; every seed, every tier.
- Secondary: concat_tier - original = +0.0252 / +0.0133 / +0.0200, macro +0.0195; every seed, every
  tier above the original. Note this mixes two changes (tier flag + head retrained on validation);
  the head without the flag is BELOW the original, so the flag accounts for more than all of it.
- Context: concat_tier test Hard 0.7398 > old 3-way ensemble 0.7350 and > better-call-claude 0.731
  (official). PAN average-F1 of concat_tier 0.809 sits between openfact (0.814) and cornell-1 (0.800).
- Final-model gate curve (overlap_tier_gate, seed 1): Hard reverses trigram_dissim at every overlap
  (-0.21 to -0.90) and mostly fw_dist; Medium does NOT reverse fw_dist in this model, and Hard punct
  is not reversed here. The ONLY reversal consistent across all three tier-aware gate models
  (CV overlap_tier_gate, CV tier_gate, final overlap_tier_gate) is trigram_dissim on Hard, which is
  also the strongest Easy-vs-Hard contrast in the data (G: Easy +0.62 N, Hard -0.72 R).
- Per-seed significance on test (compute_significance, bootstrap / McNemar; Delta = A - B, so
  negative = tier head better):

| Comparison | Tier | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| concat vs tier | Easy | -0.0119, p<.0001 / <.0001 | -0.0223, <.0001 / <.0001 | -0.0181, <.0001 / <.0001 |
| concat vs tier | Medium | -0.0340, <.0001 / .0176 | -0.0270, <.0001 / .0014 | -0.0316, <.0001 / .0043 |
| concat vs tier | Hard | -0.0550, <.0001 / <.0001 | -0.0408, <.0001 / <.0001 | -0.0407, <.0001 / <.0001 |
| original vs tier | Easy | -0.0241, <.0001 / <.0001 | -0.0265, <.0001 / <.0001 | -0.0250, <.0001 / <.0001 |
| original vs tier | Medium | -0.0179, .0024 / .7896 | -0.0086, .1302 / .0082 | -0.0132, .0126 / .0035 |
| original vs tier | Hard | -0.0270, <.0001 / <.0001 | -0.0123, .0810 / <.0001 | -0.0207, .0018 / <.0001 |

  Tier flag vs no flag: significant on both tests for all 9 seed x tier cells. Tier head vs original
  DeSAF: Easy significant on all seeds; Hard and Medium significant on 2 of 3 seeds by bootstrap.
  These tests cover test-sample noise only, not encoder training randomness (next: seed44/45 encoders).

**ALL THREE ENCODERS (seed43/44/45 gated checkpoints, frozen) x 3 head seeds, official test
(2026-11-01). Caches reproduced each checkpoint's official F1 exactly.**

| Encoder | Head | Easy | Medium | Hard |
|---|---|---|---|---|
| 44 | original | 0.9315 | 0.7229 | 0.7019 |
| 44 | concat s1/s2/s3 | 0.9340/0.9417/0.9413 | 0.7060/0.7194/0.6926 | 0.7029/0.7112/0.6872 |
| 44 | concat_tier s1/s2/s3 | 0.9513/0.9497/0.9549 | 0.7410/0.7368/0.7356 | 0.7281/0.7462/0.7388 |
| 45 | original | 0.9181 | 0.7155 | 0.7182 |
| 45 | concat s1/s2/s3 | 0.9247/0.9222/0.9295 | 0.7026/0.6983/0.7172 | 0.6905/0.7218/0.6979 |
| 45 | concat_tier s1/s2/s3 | 0.9456/0.9499/0.9487 | 0.7382/0.7250/0.7365 | 0.7365/0.7523/0.7466 |

Encoder-level summary (mean over encoders of the per-encoder head mean; SD across the 3 encoders):

| Model | Easy | Medium | Hard | Macro |
|---|---|---|---|---|
| Original DeSAF | 0.9261 +/- 0.0071 | 0.7194 +/- 0.0037 | 0.7133 +/- 0.0099 | 0.7863 +/- 0.0028 |
| Head without tier flag | 0.9336 +/- 0.0072 | 0.7048 +/- 0.0022 | 0.6994 +/- 0.0046 | 0.7793 +/- 0.0022 |
| **Tier-aware head** | **0.9513 +/- 0.0029** | **0.7347 +/- 0.0027** | **0.7409 +/- 0.0038** | **0.8090 +/- 0.0002** |

- Tier-aware head beats the original DeSAF in 27/27 (encoder x head seed x tier) cells, and beats the
  no-flag head in 27/27. Its WORST run on each tier beats the original's BEST seed on that tier
  (Easy 0.9456 > 0.9315, Medium 0.7250 > 0.7229, Hard 0.7281 > 0.7198).
- Paired t-test across the 3 encoders (covers encoder training randomness), tier head vs original:
  Easy p = 0.012, Medium p = 0.007, Hard p = 0.026, macro p = 0.005. Tier vs no-flag head: p = 0.024,
  0.002, 0.003.
- Mean gain over original: Easy +0.025, Medium +0.015, Hard +0.028, macro +0.023.
- Disclose in the paper: (1) the tier flag uses PAN's per-tier test-set split, not available for an
  arbitrary document; (2) the head is retrained on the official validation set with the encoder
  frozen; the test set is used only for this pre-registered evaluation.

**Can the tier be inferred from the text? (scripts/predict_tier.py, 2026-11-01)**
- Text-only document features (24: boundaries, overlap stats, 3 surface distances, style L2), gradient
  boosting chosen by validation CV (0.762; logistic regression 0.724). TEST accuracy 0.762.
  Recall: Easy 0.829, Medium 0.702, Hard 0.756 (Medium mostly confused with Hard: 195/900).
- Validation medians: mean consecutive overlap Easy 0.047, Medium 0.050, Hard 0.090; share > 0.075
  Easy 0.222, Medium 0.222, Hard 0.429; mean style L2 Easy 104.9, Medium 64.1, Hard 68.1.
- Same trained tier-aware heads fed PREDICTED tiers (hard argmax), all 3 encoders x 3 heads, mean:
  Easy 0.9116, Medium 0.6975, Hard 0.6746, macro 0.7613. That is BELOW the no-flag head (0.7793)
  and the original DeSAF (0.7863). A wrong tier is worse than no tier, consistent with the tier
  reversing feature use. (True-tier numbers reproduced exactly in this rerun: deterministic.)
  [REVISED 2026-11-09, see (T): the [CLS]-only head gains as much from the flag as the paper head, so do NOT
  write that a wrong tier hurts because the tier reverses (stylometric) feature use.]
- With encoder features (795 per document: text features + mean [CLS] + predicted-change stats):
  validation CV 0.845-0.853, TEST accuracy 0.861 / 0.863 / 0.853 (encoders 43/44/45). Medium is still
  the most confused (recall ~0.81, mostly predicted as Hard).
- Soft predicted tiers (probabilities) into the true-tier-trained heads, 3 encoders x 3 heads:
  Easy 0.9259, Medium 0.7111, Hard 0.7033, macro 0.7801 (about the no-flag head's 0.7793; below the
  original DeSAF 0.7863 in 9/9 runs).
- Heads trained on out-of-fold predicted validation tiers and tested on predicted tiers:
  Easy 0.9341, Medium 0.7132, Hard 0.6855, macro 0.7776 (above the original in 1/9 runs).
- CONCLUSION: the tier-aware gain needs the true tier, which PAN provides via per-tier test sets.
  Inferring the tier from the text (86% at best) does not recover it; predicted tiers bring the
  model back to about the no-flag level. Consistent with the tier setting the direction of the
  stylometric evidence. Report as a limitation; do not claim the method works without tier labels.
  [REVISED 2026-11-09, see (T): drop "Consistent with the tier setting the direction of the stylometric
  evidence"; the flag's gain does not need the stylometric input. Mechanism wording waits for tier_ranking_check.]

---

## 5. Tier B architectural sweep (validation set, NOT official test; seed 42, single seed)

| Technique | Easy | Medium | Hard | Note |
|---|---|---|---|---|
| XBM cross-batch memory | 0.9168 | 0.7495 | 0.6726 | |
| R-Drop | 0.9104 | **0.7560** | 0.6825 | best Medium result overall |
| Self-adjusting Dice loss | 0.9274 | 0.7382 | 0.6741 | |
| FiLM affine fusion | 0.9173 | 0.7271 | **0.7091** | best single-model Hard result |
| FiLM + R-Drop combined (official test, seed42) | 0.9245 | 0.7245 | 0.7011 | |
| Transformer-seq on BiLSTM | n/a | n/a | n/a | negative result, flatlined, did not train |

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
a stable run (≥3 away): mean 72.19. p<0.000001: significant difference, some evidence
of construction leakage into nearby text.

**(C) Continuous topic-overlap trend, all 18,000 PAN docs pooled:**
Smooth, monotonic transition from "normal" direction (low overlap) to "inverted" (high
overlap), crossing zero around measured overlap ≈ 0.07-0.08, nearly every bin
significant.

**(D) Within-tier effect-size stratification: the Medium-instability finding:**
- Easy: effect sizes [-110.24, -211.20, -251.39]: consistently normal, large magnitude.
- Hard: effect sizes [18.91, 32.74, 27.02, 18.58]: consistently inverted.
- Medium: effect sizes [+8.85, +2.34, +3.20, -11.16]: near zero, **sign flips**.

**(E) C50/Reuters-50-50 cross-corpus validation (zero splicing, real naturally-written text):**
Same-author mean distance 73.28 vs different-author mean distance 87.43, p<0.000001:
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

## 11. OFFICIAL-METRIC RE-SCORE (pooled two-class macro-F1, the PAN 2025 leaderboard metric)

User's run 2026-11-01, `python scripts/rescore_official.py` (logs\rescore_official.log, results\rescore_official.csv).
Re-scores the SAVED official-test predictions; no prediction or threshold was changed, so this is not a new
test evaluation. Aggregated by the main session from the pasted output; the positive-class column of the same
aggregation reproduces sections 2, 4b and 4c exactly (0.7863 +/- 0.0028, 0.7886, 0.8090, 0.7793, 0.7612, 0.7801),
which checks the transcription. "avg" = unweighted mean over the three tiers (PAN ranks per tier; the
average is our summary). SD = sample SD (ddof=1) over runs.

| System (runs) | Easy | Medium | Hard | avg (+/- SD) | positive-class avg |
|---|---|---|---|---|---|
| Original DeSAF, gated (3 encoder seeds) | 0.9530 | 0.8242 | 0.8261 | 0.8678 +/- 0.0019 | 0.7863 |
| Plain concat end-to-end (3 seeds) | 0.9564 | 0.8218 | 0.8288 | 0.8690 +/- 0.0009 | 0.7886 |
| Tier-aware head concat_tier (3 encoders x 3 heads) | **0.9688** | **0.8331** | **0.8440** | **0.8820 +/- 0.0011** | 0.8090 |
| Same head without the tier flag (9) | 0.9576 | 0.8175 | 0.8200 | 0.8650 +/- 0.0027 | 0.7793 |
| overlap_tier_gate (3 heads, encoder 43 only) | 0.9648 | 0.8281 | 0.8344 | 0.8758 +/- 0.0022 | 0.7995 |
| Tier head with predicted tier, hard (9) | 0.9437 | 0.8123 | 0.8037 | 0.8532 +/- 0.0024 | 0.7612 |
| ... soft tier (9) | 0.9526 | 0.8198 | 0.8209 | 0.8644 +/- 0.0021 | 0.7801 |
| ... trained on predicted tiers (9) | 0.9579 | 0.8223 | 0.8121 | 0.8641 +/- 0.0032 | 0.7776 |

Per seed (original gated): seed43 0.9546/0.8249/0.8301; seed44 0.9564/0.8259/0.8190; seed45 0.9479/0.8218/0.8293.
Per seed (concat): 43 0.9587/0.8211/0.8299; 44 0.9566/0.8250/0.8256; 45 0.9540/0.8194/0.8310.
Tier-head per-encoder means: enc43 0.9704/0.8327/0.8435; enc44 0.9693/0.8352/0.8424; enc45 0.9667/0.8314/0.8460.
Worst of the 9 tier-head runs per tier: 0.9652 / 0.8287 / 0.8365; best original seed per tier: 0.9564 / 0.8259 / 0.8301.

Paired t-tests across the 3 encoders (df = 2, two-sided), official metric:
- Tier head vs original: Easy +0.0159 (p 0.012), Medium +0.0089 (p 0.004), Hard +0.0178 (p 0.026), avg +0.0142
  (p 0.004). 27/27 tier-head cells (3 encoders x 3 heads x 3 tiers) beat the original of the same encoder.
- Tier head vs the same head without the flag: Easy +0.0113 (p 0.024), Medium +0.0156 (p 0.005), Hard +0.0240
  (p 0.002), avg +0.0169 (p 0.001).
- Concat vs gated end-to-end: avg +0.0013 (p 0.18); no tier significant. The gate-vs-concat tie holds.
Per-run significance under the official metric (scripts/significance_official.py, user's run 2026-11-01,
logs\significance_official.log, results\significance_official.csv): paired document bootstrap (10,000
resamples, seed 42) of the macro-F1 difference, Holm correction within each family of 9 or 27 tests, McNemar
alongside.
- Gate vs concat (3 seeds x 3 tiers): gate better in 4/9; significant after Holm 0/9 (smallest raw p 0.042,
  seed45 Easy, favouring concat; Holm 0.38). The tie holds under the official metric.
- Tier head vs original (oracle tier; 9 heads x 3 tiers): better in 27/27; significant after Holm 19/27
  (Easy 9/9, Hard 8/9, Medium 2/9). Per-run Medium gains (+0.006 to +0.014) mostly miss significance after
  Holm (Holm p 0.066-0.075); across encoders the Medium gain is significant (paired t p = 0.004, above).
- Tier head vs the same head without the flag: better in 27/27; significant after Holm 26/27 (the exception:
  45h3 Medium +0.0062, p 0.069).
- Predicted tier (soft) vs original: better in only 9/27; significant after Holm 2/27, BOTH favouring the
  original (Hard 45h1 -0.0154, 45h3 -0.0151; Holm p 0.016). A guessed tier gives no gain and sometimes hurts.

Descriptive comparison with the PAN 2025 official table (section 10), SAME metric and same test set:
| System | Easy | Medium | Hard | avg |
|---|---|---|---|---|
| wqd (1st) | 0.958 | 0.823 | 0.830 | 0.870 |
| xxsu-team | 0.955 | 0.825 | 0.829 | 0.870 |
| stylospies | 0.959 | 0.786 | 0.791 | 0.845 |
| better-call-claude | 0.929 | 0.815 | 0.731 | 0.825 |
| openfact | 0.919 | 0.771 | 0.752 | 0.814 |
| DeSAF original (3-seed mean) | 0.953 | 0.824 | 0.826 | 0.868 |
| DeSAF + tier-aware head (9-run mean) | 0.969 | 0.833 | 0.844 | 0.882 |

Readings:
- The original DeSAF is level with the two best PAN 2025 systems (within 0.005 on every tier); it is NOT
  "~6.5 points behind" as the DELCON draft said (that compared positive-class F1 with macro-F1).
- The tier-aware head is above every PAN 2025 system on every tier, and so is its worst single run.
- Against better_call_claude: original DeSAF +0.024 Easy, +0.009 Medium, +0.095 Hard.
CAVEATS that must accompany these in the paper: (1) our test evaluation is post hoc on the released test labels,
not a blind TIRA submission, so this is a descriptive comparison, not a shared-task ranking; (2) the tier head is
a stage-2 head trained on the labelled validation split on frozen encoders, with the tier taken from the
dataset folder (available to every PAN team, since tiers were separate datasets); (3) thresholds were chosen
on validation with positive-class F1, never on test; the tier-head test evaluation was a single pre-registered
run (section 4c).

---

## Done since the last version

- Isolated gate-vs-concat ablation, seed 43: run, evaluated on official test, significance
  tested (section 4b). Result: no significant difference on any tier.
- Significance testing pipeline confirmed working on real predictions: gate_seed43 F1s
  reproduce the headline numbers exactly.
- Seed-noise calibration (seed43 vs seed44): Hard differs significantly from seed alone.

**(G) Feature direction by tier and overlap (scripts/feature_direction_by_tier.py, seed43_gate
cache, validation+test, 68,028 boundaries; first run used Cohen's d + Mann-Whitney):**

| Tier | Subset (n) | fw_dist | trigram_dissim | punct_dist | style_L2 (193) |
|---|---|---|---|---|---|
| Easy | all (20,575) | -0.203 R*** | +0.621 N*** | +0.469 N*** | +1.284 N*** |
| Easy | overlap <= 0.075 (15,661) | -0.291 R*** | +0.656 N*** | +0.334 N*** | +1.237 N*** |
| Easy | overlap > 0.075 (4,914) | +0.387 N** | +0.693 N*** | +1.484 N*** | +2.866 N*** |
| Medium | all (25,725) | +0.253 N*** | +0.086 N | +0.018 N*** | -0.086 R*** |
| Medium | overlap <= 0.075 (19,421) | +0.233 N*** | -0.222 R*** | -0.014 R*** | -0.107 R*** |
| Medium | overlap > 0.075 (6,304) | +0.086 N* | +0.225 N*** | +0.007 N* | +0.065 N |
| Hard | all (21,728) | -0.328 R*** | -0.720 R*** | +0.066 N*** | -0.509 R*** |
| Hard | overlap <= 0.075 (12,745) | -0.473 R*** | -0.704 R*** | +0.539 N | -0.453 R*** |
| Hard | overlap > 0.075 (8,983) | -0.077 R*** | -0.250 R*** | -0.103 R** | -0.462 R*** |

(N = more dissimilar at a change = normal, R = reversed; values are Cohen's d.)
Composition: share of boundaries with overlap > 0.075 is Easy 0.239, Medium 0.245, Hard 0.413;
Hard is 26.6% of low-overlap boundaries but 44.5% of high-overlap boundaries.

Reading:
- Hard is reversed at both overlap levels on fw_dist, trigram and style_L2; Easy is normal at both
  levels on trigram, punct and style_L2. Direction is set mainly by the tier.
- Within a tier, higher overlap does NOT bring inversion (Easy style_L2 d 1.24 -> 2.87; Medium
  style_L2 -0.11 -> +0.07; Hard -0.45 -> -0.46). The pooled "inversion rises with overlap" trend
  (finding C) is therefore a tier-composition effect (Simpson's paradox), not a within-tier effect.
  REVISE finding C in the paper accordingly. Consistent with finding D.
- Medium: features disagree (fw normal, trigram/style reversed at low overlap) and two features flip
  with overlap inside Medium, so aggregate distance cancels: a mechanism for Medium's instability.
- Some cells have mean-based d and rank-based p disagreeing (e.g. Hard punct low overlap d +0.54,
  n.s.), so the script was switched to rank-based AUC for the paper figure.

**(G') AUC version (USE THIS IN THE PAPER). AUC = P(change boundary more dissimilar than a
no-change boundary); > 0.5 normal (N), < 0.5 reversed (R). Validation + test, 68,028 boundaries:**

| Tier | Subset (n) | fw_dist | trigram_dissim | punct_dist | style_L2 (193) |
|---|---|---|---|---|---|
| Easy | all (20,575) | 0.430*** R | 0.772*** N | 0.701*** N | 0.833*** N |
| Easy | overlap <= 0.075 (15,661) | 0.410*** R | 0.687*** N | 0.653*** N | 0.828*** N |
| Easy | overlap > 0.075 (4,914) | 0.588** N | 0.896*** N | 0.814*** N | 0.898*** N |
| Medium | all (25,725) | 0.586*** N | 0.503 N (n.s.) | 0.457*** R | 0.482*** R |
| Medium | overlap <= 0.075 (19,421) | 0.584*** N | 0.428*** R | 0.426*** R | 0.477*** R |
| Medium | overlap > 0.075 (6,304) | 0.523* N | 0.581*** N | 0.522* N | 0.519 N (n.s.) |
| Hard | all (21,728) | 0.417*** R | 0.260*** R | 0.429*** R | 0.339*** R |
| Hard | overlap <= 0.075 (12,745) | 0.410*** R | 0.338*** R | 0.489 R (n.s.) | 0.358*** R |
| Hard | overlap > 0.075 (8,983) | 0.477*** R | 0.394*** R | 0.482** R | 0.365*** R |

Composition: share with overlap > 0.075 is Easy 0.24, Medium 0.25, Hard 0.41; Hard is 27% of
low-overlap and 44% of high-overlap boundaries.
- Hard: all four features reversed at both overlap levels (trigram AUC 0.260 overall). The d-based
  table's "Hard punct normal" was an outlier effect; rank-based AUC shows it reversed.
- Easy: trigram, punct, style normal at both levels; fw reversed at low overlap, normal at high.
- Medium: features disagree and trigram/punct flip with overlap inside the tier; all near 0.5.
- Within Hard, reversal is not stronger at high overlap (trigram 0.338 low vs 0.394 high).
- Agrees with the validation-CV tier-aware gate curves (Hard trigram + punct reversed vs Easy; Medium
  fw opposite to Easy).

---

**(H) C50 construction experiment, MODEL-FREE part (scripts/c50_multiauthor_test.py --build-only).**
PROVENANCE: computed in the analysis sandbox on the user's uploaded C50 copy (4,961 articles with
>= 6 sentences), seed 0; deterministic, so the user's own run should reproduce these exactly; confirm
when that log arrives. 900 documents per construction, ~12 sentences, change rate ~0.18, 2-4 authors,
segments = consecutive sentences of one article. AUC > 0.5 = more dissimilar at an author change.

Splice filter on ALL words (TF-IDF incl. function words):

| Construction | Overlap at change | Within author | fw_dist | trigram | punct |
|---|---|---|---|---|---|
| random | 0.0216 | 0.0983 | 0.537 N | 0.718 N | 0.549 N |
| same_topic | 0.0402 | 0.0988 | 0.531 N | 0.672 N | 0.532 N |
| splice >= 0.05 | 0.0889 | 0.0997 | 0.452 R | 0.515 N | 0.497 |
| splice >= 0.10 | 0.1365 | 0.0993 | 0.447 R | 0.412 R | 0.486 |
| splice >= 0.15 | 0.1822 | 0.1000 | 0.438 R | 0.353 R | 0.495 |
| splice >= 0.20 | 0.2043 | 0.0995 | 0.423 R | 0.318 R | 0.482 R |
| splice >= 0.30 | 0.2230 | 0.0970 | 0.414 R | 0.292 R | 0.472 R |
| most-similar splice | 0.2327 | 0.0979 | 0.406 R | 0.293 R | 0.474 R |

Splice filter on CONTENT words only (stop words removed; fw and punct are held out), FINAL design:

| Construction | Overlap at change | Within author | fw_dist (held out) | trigram | punct (held out) |
|---|---|---|---|---|---|
| random | 0.0217 | 0.0983 | 0.537 N | 0.718 N | 0.549 N |
| same_topic | 0.0403 | 0.0990 | 0.531 N | 0.672 N | 0.532 N |
| splice >= 0.05 | 0.1008 | 0.1003 | 0.509 | 0.499 | 0.508 |
| splice >= 0.10 | 0.1391 | 0.0993 | 0.518 N | 0.441 R | 0.515 |
| splice >= 0.15 | 0.1726 | 0.0988 | 0.507 | 0.391 R | 0.505 |
| splice >= 0.20 | 0.1914 | 0.0995 | 0.499 | 0.365 R | 0.504 |
| splice >= 0.30 | 0.2083 | 0.0972 | 0.497 | 0.340 R | 0.487 |
| most-similar splice | 0.2224 | 0.0987 | 0.481 R | 0.337 R | 0.503 |

(fallback = no candidate above threshold, most similar used: 1.3%, 7.5%, 26.6%, 48.4%, 78.9%.)
Reading:
- Topic matching alone keeps the classical direction; similarity filtering at the splice removes it.
- Dose-response: trigram AUC falls steadily as the filter tightens and crosses 0.5 where overlap at
  author changes equals natural within-author overlap.
- Held-out features (fw, punct under the content filter) do NOT reverse; they fall to chance. Only
  features correlated with the filter's similarity measure reverse. So the reversal is a selection
  effect of how splice points are chosen. PAN 2025 Hard reverses ALL four features, consistent with a
  filter that uses stylistic as well as semantic similarity (the PAN 2026 overview describes such a
  filter for 2026; unverified for 2025).
- At matched overlap (splice >= 0.05, content filter: 0.1008 vs 0.1003) all three sentence-level
  surface cues are at chance (0.499-0.509): in C50, these cues carry little authorship signal once topic
  continuity is matched. Caveats: Reuters house style homogenises writing; only 3 surface features here;
  the 193-feature distance and DeSAF detection come from the user's full run.

**(H, authoritative) User's own run, 2026-11-02, logs\c50_multiauthor_seed43.log (USE THESE in the
paper; they differ from the sandbox numbers above only in the 3rd decimal, from library versions;
every direction matches).** Content-word splice filter, 900 docs per construction:

| Construction | Overlap change / within | fw_dist | trigram | punct | style_L2 (193) | DeSAF ROC-AUC | F1 @0.50 (P / R) | best F1 (thr) |
|---|---|---|---|---|---|---|---|---|
| random | 0.0220 / 0.0977 | 0.540 N | 0.716 N | 0.555 N | 0.552 N | 0.7783 | 0.2203 (0.596 / 0.135) | 0.4556 (0.02) |
| same_topic | 0.0400 / 0.0986 | 0.535 N | 0.675 N | 0.537 N | 0.546 N | 0.7175 | 0.1388 (0.450 / 0.082) | 0.3579 (0.02) |
| splice >= 0.05 | 0.1008 / 0.0995 | 0.500 | 0.498 | 0.512 | 0.530 N | 0.6994 | 0.1109 | 0.3361 (0.02) |
| splice >= 0.10 | 0.1399 / 0.0995 | 0.509 | 0.432 R | 0.514 | 0.522 N | 0.7073 | 0.1381 | 0.3531 (0.02) |
| splice >= 0.15 | 0.1732 / 0.0980 | 0.502 | 0.382 R | 0.502 | 0.512 | 0.7051 | 0.1485 | 0.3455 (0.02) |
| splice >= 0.20 | 0.1908 / 0.0990 | 0.498 | 0.360 R | 0.498 | 0.510 | 0.7158 | 0.1800 | 0.3692 (0.02) |
| splice >= 0.30 | 0.2063 / 0.0997 | 0.499 | 0.344 R | 0.489 | 0.510 | 0.7337 | 0.1838 | 0.3936 (0.02) |
| most-similar splice | 0.2191 / 0.0984 | 0.486 | 0.337 R | 0.492 | 0.513 | 0.7330 | 0.1974 (0.522 / 0.122) | 0.3915 (0.02) |

(all-change F1 baseline ~0.31; checkpoint = gated seed43, threshold 0.50 tuned on PAN; no retraining.)
- PAN-trained DeSAF detects author changes in natural news text above chance (ROC-AUC 0.70-0.78) but
  far below in-domain PAN performance; its PAN threshold does not transfer (recall 0.06-0.14 at 0.50;
  the best threshold is at the bottom of the grid, 0.02, so probabilities collapse out of domain).
  "best F1" picks the threshold on the same documents, so it is optimistic.
- Detection is best when topics shift (random 0.778) and drops when topic is matched (0.718).
  Similarity-filtered splicing does not lower it further (0.70-0.73): the model is not relying on the
  filtered surface cue.
- The 193-feature distance (not filtered) stays weakly normal / near chance (0.51-0.53) under every
  splice filter: held-out cues go to chance, they do not reverse.

**C50, all three DeSAF encoders (logs\c50_multiauthor_seed43/44/45.log; same documents; feature
directions identical across runs since they are model-free). DeSAF ROC-AUC, mean +/- sample SD:**

| Construction | seed43 | seed44 | seed45 | Mean +/- SD | F1 at PAN threshold (mean) |
|---|---|---|---|---|---|
| random | 0.7783 | 0.7592 | 0.7616 | 0.7664 +/- 0.0104 | 0.224 |
| same_topic | 0.7175 | 0.6950 | 0.6894 | 0.7006 +/- 0.0149 | 0.133 |
| splice >= 0.05 | 0.6994 | 0.6820 | 0.6884 | 0.6899 +/- 0.0088 | 0.111 |
| splice >= 0.10 | 0.7073 | 0.6866 | 0.6818 | 0.6919 +/- 0.0136 | 0.125 |
| splice >= 0.15 | 0.7051 | 0.6861 | 0.6765 | 0.6892 +/- 0.0146 | 0.137 |
| splice >= 0.20 | 0.7158 | 0.6903 | 0.6848 | 0.6970 +/- 0.0165 | 0.157 |
| splice >= 0.30 | 0.7337 | 0.7028 | 0.6970 | 0.7112 +/- 0.0197 | 0.163 |
| most-similar splice | 0.7330 | 0.7072 | 0.7109 | 0.7170 +/- 0.0140 | 0.172 |

- Topic shift helps on every encoder: random - same_topic = +0.061 / +0.064 / +0.072 (mean +0.066).
- Similarity filtering never pushes DeSAF below the same-topic level; the strictest filters are slightly
  ABOVE it on every encoder (most-similar - same_topic = +0.016 / +0.012 / +0.022). The PAN-trained
  model does not depend on the cue the filter manipulates. Do not over-interpret this small rise.
- Best-F1 threshold is 0.02 (grid minimum) for every encoder and construction: calibration collapse
  out of domain is consistent across seeds.

**(G'') Feature direction on raw text, PAN 2025 vs PAN 2026 (scripts/pan_feature_direction.py, user's
runs 2026-10-31, logs\feature_direction_pan2025.log and logs\feature_direction_pan2026.log). VALIDATION split
only. AUC = P(change boundary more dissimilar than a no-change one); N > 0.5, R < 0.5. "all" rows have
document-bootstrap 95% CIs (200 resamples); subset rows carry only boundary-level Mann-Whitney stars (they
ignore clustering within documents, so they are anti-conservative). Overlap split = median of the overlap
scores POOLED over all three tiers of that edition (2025: 0.0270; 2026: 0.1160), not per tier.
Overlap = all-word TF-IDF cosine (max_features 20000, fitted on that edition's train sentences; function
words included): absolute overlap levels are NOT comparable across editions, only within-edition ratios.**

PAN 2025 (Reddit), validation, 33,654 boundaries:

| Tier (change rate; overlap change / within) | Subset (n) | fw_dist | trigram | punct | style_L2 |
|---|---|---|---|---|---|
| Easy (0.215; 0.0088 / 0.0745) | all (10,247) | 0.436 R [0.419, 0.451] | 0.767 N [0.758, 0.778] | 0.707 N [0.695, 0.718] | 0.832 N [0.821, 0.842] |
| | overlap <= 0.027 (6,231) | 0.403*** R | 0.636*** N | 0.627*** N | 0.817*** N |
| | overlap > 0.027 (4,016) | 0.659*** N | 0.884*** N | 0.774*** N | 0.904*** N |
| | both >= 4 words (7,814) | 0.506 (n.s.) | 0.784*** N | 0.652*** N | 0.835*** N |
| Medium (0.216; 0.0387 / 0.0616) | all (12,759) | 0.596 N [0.583, 0.608] | 0.506 n.s. [0.495, 0.519] | 0.456 R [0.442, 0.470] | 0.480 R [0.468, 0.491] (**) |
| | overlap <= 0.027 (6,453) | 0.605*** N | 0.428*** R | 0.439*** R | 0.418*** R |
| | overlap > 0.027 (6,306) | 0.562*** N | 0.557*** N | 0.457*** R | 0.540*** N |
| | both >= 4 words (11,805) | 0.604*** N | 0.504 (n.s.) | 0.449*** R | 0.489 (n.s.) |
| Hard (0.179; 0.1574 / 0.0813) | all (10,648) | 0.411 R [0.399, 0.425] | 0.259 R [0.247, 0.270] | 0.419 R [0.406, 0.428] | 0.348 R [0.335, 0.360] |
| | overlap <= 0.027 (4,143) | 0.396*** R | 0.500 (n.s.) | 0.592*** N | 0.407*** R |
| | overlap > 0.027 (6,505) | 0.441*** R | 0.316*** R | 0.458*** R | 0.361*** R |
| | both >= 4 words (9,426) | 0.398*** R | 0.230*** R | 0.422*** R | 0.348*** R |

Share above the pooled median: Easy 0.39, Medium 0.49, Hard 0.61.

PAN 2026 (AO3 fan fiction), validation, 523,803 boundaries (2,250 documents per tier):

| Tier (change rate; overlap change / within) | Subset (n) | fw_dist | trigram | punct | style_L2 |
|---|---|---|---|---|---|
| Easy (0.303; 0.0294 / 0.2148) | all (117,435) | 0.764 N [0.758, 0.770] | 0.953 N [0.951, 0.954] | 0.676 N [0.669, 0.682] | 0.933 N [0.930, 0.937] |
| | overlap <= 0.116 (48,255) | 0.670*** N | 0.861*** N | 0.650*** N | 0.941*** N |
| | overlap > 0.116 (69,180) | 0.679*** N | 0.871*** N | 0.675*** N | 0.973*** N |
| Medium (0.044; 0.0554 / 0.0974) | all (278,988) | 0.588 N [0.582, 0.594] | 0.652 N [0.646, 0.656] | 0.565 N [0.558, 0.571] | 0.486 R [0.480, 0.491] |
| | overlap <= 0.116 (189,096) | 0.560*** N | 0.589*** N | 0.556*** N | 0.493* R |
| | overlap > 0.116 (89,892) | 0.578*** N | 0.659*** N | 0.592*** N | 0.520** N |
| Hard (0.212; 0.1747 / 0.2100) | all (127,380) | 0.576 N [0.571, 0.582] | 0.642 N [0.637, 0.648] | 0.581 N [0.575, 0.587] | 0.476 R [0.472, 0.480] |
| | overlap <= 0.116 (24,551) | 0.548*** N | 0.577*** N | 0.561*** N | 0.478*** R |
| | overlap > 0.116 (102,829) | 0.571*** N | 0.642*** N | 0.585*** N | 0.476*** R |

"both >= 4 words" removes 2 of 117,435 Easy pairs and none in Medium/Hard: PAN 2026 has essentially no
1-3-word lines, so that check is uninformative there. Share above the pooled median: Easy 0.59, Medium 0.32,
Hard 0.81. 2026 Medium: ~124 boundaries and ~5.5 changes per document (Easy ~52 / ~15.8, Hard ~57 / ~12.0).

Verified readings (each checked by an adversarial verifier against the logs and code, 2026-10-31):
- PAN 2025 consistency: validation "all" AUCs agree with (G') (val+test) to within 0.010 on rounded values,
  same N/R letter on all 12 cells. This is a PIPELINE CONSISTENCY CHECK, not a replication: validation is
  33,654 of the 68,028 G' boundaries (49.5%) and the new script reuses the old feature code verbatim. A real
  out-of-sample check = rerun with --splits test.
- PAN 2026 Hard does NOT reverse the three surface cues (fw 0.576, trigram 0.642, punct 0.581; CIs above 0.5;
  normal in both overlap halves). Only style_L2 is reversed, weakly (0.476), and also in 2026 Medium (0.486).
  PAN 2025 Hard reverses all four (CIs below 0.5). So "Hard reverses everything" is a PAN 2025 property.
  This does NOT by itself show a corpus effect or a construction effect: the editions differ in corpus
  (Reddit vs fan fiction), segmentation, document length, and in how the filter behaved (all-word overlap
  ratio at changes 1.94 in 2025 Hard vs 0.83 in 2026 Hard). Present 2025 vs 2026 as a descriptive contrast.
- Overlap overshoot: only 2025 Hard has higher mean overlap at changes than within an author (ratio 1.94);
  ratios elsewhere: 2025 Easy 0.12, Medium 0.63; 2026 Easy 0.14, Medium 0.57, Hard 0.83 (means, no CIs).
  The sign rule "ratio > 1 <=> trigram reversed" holds for all six PAN tiers and the C50 sweep, but the PAN
  tiers do NOT lie on the C50 curve (2025 Hard is more reversed than any C50 condition; 2025 Medium reaches
  chance at ratio 0.63; 2026 Hard is more normal than C50 predicts), the rule is for trigram only, and it is
  partly mechanical (trigram dissimilarity and word overlap measure the same words). Descriptive only.
- 2025 Hard at near-zero overlap (<= 0.027, 39% of Hard): trigram at chance (0.500), punct NORMAL (0.592),
  fw (0.396) and style_L2 (0.407) still reversed. For trigram, punct and style_L2 the overall Hard AUC is more
  reversed than either half (e.g. trigram 0.259 vs 0.500 / 0.316): much of the reversal lies BETWEEN overlap
  levels, as expected if it mainly reflects the higher overlap at Hard changes.
- 2025 Easy fw reversal (0.436) depends on the pairs with a sentence < 4 words (0.506 n.s. without them).
  Whether this is a measurement artifact (coarse function-word rates in 1-3-word lines), composition, or the
  same pairs as the low-overlap reversal cannot be told from this log.
- 2025 Medium: cues disagree (fw N 0.596; trigram chance; punct R 0.456; style_L2 R 0.480, n.s. without short
  lines). Which cues flip with overlap depends on the split point: only the trigram flip appears at both the
  0.075 split (G') and the median split.

CORRECTIONS to earlier readings (do not use the old wording):
- (G') "Hard: all four features reversed at both overlap levels": at the 0.075 split Hard punct low-overlap
  was already n.s. (0.489); at the median split trigram is at chance and punct normal in the lowest-overlap
  39%. Say: "all four reversed overall; trigram and punct reversal is concentrated in boundaries with shared
  vocabulary; fw and style_L2 stay reversed at near-zero overlap".
- (G) / (G') Simpson's paradox ("within a tier, higher overlap does NOT bring inversion"): soften. With the
  median split, 2025 Hard is MORE reversed at higher overlap on trigram (0.500 -> 0.316) and style_L2
  (0.407 -> 0.361), punct flips N -> R; with the 0.075 split it was the opposite. Tier composition contributes
  (61% of Hard boundaries lie above the pooled median, vs 39% of Easy and 49% of Medium), but whether overlap changes
  direction WITHIN a tier is open until the finer, overlap-matched analysis (pan_direction_followup.py part A).
- style_L2 (193) is NOT a 193-feature style distance: it is the unscaled L2 of the raw |difference| vector,
  so char_count, word_count, avg_sentence_length, Flesch, Fog and Coleman-Liau dominate; the 176 [0,1]
  features can add at most sqrt(11) = 3.3 to a pair's distance while mean distances are ~47-105. On C50 text
  (verifier's illustration, no POS) char_count alone gave ~82% of the squared distance and Flesch ~12%.
  Read every style_L2 column ((G), (G'), (G''), C50 tables, section 9(A)-(F), 9(A)'s "46.86 vs 74.81" from
  analysis_construction_artifact.py, and the predict_tier style-L2 feature) as "sentence-length (and syllable)
  difference". Rename it in the paper (e.g. "length-dominated feature distance") unless the z-scored group
  analysis (follow-up part C) shows the same direction.
- C12-style framing ("direction is a property of how each edition was built"): the PAN comparison alone
  cannot attribute the 2025/2026 difference to construction rather than corpus. Defensible version: "cue
  direction is not a fixed property of hard multi-author text; in C50, holding the corpus fixed, a similarity
  filter at the splice reverses the cue linked to the filter while held-out cues fall to chance".
- Both overviews describe tiers built from topical + stylistic similarity of consecutive sentences, but
  neither publishes the measure or thresholds and their threshold wording is opposite (2025 "relatively high
  for easy ... low for difficult"; 2026 "relatively low for Easy ... high for Hard"). The earlier note "same
  meaning" is our interpretation, not in either source.

Reviewer objections still open (from the critic pass): within-document AUC and doc-bootstrap CIs on subset
cells; content-word (stop-word-free) overlap for matching; length-matched cue AUCs; dialogue/quotation
structure in fan fiction; exact p-values with Holm correction and a minimum |AUC - 0.5| (e.g. 0.05) before
calling a direction; out-of-sample splits (PAN 2025 test, PAN 2026 train); whether the tier-aware head's gain
comes from cue direction at all (flag vs no-flag with the stylometric input removed). The first four are in
pan_direction_followup.py v2.

**(G''') Follow-up, v1 of scripts/pan_direction_followup.py (user's runs 2026-11-01, logs\followup_pan2025.log
and logs\followup_pan2026.log; validation only). v1 matched on the ALL-WORD overlap only (function words
included, so partly mechanical for fw_dist and trigram), part C z-scored WITHOUT clipping or rare-feature
filtering and WITHOUT length matching. Treat part C as provisional until v2 (clipped, length-matched) runs.**

A. AUC at matched all-word overlap (zero bin + 10 within-tier quantile bins; 95% doc-bootstrap CI; 5 / 20 bins):

| | fw_dist | trigram | punct | style_L2 (unscaled) |
|---|---|---|---|---|
| 2025 Easy | 0.397 R [0.383, 0.410] (0.404 / 0.394) | 0.608 N [0.596, 0.620] | 0.610 N [0.597, 0.623] | 0.811 N [0.798, 0.822] |
| 2025 Medium | 0.575 N [0.563, 0.590] | 0.464 R [0.447, 0.479] | 0.468 R [0.452, 0.483] | 0.457 R [0.440, 0.472] |
| 2025 Hard | 0.429 R [0.410, 0.447] (0.447 / 0.408) | 0.443 R [0.424, 0.466] (0.411 / 0.488) | 0.542 N [0.521, 0.560] (0.510 / 0.582) | 0.398 R [0.378, 0.414] (0.385 / 0.415) |
| 2026 Easy | 0.601 N [0.590, 0.611] | 0.738 N [0.727, 0.747] | 0.641 N [0.631, 0.652] | 0.966 N [0.963, 0.969] |
| 2026 Medium | 0.541 N [0.536, 0.548] | 0.555 N [0.550, 0.562] | 0.556 N [0.550, 0.562] | 0.507 N [0.501, 0.512] |
| 2026 Hard | 0.552 N [0.547, 0.558] | 0.604 N [0.598, 0.610] | 0.577 N [0.571, 0.584] | 0.476 R [0.472, 0.480] |

Overlap AUC (P(change has LESS overlap)): 2025 Easy 0.711, Medium 0.547, Hard 0.291 [0.280, 0.302];
2026 Easy 0.984, Medium 0.670, Hard 0.603 [0.596, 0.610]. Zero-overlap boundaries: 2025 Easy 46.8% (72.9% of
changes), Medium 30.5%, Hard 25.0% (9.9% of changes, 28.3% of no-changes); 2026 1.8% / 1.8% / 0.1%.
2025 Hard per bin: zero-overlap bin (n 2,660, 189 changes) fw 0.341 R, trigram 0.628 N, punct 0.707 N, style
0.469 R; positive-overlap bins 1-8 trigram 0.190-0.382 R, bin 9 0.430, top bin 0.555 N. The matched trigram
0.443 is a weighted mix of these (recomputed by hand from the bin table: 0.4426), and it moves with the number
of bins (0.411 / 0.443 / 0.488), so residual within-bin confounding remains. 2025 Medium bins 2 and 4 show
extreme AUCs (bin 2: fw 0.860, trigram 0.224, punct 0.205, style 0.212; bin 4: punct 0.292, style 0.801):
suspected clusters of repeated boilerplate sentences (bcc: >10% exact duplicates in PAN 2025); v2 tests this.

B. Length cues (AUC of the length difference; > 0.5 = changes join sentences of MORE different length):

| | abs d words | abs d chars | abs log word ratio | style_L2 vs abs d chars (Spearman) | style_L2 at matched abs d chars (deciles) |
|---|---|---|---|---|---|
| 2025 Easy | 0.731 N | 0.741 N | 0.785 N | 0.879 | 0.803 N |
| 2025 Medium | 0.508 | 0.500 | 0.478 R | 0.913 | 0.461 R |
| 2025 Hard | 0.364 R [0.352, 0.374] | 0.352 R [0.338, 0.362] | 0.287 R [0.277, 0.296] | 0.906 | 0.390 R |
| 2026 Easy | 0.838 N | 0.869 N | 0.857 N | 0.936 | 0.836 N |
| 2026 Medium | 0.483 R [0.478, 0.489] | 0.479 R | 0.488 R | 0.926 | 0.517 N |
| 2026 Hard | 0.472 R [0.467, 0.476] | 0.466 R | 0.486 R | 0.919 | 0.515 N |

C. z-scored 193 features, one distance per group (v1: no clipping, no length matching; 2025 all 900 docs per
tier, 2026 400 docs per tier):

| | raw L2 | all_z | length | readability | punctuation | function words | POS | other surface | all_z without length+readability |
|---|---|---|---|---|---|---|---|---|---|
| 2025 Easy | 0.832 N | 0.622 N | 0.741 N | 0.889 N | 0.689 N | 0.454 R | 0.745 N | 0.788 N | 0.553 N |
| 2025 Medium | 0.480 R | 0.518 N | 0.509 | 0.437 R | 0.483 R | 0.532 N | 0.469 R | 0.479 R | 0.521 N |
| 2025 Hard | 0.348 R | 0.352 R | 0.364 R | 0.315 R | 0.423 R | 0.385 R | 0.306 R | 0.403 R | 0.361 R |
| 2026 Easy | 0.933 N | 0.763 N | 0.865 N | 0.928 N | 0.613 N | 0.633 N | 0.849 N | 0.782 N | 0.682 N |
| 2026 Medium | 0.489 R | 0.570 N | 0.495 (CI incl. 0.5) | 0.521 N | 0.525 N | 0.561 N | 0.573 N | 0.544 N | 0.568 N |
| 2026 Hard | 0.479 R | 0.562 N [0.551, 0.576] | 0.493 [0.482, 0.505] | 0.527 N | 0.550 N | 0.557 N | 0.553 N | 0.526 N | 0.562 N |

(All 2025 Hard group CIs are below 0.5; all 2026 Hard group CIs except length are above 0.5.)

Readings (by the main session from the logged numbers; NOT adversarially verified. For PAN 2025 they are
superseded by the verified wording in (G'''') below; the PAN 2026 readings still await the v2 2026 run):
- The 2026 style_L2 "reversal" (0.476 Hard, 0.486 Medium) is a scaling artifact: style_L2 tracks |d chars|
  (Spearman 0.92-0.93), turns normal at matched length (0.515 / 0.517), and the properly z-scored 193 features
  are NORMAL in 2026 Medium (0.570) and Hard (0.562), with every family except length above 0.5.
- PAN 2026 Hard is therefore "weakly normal on every stylometric family; sentence length slightly matched"
  (length AUCs 0.466-0.486).
- PAN 2025 Hard is "reversed on every stylometric family" (0.306-0.423 unmatched), and the most strongly
  reversed single cue is the sentence-length ratio (0.287): at 2025 Hard author changes the two sentences are
  more alike in length than two neighbouring sentences by the same author. Whether the other families are
  reversed beyond what length explains is the open question v2 part C (length-matched) answers.
- Matching on lexical overlap REDUCES the 2025 Hard reversal but does not remove it: fw 0.429 and style_L2
  0.398 stay reversed under 5/10/20 bins; punct turns normal (0.542); trigram is unstable (0.411-0.488).
  Within the narrow positive-overlap bins trigram stays strongly reversed (0.19-0.38), plausibly via the
  length matching (similar-length sentences share more character trigrams); untested.
- PAN 2025 Easy's fw reversal survives overlap matching (0.397) and sits in the zero-overlap bin (0.390;
  that bin is 47% of Easy and 73% of its changes, mostly short lines).

**(G'''') v2 follow-up on PAN 2025 validation (scripts/pan_direction_followup.py v2, user's run 2026-11-01,
logs\followup2_pan2025.log). Readings below were checked by adversarial verifiers against the log and the
code; use THESE wordings, not the main session's draft (D1-D8, logs_user/draft_claims_v2.md).**

Key numbers (AUC; N > 0.5, R < 0.5; CIs are document-bootstrap where given):
- Hard: content-word overlap at changes vs within author 0.1552 / 0.0649 (ratio 2.39; overlap AUC 0.277);
  zero content overlap for 22.4% of changes vs 66.2% of no-changes. Length cues: |d words| 0.364, |d chars|
  0.352, |log length ratio| 0.287 [0.277, 0.297].
- Hard, four cues (fw / trigram / punct / style_L2): matched content overlap 0.416 / 0.418 [0.389, 0.453] /
  0.519 [0.493, 0.543] / 0.371 [0.345, 0.397]; within documents 0.423 / 0.294 / 0.446 [0.428, 0.463] / 0.344;
  without repeated lines 0.411 / 0.221 / 0.396 / 0.333; without repeated lines + content-matched 0.468 /
  0.274 / 0.398 / 0.285; both sentences >= 8 words 0.414 / 0.263 / 0.467 / 0.378; matched length 0.476 /
  0.354 [0.339, 0.371] / 0.544 [0.527, 0.562] / 0.421.
- Hard, z-scored families (clipped |z| <= 5, rare features dropped; 137 of 193 kept), unmatched -> matched
  length: all_z 0.327 -> 0.465 [0.449, 0.478]; length 0.363 -> 0.466 [0.452, 0.485]; readability 0.315 ->
  0.400 [0.386, 0.418]; punctuation 0.482 -> 0.514; function words 0.408 -> 0.480; POS 0.305 -> 0.462;
  other surface 0.403 -> 0.472; all_z without length+readability 0.353 -> 0.475; all_z without function
  words 0.298 -> 0.439 [0.422, 0.454].
- Repeated lines (exact lines occurring >= 10 times across the 2,700 validation documents, all tiers pooled):
  touch 41.2% of Easy changes vs 13.3% of no-changes; Medium 41.7% vs 21.3%; Hard 6.3% vs 3.6%. As a yes/no
  cue: AUC about 0.64 (Easy), 0.60 (Medium), 0.51 (Hard); fewer than half of flagged boundaries are changes
  (46% Easy, 35% Medium). Content of the flagged lines NOT inspected (do not call them moderation messages).
- Medium without repeated lines: 0.524 / 0.530 / 0.520 / 0.510 (point estimates, no CIs; change rate ~0.17);
  + content-matched 0.507 / 0.475 / 0.498 / 0.494.
- Easy fw_dist: 0.436 overall; 0.413 [0.400, 0.427] matched content; 0.422 [0.402, 0.440] within documents;
  0.422 [0.406, 0.438] matched length; 0.591 with both sentences >= 8 words (n 4,775); 0.486 without repeated
  lines (n 8,272). z-scored function-word family in Easy: 0.390 unmatched, 0.428 at matched length.

Verified wording for the paper:
- PAN 2025 Hard: "Adjacent sentences across an author change tend to be more alike than adjacent sentences
  by the same author: closer in length (|log length ratio| AUC 0.287), sharing more content words (0.277) and
  more character trigrams (0.259; 0.294 within documents); every z-scored feature family points the same way
  (0.305-0.408; punctuation 0.482 near chance). This fits the overview's statement that Hard documents were
  built from stylistically similar consecutive sentences. Matching on sentence length moves the scaled
  all-feature distance from 0.327 to 0.465 (about 80% of its distance from 0.5), so length accounts for most of
  the feature-level reversal. The residual reversal of readability (0.400) and character trigrams (0.354)
  cannot yet be separated from leftover length (the length family itself is still 0.466 after matching) and
  topic overlap; three of the five readability scores contain sentence length."
- Do NOT write "word complexity" for readability; do NOT write "not a topic-overlap artifact": content matching
  removes about two-thirds of the trigram reversal (0.259 -> 0.418), the matched-content AUC is ~85% the
  zero-content-overlap bin, and in the provisional zero ALL-word-overlap bin (v1, 189 changes) trigram is
  normal (0.628).
- Hard punctuation is not an "exception": at matched content overlap it is at chance only while repeated lines
  are included (0.519); without them it is reversed (0.396 / 0.398), and it is reversed within documents.
- Hard fw_dist: reversed under each single control, but near chance when repeated-line removal and content
  matching are combined (0.468) and among pairs that share content words (pooled bins 1-10: 0.477).
- PAN 2025 Easy and Medium: "the direction of single raw cues is fragile: it changes when pairs containing a
  sentence under 8 words, or boundaries next to a line that occurs 10+ times in the validation set, are
  excluded; such repeated lines touch 41-42% of changes but only 13-21% of no-changes." Do not say the
  reversals "come from" short lines or boilerplate (the two filters were never combined; no CIs on subsets;
  the z-scored Easy function-word family 0.390 and Medium readability 0.437 were not re-tested).
- Medium: removing repeated lines also removes Medium's strongest NORMAL cue (fw 0.596 -> 0.524): both the
  disagreement between cues and most of the cue signal sit at boundaries next to repeated lines. The scaled
  all-feature normal signal in Medium (0.540 / 0.577 at matched length) comes from the function-word family
  (108 of 137 kept features); without function words it is 0.443 (reversed) / 0.474.
- Within documents vs content-matched in Medium condition on different variables; no AUC is both, so nothing
  can be said about cross-document mixing from this log.
- "A detector can exploit repeated lines by memorising them" is UNTESTED; say only that the regularity exists.

Open checks (cheap; saved npz + text, no spaCy unless stated): exact length cells and joint length x overlap
(and within-document x length) matching; bin-0 vs bins 1-10 separately with CIs; length-free readability
features (syllables per word, complex-word ratio) vs length-containing ones; what the repeated lines are
(most_common), dup x short x label cross-table, DUP_MIN 3 / 30; DeSAF test macro-F1 on boundaries next to vs
not next to repeated lines; the same analysis on the PAN 2025 test split; CIs on every cited subset row; Holm.

**(G''''') v2 follow-up on PAN 2026 validation (user's run 2026-11-01, logs\followup2_pan2026.log). Verified
wording (adversarial verifiers vs the log and code). Part C used 400 of 2,250 documents per tier (~18%).**
- PAN 2026 Hard, raw surface cues (fw / trigram / punct) point the normal way under each single control:
  matched content overlap 0.570 [0.564, 0.575] / 0.623 [0.617, 0.628] / 0.582 [0.576, 0.589]; within documents
  0.574 / 0.640 / 0.582; matched length 0.573 / 0.633 / 0.578 (CIs exclude 0.5). Point estimates stay above 0.5 in
  all 11 content-overlap bins and all quotation groups (no CIs); in the zero-content-overlap bin fw is near chance
  (0.533). Modest effects (0.57-0.64). The 8-word and repeated-line filters remove nothing in 2026.
- PAN 2026 style_L2 "reversal" (Hard 0.476, Medium 0.486; both within 0.05 of chance) follows the character-length
  difference (Spearman 0.919 / 0.926; |d chars| 0.466 / 0.479); at matched length 0.500 / 0.502. Z-scored families
  at matched length, Hard: length 0.516, readability 0.519, punctuation 0.537, function words 0.554, POS 0.539,
  other 0.519, all_z 0.562 [0.549, 0.575]: no family reversed; five of six within 0.05 of chance.
- PAN 2026 Medium: surface cues normal under each control (content-matched 0.561 / 0.585 / 0.559; within documents
  0.582 / 0.646 / 0.562; matched length 0.587 / 0.652 / 0.564). Easy: all strongly normal; content overlap alone
  nearly separates changes (overlap AUC 0.973).
- In the released PAN 2026 validation files there are no lines repeated >= 10 times and essentially no lines
  under 8 words (11 of 117,435 Easy pairs; none in Medium/Hard); median shorter line 21-22 words vs 7/10/12 in
  PAN 2025. Whether the organisers filtered them is not stated.
- 2025 vs 2026 Hard (descriptive; content overlap uses a separate TF-IDF per edition, so compare within-edition
  AUCs/ratios only): 2025 Hard changes share MORE content words than same-author neighbours (overlap AUC 0.277,
  ratio 2.39), are closer in length (0.287) and share more trigrams (0.259), scaled all-feature 0.327 (reversed);
  2026 Hard changes share slightly FEWER content words (0.579, ratio 0.84) and fewer trigrams (0.642), scaled
  all-feature 0.576 (weakly normal); length only marginally matched (0.466-0.486). Say "reversed vs not reversed",
  not "opposite". At matched length the all_z gap between editions shrinks from 0.249 to 0.097 (61% is length).
- Verified paper sentence: "At PAN 2025 Hard author changes, adjacent sentences tend to be more alike than
  adjacent same-author sentences (content-word overlap AUC 0.277, |log length ratio| 0.287, scaled all-feature
  distance 0.327), mostly through sentence length (0.465 once length is matched). At PAN 2026 Hard the same
  measures point weakly the normal way. The sign of classical stylometric evidence at 'Hard' author changes is
  thus not fixed across PAN editions; the editions differ in corpus, segmentation and sentence length as well as
  in how their similarity filter behaved (content overlap at changes 2.39x vs 0.84x the within-author level), so
  this contrast alone cannot say why." The causal part must come from the construction experiment (I).

**(I) Construction experiment (scripts/construction_experiment.py), SANDBOX PREVIEW on the user's uploaded C50
copy (4,961 articles, 109,290 sentences), 900 documents per construction, seed 0, 200 doc-bootstrap resamples;
style vector = 112 kept z-scored non-POS features. PROVISIONAL until the user's own run reproduces it (it is
deterministic). AUC; content overlap AUC = P(change shares LESS content); others = P(change more dissimilar).**

| Construction | content overlap | abs log length ratio | fw | trigram | punct | all_z | length | readability | function words |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.771 N | 0.541 N | 0.538 N | 0.718 N | 0.549 N | 0.556 N | 0.559 N | 0.557 N | 0.530 N |
| same_topic | 0.697 N | 0.530 N | 0.536 N | 0.673 N | 0.537 N | 0.539 N | 0.547 N | 0.541 N | 0.519 N |
| content_splice | 0.165 R | 0.527 N | 0.493 | 0.345 R | 0.499 | 0.497 | 0.518 | 0.492 | 0.491 |
| length_splice | 0.758 N | 0.020 R | 0.502 | 0.670 N | 0.481 | 0.508 | 0.193 R | 0.488 | 0.526 N |
| style_splice | 0.737 N | 0.379 R | 0.208 R | 0.495 | 0.456 R | 0.130 R | 0.352 R | 0.356 R | 0.154 R |
| style_nolen | 0.738 N | 0.444 R | 0.220 R | 0.496 | 0.472 R | 0.152 R | 0.455 R | 0.475 R | 0.149 R |
| topic_style | 0.315 R | 0.403 R | 0.293 R | 0.301 R | 0.434 R | 0.254 R | 0.384 R | 0.366 R | 0.280 R |

(content overlap change/within ratio: random 0.06, same_topic 0.29, content_splice 2.61, topic_style 1.65.)
Reading (provisional): with the text held fixed, random assembly gives the classical direction on every cue;
each filter reverses the cue family it selects on while unselected families stay near chance (content filter ->
content + trigram only; length filter -> length only); a combined topic + style filter reverses every family,
reproducing the PAN 2025 Hard pattern (content 0.315 vs 0.277, length 0.403 vs 0.287, trigram 0.301 vs 0.259,
all_z 0.254 vs 0.327). Style vector without length still leaves length slightly reversed (0.444/0.455), because
style features correlate with length. The same experiment on PAN 2025 Reddit segments (same text as PAN) is the
user's run: `--source pan --data-root data/raw/pan2025_full --name reddit2025`.

**(J) User's runs 2026-11-02 (logs: c50_rescore_official, construction_reddit2025, construction_c50, checks_pan2025).
Readings below use the adversarial verifiers' corrected wording (logs_user/F_verdicts.txt). The user's C50 construction
run SUPERSEDES the sandbox preview (I): it agrees within 0.016 AUC in all 91 cells (mean abs diff 0.005), no direction
changes, but boundary counts differ, so the built documents were not identical. Cite the user's run.**

Construction experiment, AUC (content overlap AUC = P(change shares LESS content); others = P(change more dissimilar)).
Reddit = 27,230 single-author segments (>= 2 sentences) cut at PAN's change points from the 12,600 PAN 2025 TRAIN
documents of all tiers, after dropping 9,953 segments (26.8%) containing a line repeated >= 10 times; "author" =
source document. C50 = 4,961 articles, real author ids. Guard: 0.00% natural same-author pairs at changes everywhere.

| Construction (Reddit / C50) | content ovl | abs log len ratio | fw | trigram | punct | all_z | length fam | readab. | fw fam |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.653 / 0.764 | 0.556 / 0.532 | 0.542 / 0.540 | 0.649 / 0.716 | 0.548 / 0.555 | 0.565 / 0.563 | 0.558 / 0.555 | 0.558 / 0.559 | 0.530 / 0.537 |
| same_topic | 0.556 / 0.699 | 0.561 / 0.532 | 0.544 / 0.540 | 0.587 / 0.676 | 0.568 / 0.537 | 0.557 / 0.545 | 0.568 / 0.542 | 0.552 / 0.546 | 0.525 / 0.525 |
| content_splice | 0.226 / 0.160 | 0.573 / 0.521 | 0.523 / 0.495 | 0.425 / 0.343 | 0.573 / 0.504 | 0.543 / 0.497 | 0.575 / 0.512 | 0.540 / 0.487 | 0.513 / 0.493 |
| length_splice | 0.644 / 0.762 | 0.045 / 0.018 | 0.508 / 0.508 | 0.570 / 0.678 | 0.409 / 0.481 | 0.477 / 0.500 | 0.135 / 0.195 | 0.448 / 0.478 | 0.512 / 0.521 |
| style_splice | 0.630 / 0.730 | 0.396 / 0.369 | 0.309 / 0.211 | 0.453 / 0.490 | 0.492 / 0.441 | 0.239 / 0.129 | 0.427 / 0.349 | 0.424 / 0.344 | 0.268 / 0.155 |
| style_nolen | 0.628 / 0.744 | 0.455 / 0.446 | 0.321 / 0.209 | 0.461 / 0.499 | 0.506 / 0.487 | 0.264 / 0.149 | 0.476 / 0.460 | 0.505 / 0.489 | 0.264 / 0.139 |
| topic_style | 0.347 / 0.314 | 0.430 / 0.405 | 0.388 / 0.302 | 0.381 / 0.305 | 0.514 / 0.445 | 0.348 / 0.257 | 0.456 / 0.388 | 0.431 / 0.374 | 0.371 / 0.286 |

(Content-overlap ratio change/within: Reddit random 0.05, content_splice 2.59, topic_style 1.69; C50 0.06 / 2.64 / 1.65.
CIs in the logs; all random-assembly CIs exclude 0.5. Style vector = z-scored non-POS features; 108 of 125 kept (Reddit)
and 97 of 112 (C50) are function words, so "style similarity" here is function-word-led.)

Verified readings:
- Holding the text fixed, random assembly gives the classical direction on every cue (CIs above 0.5): clearly for content
  overlap and trigrams (Reddit 0.653 / 0.649; C50 0.764 / 0.716), only weakly for the style families (Reddit 0.514-0.565;
  6 of 13 columns within 0.05 of chance in each corpus). Do NOT write "clearly normal on every cue".
- Starting each new author at the sentence most similar to the previous one on a cue reverses that cue: content overlap
  0.226 / 0.160, |log length ratio| 0.045 / 0.018, function-word family 0.268 / 0.155. Correlated cues reverse more weakly
  (Reddit length filter also reverses punct_dist 0.409, a per-character rate). Topic matching alone (same_topic) reverses
  nothing. Statement: "the sign of classical stylometric evidence at author changes is not a property of the text alone;
  the way splice points are chosen can reverse it."
- Reddit topic_style (20 most on-topic sources + splice on content-plus-style rank) reverses content (0.347), |log length
  ratio| (0.430; length family 0.456 near chance), function words (0.388; family 0.371), trigrams (0.381), readability
  (0.431) and the non-POS scaled distance (0.348); punctuation is not reversed (0.514; family 0.512). This matches the SIGN
  of PAN 2025 Hard on most cues but NOT its profile: length and trigram reversals are about 1/3 and 1/2 of Hard's
  (|AUC-0.5| 0.070 vs 0.213; 0.119 vs 0.241); function words reverse MORE than in Hard (0.388 vs 0.411); Hard's
  punctuation reversal (0.419) is absent; PAN Hard is length-led, the construction is function-word-led. It is a
  sufficiency demonstration on PAN-assembled source text; it does not show that PAN built Hard this way.
- PAN 2026 Hard (also described with a topical + stylistic similarity filter) is not reversed: present 2026 as a boundary
  condition. The sign agrees with whether cross-author similarity is pushed ABOVE natural within-author similarity
  (content ratio 2.39 in 2025 Hard vs 0.84 in 2026 Hard), but the magnitudes do not lie on the C50 dose-response curve.
- PAN 2025 Hard validation, joint matching: exact length alone brings fw_dist to chance (0.490 [0.471, 0.515]) and punct
  normal (0.546 [0.520, 0.574]); with length and content overlap jointly matched fw is near chance (0.533 [0.499, 0.569]),
  punct 0.607 [0.549, 0.672], style_L2 0.479 [0.417, 0.550] (inconclusive); character trigrams stay reversed (0.427
  [0.389, 0.464]; ~30% of the unmatched 0.259 reversal remains; within documents at matched length 0.401 [0.309, 0.485]).
  Length-free readability at exact matched length (no topic matching): syllables per word 0.444 [0.423, 0.467], complex-
  word ratio 0.437 [0.418, 0.456] (weakly reversed). Joint cells keep only pairs in two-class cells; CIs 2-3x wider.
- Repeated lines (PAN 2025 validation, all tiers pooled): the most frequent are subreddit moderation and bot messages
  (three r/politics rule lines, 595 copies each, 507 in Medium; a media-outlet notice 533; a moderator-recruitment pair 446;
  "I am a bot, and this action was performed automatically." 370), ~86% of the occurrences of the 25 most frequent lines;
  the rest are punctuation-only fragments, one-word replies, list numbers and "Fox News.". Changes are over-represented next
  to repeated lines at every cut-off (Easy 45.5/41.2/37.3% of changes vs 15.2/13.3/10.9% of no-changes at >= 3/10/30;
  Medium 42.5/41.7/41.5% vs 22.3/21.3/20.8%).
- Easy clean cell (no repeated line, both sentences >= 8 words; n 4,439) has change rate 0.054 vs 0.275-0.465 elsewhere:
  ~89% of Easy changes sit next to a repeated line or a sentence under 8 words. All raw cues normal there (0.592 / 0.837 /
  0.768 / 0.883; function-word family at chance 0.518). Medium clean cell (n 6,913, rate 0.180, ~45% of Medium changes): raw
  cues weakly normal (0.530-0.559), families within 0.05 of chance except length 0.556. Hard clean cell (n 7,566, rate
  0.217, ~86% of Hard changes): still reversed (fw 0.414, trigram 0.263, style_L2 0.378, all_z 0.363, readability 0.328,
  POS 0.356; punct 0.467 near chance).
- PAN 2025 TEST, saved predictions (3 gated + 3 concat encoders), official macro-F1: Medium boundaries next to a line with
  >= 10 copies (26.5% of boundaries, 41.4% of changes) 0.9835-0.9860; elsewhere 0.702-0.715 (above that subset's predict-0
  0.453); overall 0.819-0.826. Easy 0.994-0.997 near (20.8%) vs 0.915-0.933 elsewhere; Hard 0.908-0.940 near (4.2%) vs
  0.812-0.824 elsewhere. One-bit rule 'repeated => change': 0.650 Easy, 0.587 Medium (Hard 0.499). Estimated from the
  confusion counts: these boundaries supply ~60% of correctly detected Medium changes; if predicted as well as the rest,
  Medium macro-F1 would be ~0.70-0.72. Wording: "DeSAF is near-perfect next to repeated lines"; do NOT write "exploits" or
  "memorises" (untested; bot/moderator lines are genuine author changes). Parity with the PAN winners should be led by
  HARD, where repeated lines touch 4.2% of boundaries (DeSAF elsewhere 0.812-0.824). Whether other systems gain the same
  way is unverifiable.
- C50 transfer, official metric (3 gated encoders, no retraining): at each encoder's PAN threshold (0.50-0.64) 0.548-0.580
  on random (mean 0.563), 0.493-0.549 on topic-matched and spliced constructions (means 0.503-0.535): above predict-0
  (0.450) everywhere, at or barely above 0.5. ROC-AUC 0.68-0.78. C50-optimal thresholds 0.010-0.100. One threshold tuned on
  'random' and applied to the other constructions: 0.586-0.624 (means same_topic 0.599, similar_splice 0.613); on 'random'
  itself the tuned value is in-sample. In-domain: 0.953 / 0.824 / 0.826.

[Update 2026-11-09: all of the following have now been run; results in (K) and (L), except DeSAF on the Reddit TEST
constructions, now running.] Open (scripts written 2026-11-02): lag test (splice filter vs reply-echoing) and construction mechanism profiles
(pan_extra_checks.py); one shared TF-IDF for 2025 vs 2026 Hard; boilerplate edge rule + train overlap; construction runs on
Hard-only and Easy-only source text; DeSAF on constructed documents (desaf_on_constructions.py, C50 + Reddit TEST segments).

**Literature check (workflow, 2026-11-02; snippet-level, verify before citing):** verdict "partially
established". The broad idea that benchmark construction shapes what authorship features capture is
known (AV: Bevendorff 2019, Brad 2022, Wegmann 2022, Sawatphol 2024; multi-author decomposition:
Giannella 2016; PAN organisers on topic control). Not found anywhere: the sign reversal of classical
features across tiers, the Simpson's paradox, and a causal construction experiment. Tier routing is
anticipated by ULILLE (PAN 2026). Must read in full (ceur-ws blocked in sandbox): PAN 2025 notebook
Vol-4038 paper_295 (possibly OpenFact, per-tier feature tests), PAN 2026 overview (Vol-4283 paper388),
ULILLE (paper415), rheaveaura (paper414; check whether this is the user's own team).

---

**Primary sources read in full (uploaded PDFs, 2026-11-02):**
- PAN 2025 overview (paper_279): "For each sentence, we calculate stylistic and semantic feature
  vectors ... calculation of both the topic (semantic) and stylistic similarity ... applying different
  similarity thresholds for pairs of consecutive sentences ... relatively high for the easy dataset,
  moderate for the medium dataset, and low for the difficult dataset." Hard: "All sentences in a
  document pertain to a single topic and are stylistically similar." Easy: "relatively low stylistic
  similarity". => Hard is BUILT from stylistically similar consecutive sentences. The Hard reversal is
  (at least partly) a selection effect of the stated design. Quote descriptions, not threshold numbers.
- PAN 2026 overview (Vol-4283 paper388): fanfiction (AO3, top-100 fandoms, character pairs); per-
  sentence stylistic feature vector + SBERT semantic vector; "concatenated sentences based on their
  topical and stylistic similarity"; threshold "relatively low for the Easy ... high for the Hard"
  (opposite numeric wording to 2025, same meaning). 15,000 docs per tier. Results: best Medium 0.768,
  best Hard 0.780 (Hard > Medium again, as in 2025); easy "solved". The user's team appears in the
  results table as "rheaveauro" (typo for rheaveaura): 0.997 / 0.558 / 0.580.
- OpenFact (PAN 2025, Vol-4038 paper_295, Ksiezniak, Wecel, Sawinski): chi-squared tests of
  independence between punctuation-feature changes (ellipsis, ?, !, quotes, all-caps) and author change,
  per tier; significant associations in all tiers. NO direction reported. Does not pre-empt the sign
  reversal; cite as related per-tier evidence.
- ULILLE (PAN 2026, Vol-4283 paper415, Khelfaoui, Sileo, Loiseau): per-regime Ettin-400M cross-encoder
  specialists + a document-level router (head-tail, 64 sentences) with 0.9867 validation accuracy
  (confusions mainly easy<->hard; medium almost perfect). They state the router "learns dataset-level
  regularities of the data construction procedure". => tier routing is PRIOR WORK, and a strong router
  identifies the tier almost perfectly (on 2026 data). Our "86% is not enough" must be presented as a
  simple-classifier result, not as "tiers cannot be inferred".
- rheaveaura (PAN 2026, Vol-4283 paper414) = THE USER'S OWN notebook (Prisha Khalasi): SBERT + 12
  stylometric features + gradient boosting on PAN 2026; SBERT change/no-change similarity gap Easy
  0.549, Medium 0.118, Hard 0.038 (no-change 0.618 vs change 0.579 on Hard: still the normal direction,
  just tiny); GPT-2 NLL gap 0.788 / 0.333 / 0.190; validation macro-F1 0.998 / 0.555 / 0.582. Framed Hard
  as "leaving only pure stylistic differences"; the new paper should correct that framing (Hard is
  style-MATCHED by construction), cite it in the third person, and state the differences (PAN 2025 vs
  2026; classical features reverse vs an embedding gap that shrinks; causal C50 experiment).
  Note: its Table 3 cites better_call_claude's self-reported 0.923/0.828/0.724, not the official
  0.929/0.815/0.731.
- better_call_claude system notebook (arXiv 2508.00675v1, Schmidt, Roemisch, Halchynska, Gorovaia,
  Yamshchikov; uploaded PDF read in full 2026-10-31). "Sequential Sentence Pair Classifier" (SSPC):
  * Encoder: StyleDistance/styledistance, FROZEN (Table 2 "Base transformer frozen: True"), mean pooling,
    one vector PER SENTENCE (each sentence encoded alone; no cross-encoding of pairs).
  * Sequence model: BiLSTM, 5 layers, dropout 0.2, over the SENTENCE sequence; adjacent BiLSTM outputs are
    concatenated and fed to a 3-layer MLP (GELU, dropout 0.2) -> one logit per boundary. Loss: BCE. Batch 4,
    lr 5e-4 cosine to 5e-5, 30,000 steps, 2,600 warmup. One model for all three tiers (no tier input).
  * Training data: PAN 2025 train (all three tiers) PLUS the three PAN 2024 paragraph-level datasets turned
    into sentence-level labels (within-paragraph transitions = 0, first sentence of a new paragraph = 1).
  * Their Table 1 (split not stated) shows what the 2024 augmentation bought them: StyleDistance
    0.891/0.780/0.629 -> 0.922/0.828/0.723 (Hard +0.094, Medium +0.048); RoBERTa-base 0.929/0.764/0.613 ->
    0.930/0.787/0.656. StyleDistance beat RoBERTa-base, XLM-R-base and all-MiniLM-L6-v2 after augmentation.
  * Official test (their Table 3): 0.929 / 0.815 / 0.731, identical to the PAN overview. The abstract's
    "0.923, 0.828, 0.724 on the official test" matches their Table 1 augmented row, not Table 3: cite Table 3.
    Claude-3.7-sonnet zero-shot on the official test: 0.856 / 0.818 / 0.661.
  * Their own reading (Discussion): the model's strength "lies in exploiting the macrostructure of the
    problem and inter-sentence contextual patterns ... rather than in isolating purely stylistic signals."
    Limitations they state: frozen encoder; BiLSTM may not be optimal.
  * Lineage they credit: text segmentation as supervised sequence labelling with a BiLSTM over sentence
    embeddings (Koshorek et al., NAACL 2018), SegBot, SECTOR, and Hosseinia & Mukherjee (PAN 2018).
    => The encoder -> BiLSTM -> MLP skeleton predates both systems; cite Koshorek et al. 2018 as its origin.
  * Data facts they report (Appendix A, PAN 2025 train+validation): over 10% of sentences are exact
    duplicates (subreddit moderation boilerplate, up to 3,296 copies); hundreds of punctuation-only lines.
  * Differences from DeSAF: DeSAF cross-encodes each adjacent PAIR with DeBERTa-v3-small (top 2 layers
    trained), adds the stylometric difference vector, runs the BiLSTM over BOUNDARIES, trains on PAN 2025
    only, and the tier-aware head conditions on the tier. Shared: the encoder -> BiLSTM -> MLP skeleton.

---

**(K) User's runs 2026-11-02, second batch: PAN 2025 TEST-split rerun of the direction analysis
(logs: feature_direction_pan2025test, followup2_pan2025test) and the joint-matching checks on PAN 2026
validation (checks_pan2026). Verified in two rounds (one verifier, then three lenses: numbers, inference,
omissions).**

- Test replication (closes the "test-split confirmation" item). PAN 2025 TEST split: 900 documents per tier,
  34,371 boundaries (Hard 11,077 vs 10,648 on validation). Both TF-IDFs are fitted on the same 170,880 train
  sentences; overlap medians, quantile bins, length strata and part-C z-score SDs are recomputed within each split.
  One Hard test document whose number of labels was not its number of sentences minus one was truncated to the
  shorter. Of the 123 AUC cells that print a 95% CI in both the validation and the test logs (6 of them are raw_L2
  family duplicates of style_L2), none is significant in opposite directions and 114 keep the same CI class (above /
  below / spanning 0.5). The 9 that change class move between weakly significant and spanning 0.5, and all lie
  within 0.05 of chance on both splits (largest 0.038). |validation - test| over the 118 cells compared by script:
  mean 0.009, median 0.007, max 0.029 (Hard style_L2 at matched length, 0.421 [0.403, 0.439] vs 0.392 [0.379,
  0.404]). Two cells have non-overlapping CIs: Hard punct_dist unmatched (0.419 [0.406, 0.428] vs 0.440 [0.429,
  0.454]) and the Easy other-surface family (0.788 vs 0.806). Cells without CIs are less stable: Medium one-quote fw
  0.563 vs 0.496; Hard both-quote trigram 0.170 vs 0.263; Easy above-median-overlap fw 0.659 vs 0.609; several Hard
  per-bin rows change direction.
  Hard on test: fw 0.423 [0.410, 0.434], trigram 0.261 [0.249, 0.272], punct 0.440 [0.429, 0.454], style_L2 0.330
  [0.319, 0.343] (validation 0.411 / 0.259 / 0.419 / 0.348); content-overlap AUC 0.283 [0.272, 0.294], change/within
  content ratio 2.52 (validation 0.277, 2.39); |log length ratio| 0.281 [0.270, 0.294] (validation 0.287).
  Wording: "the direction pattern in the CI-bearing cells replicates on the held-out PAN 2025 test split; small
  sub-cells (per-bin rows, quotation groups) do not."
- Hard function words and length. Pairing changes with no-changes in the same length stratum brings most of the
  Hard function-word reversal to near chance: on test 0.423 -> 0.498 [0.485, 0.513] (function-word family 0.499
  [0.482, 0.515]; both CIs span 0.5); on validation 0.411 -> 0.476 [0.459, 0.494] (family 0.480 [0.464, 0.494];
  within 0.05 of 0.5 but the CIs are just below it). Exact length cells (validation only) give 0.490 [0.471,
  0.515]. Dropping short sentences does not do the same: with both sentences >= 8 words Hard fw stays reversed
  (validation 0.414, test 0.432; validation clean cell 0.414 [0.397, 0.430]); topic matching alone also leaves it
  reversed (0.416 / 0.404). Character trigrams stay reversed at matched length on both splits (test 0.349 [0.333,
  0.369], validation 0.354 [0.339, 0.371]) and under all three joint matchings on validation (0.333 / 0.427 / 0.401).
- PAN 2026, length and topic matched together (validation, 2,250 documents per tier): fw_dist, trigram_dissim and
  punct_dist have CIs above 0.5 in every tier under exact length, length x topic and document x length matching (27
  of 27 cells). Hard, length x topic: fw 0.563 [0.556, 0.570], trigram 0.604 [0.597, 0.610], punct 0.575 [0.569,
  0.584]; style_L2 0.506 [0.501, 0.512] (near chance). Medium: 0.565 / 0.583 / 0.556 / 0.512.
- PAN 2025 Hard vs PAN 2026 Hard after joint matching. Punctuation is above 0.5 in 2026 under all three matchings
  (0.573-0.582); in 2025 it is above 0.5 under length x topic (0.607 [0.549, 0.672]) and exact length (0.546
  [0.520, 0.574], near chance by the rule), but its document x length CI spans 0.5 (0.525 [0.431, 0.623]).
  Function words are above 0.5 in 2026 (0.563-0.567) but indistinguishable from chance in 2025 (exact length
  0.490 [0.471, 0.515]; length x topic 0.533 [0.499, 0.569]; document x length 0.500 [0.402, 0.591]); only under
  exact length do the two editions' CIs separate. Character trigrams are the only cue whose CIs fall on opposite
  sides of 0.5 under all three joint matchings (2025: 0.333 [0.312, 0.354], 0.427 [0.389, 0.464], 0.401 [0.309,
  0.485]; 2026: 0.624, 0.604, 0.621). style_L2 does so under exact length only (2025 0.421 [0.396, 0.448] vs 2026
  0.507 [0.502, 0.512]). Caveats: (1) the Hard tiers differ in topic structure (content-overlap AUC 0.277 in 2025 vs
  0.579 in 2026); (2) each edition has its own TF-IDF (170,880 vs 2,470,574 train sentences) and its own quantile
  topic bins; zero content overlap covers 58.4% of Hard boundaries / 22.4% of changes in 2025 vs 6.6% / 7.9% in
  2026; (3) the median shorter sentence is 12 vs 22 words; (4) the 2025 joint-cell CIs are 3-12x wider (5-12x for
  length x topic), consistent with fewer documents (900 vs 2,250), fewer boundaries (10,648 vs 127,380) and sparser
  joint cells; the logs do not separate these causes.
- Readability. PAN 2026 Medium and Hard: all five readability scores 0.502-0.522, unmatched and at exact matched
  length (near chance; at matched length 9 of the 10 CIs sit just above 0.5). Easy clearly normal (unmatched
  0.726-0.924; matched 0.849-0.910). PAN 2025 Hard length-free readability at exact matched length 0.444 [0.423,
  0.467] / 0.437 [0.418, 0.456]: reversed, just outside the near-chance band; its length-containing scores are
  reversed too (0.419-0.438); PAN 2026 Hard 0.505 / 0.507. PAN 2025 Medium is mixed (syllables per word 0.442
  [0.395, 0.484]; complex-word ratio 0.513 [0.461, 0.560]).
- PAN 2026 repeated lines. Copies counted over the analysed validation documents of all tiers: the most frequent
  line occurs 5 times (all in Medium), so no boundary is next to a line with >= 10 copies. At least 20 distinct
  lines of 18-73 words occur exactly once in each of the Easy, Medium and Hard validation documents. By character
  names and content they come from two sources: Dimitri/Byleth (9 lines, 3 without a name in the truncated listing)
  and Sherlock/Mycroft/John (11 lines). So some text appears verbatim in all three tiers. The counts do not show
  whether these are contiguous passages, whether they sit in the same documents or under the same author labels, or
  how much is shared; the listing is capped at 25 rows inside a tie at count 3 and never shows lines shared by two
  tiers, so 20 is a lower bound. (Already in G''''': no sentence under 8 words in Medium/Hard; the checks give the
  counts: 0 Medium and Hard boundaries, 11 of 117,435 Easy.)
- PAN 2026 feature families (clean cell = essentially all boundaries; z-scored families on the 400 part-C documents
  per tier). Hard (n 22,723): all_z 0.576, function words 0.567 and POS 0.554 clearly above 0.5; punctuation 0.538,
  other 0.524 and readability 0.519 have CIs above 0.5 but are near chance; length 0.496 [0.481, 0.508] (chance).
  Only the unscaled style_L2 has a CI below 0.5 (0.469 [0.459, 0.479]), and it tracks raw length differences
  (Spearman with |d chars| 0.919; |d chars| AUC 0.466); once length is matched it is at chance (stratified 0.500;
  part-C documents 0.501 [0.492, 0.512]; exact length 0.507 [0.502, 0.512]). Medium: all_z 0.583, families
  0.500-0.569, raw_L2 0.490 [0.476, 0.499]. Contrast: in the PAN 2025 Hard clean cell (both sentences >= 8 words, no
  repeated line; 7,566 of 10,648 boundaries) each family except punctuation is reversed (0.328-0.425; punctuation
  0.505); at matched length on all 2025 Hard boundaries most families move into or near the near-chance band (fw
  0.480, punct 0.514) while readability stays reversed (0.400). Family definitions differ (2025: 108 function words,
  4 'other', 56 dropped; 2026: 119, 3, 46); each edition is z-scored with its own SDs.
- Consistency. PAN 2026 clean-cell AUCs equal the all-boundary AUCs of feature_direction_pan2026 to the printed
  precision (Easy clean cell drops 11 of 117,435 boundaries; CIs differ in the third decimal from separate
  resampling). The test repeated-line shares in followup2_pan2025test (Easy 20.8% of boundaries / 44.1% of changes;
  Medium 26.5% / 41.4%; Hard 4.2% / 6.6%) equal those in checks_pan2025 part D to the printed precision (expected:
  both count copies over the test documents of all tiers with the same >= 10 rule).

**(T) Tier-flag mechanism (scripts/tier_mechanism.py, user's run, logs\tier_mechanism.log). VERIFIED 2026-11-09:
numbers lens and inference lens; their corrected wording is used below.**

- Reproduction. Part 2 retrains the concat_tier heads with gate_lab's --final procedure (default arguments, seeds
  7777+s, same 85/15 split). All 9 heads' true-flag predictions match the saved final_concat_tier_s*_test.npz files
  on 100.00% of boundaries (as printed, to 2 decimals), and their per-tier official scores equal the saved-prediction
  rescore to 4 decimals. True-flag test avg per encoder (mean of 3 heads): 0.8822 / 0.8823 / 0.8814; overall 0.8820 =
  section 11's tier-head mean. Original 0.8699 / 0.8671 / 0.8663 (mean 0.8678); no-flag concat heads 0.8643 / 0.8665 /
  0.8643 (from this log; mean 0.8650 = section 11's 9-run no-flag mean).
- Where the flag gain comes from (validation 5-fold CV only; official metric; mean of 3 head seeds; tier average).
  Flag gain: [CLS]-only head +0.0179 / +0.0187 / +0.0208 (encoders 43/44/45; mean +0.0191); paper head ([CLS] +
  stylometric difference vector) +0.0171 / +0.0180 / +0.0193 (mean +0.0181); stylometric-only head ([CLS] zeroed)
  +0.0288. The stylometric-only head is one experiment, not three: its inputs do not depend on the encoder, and its
  results are identical for the three encoders. Hard has the largest gain for the [CLS]-only head (+0.025 to +0.030)
  and the paper head (+0.028 to +0.031). The stylometric-only head gains most on Easy (+0.043; Hard +0.036, Medium
  +0.007), from a much lower base (0.737 vs 0.853-0.855 and 0.859-0.862).
  Reading (verified): the flag's gain does not depend on the explicit stylometric difference vector, since a head
  without it gains about as much. This does not show how the paper head uses the flag: the [CLS] vector comes from
  an encoder fine-tuned together with the stylometric features and sees the raw text, and a per-tier recalibration
  of the output would also give similar gains to every head (tier_ranking_check.py tests this). So the data no longer
  support "the flag works by switching the direction of the stylometric cues"; see the revision notes in section 4c.
  The per-run lines in Part 1 are gate_lab's positive-class F1; only the "flag gain in CV, official metric" lines are
  official.
- Forced-flag matrix (official test; mean of 3 heads per encoder; rows = true tier, columns = flag given). The
  diagonal equals the true-flag scores. Wrong flags: Hard documents given the Easy flag 0.650-0.668 and the Medium
  flag 0.737-0.758; Easy documents given the Hard flag 0.802-0.849 and the Medium flag 0.922-0.950; Medium documents
  given the Easy flag 0.801-0.809 and the Hard flag 0.752-0.776. Each of the 18 off-diagonal cells (3 encoders x 6)
  is below the no-flag head on that true tier, both the 9-run mean (0.9576 / 0.8175 / 0.8200) and the same encoder's
  3-head mean. The margin goes from slight (Easy read as Medium, 0.008-0.035 below) to large (Hard read as Easy,
  0.15-0.17 below). The costliest error in every encoder is a Hard document read as Easy, 0.178-0.193 below its
  true-flag score; next is an Easy document read as Hard (0.12-0.17). A wrong flag is worse than no flag.
- Accuracy curve (each test document keeps its true flag with probability a, otherwise one of the two wrong flags at
  random; 100 draws; mean over 3 heads). Avg official macro-F1 falls from 0.8814-0.8823 (a = 1) to 0.8200-0.8267 at
  a = 1/3 (flags at random). Lowest grid accuracy still at or above the original DeSAF: 0.86 / 0.86 / 0.82
  (interpolated 0.85 / 0.83 / 0.81); at or above the no-flag head: 0.80 / 0.86 / 0.80 (interpolated 0.78 / 0.82 /
  0.79). Under random errors a tier guess must be right for roughly 81-85% of documents just to match the tier-free
  DeSAF; keeping half or more of the gain over DeSAF needs about 92-95% [corrected in (P): about 91-92% by interpolation, 0.906 / 0.914 / 0.923] (a = 0.95 keeps 68-74%, a = 0.98 keeps
  87-89%). Our hard-argmax predicted-tier runs (0.852-0.854 per encoder) fall below this curve at their classifiers'
  accuracy. The soft-tier and retrained runs are not hard flags into these heads and cannot be placed on it. At
  ULILLE's accuracy on PAN 2026 validation (0.9867) the curve gives about 0.880-0.881; even if every error were the
  costliest confusion (Hard read as Easy) the loss would be only about 0.002 more, so error structure matters little
  at that accuracy. Whether such accuracy is reachable on PAN 2025 is untested; our own PAN 2025 tier classifiers
  reach 0.76-0.86.
- Lead link (3) of the paper with the controlled comparison: tier head vs the same head without the flag (+0.0169,
  26/27 runs significant after Holm), not vs the original DeSAF (+0.0142; the original never trained on validation).

**(L) User's runs pasted 2026-11-04: constructions on Hard-only, Easy-only and TEST-split Reddit text; DeSAF on the C50
constructions; pan_extra_checks (lag test, mechanism profile, shared TF-IDF, boilerplate). VERIFIED 2026-11-09:
numbers checked mechanically against the logs; interpretive sentences checked by an adversarial inference verifier,
whose corrected wording is used below.**

- Hard-only source (train split; 9,271 segments of >= 2 sentences; 1,119 dropped for containing a repeated line).
  Glued at random, Hard-tier author runs show no reversal on any of the 13 measures. Content (0.670) and trigram
  (0.654) are clearly normal; the other 11 lie at 0.524-0.554 with CIs above 0.5, nine of them within the near-chance
  band; |log length ratio| is 0.524 vs PAN Hard 0.287. So the reversed sign does not come from the Hard tier's
  sentences or author runs taken alone. It needs the way PAN joins them, and random assembly removes that together
  with the shared thread and topic, single-sentence runs and boilerplate segments. Hard text does carry weaker
  length, readability and punctuation cues than Easy text even at random junctions (0.524 / 0.548 / 0.551 vs 0.592 /
  0.631 / 0.605). Easy-only random: all 13 above 0.5 except the function-word family, 0.506 [0.494, 0.520]. Under
  topic_style, Hard-only text reverses content 0.330, fw 0.375, trigram 0.357, all_z 0.327, |log length ratio|
  0.433, function-word family 0.358 and readability 0.423; punct 0.487 [0.474, 0.500] and the punctuation family
  0.502 are near chance. That is slightly stronger than Easy-only text (content 0.375, trigram 0.414) or all-tier
  text (trigram 0.381), but still far from PAN Hard on trigram (0.259) and length (0.287).
- Reddit constructions built from the PAN 2025 TEST split (6,130 segments; 2,100 dropped for repeated lines) behave
  like the train-built ones. Random shows no reversal: all 13 CIs above 0.5 (0.528-0.658), although the fw family
  0.528, all_z_no_len_read 0.540 and the punctuation family 0.544 are near chance. topic_style reverses content
  0.358, fw 0.394, trigram 0.394 and all_z 0.357; |log length ratio| 0.450 is on the near-chance border and punct
  0.504 [0.490, 0.517] is at chance (train-built: 0.347 / 0.388 / 0.381 / 0.348 / 0.430 / 0.514). The pattern is not
  specific to the train split. A second run printed identical numbers: a reproducibility check of a deterministic
  script, not a replication.
- DeSAF on the C50 constructions (no retraining; 3 gated encoders scoring the same documents; official metric). AUC
  is highest on random for every encoder (0.759-0.778) and lowest on same_topic and topic_style (means 0.700 and
  0.687). Mean drops vs random: topic_style -0.079, same_topic -0.067, content_splice -0.051, style_splice -0.037,
  length_splice -0.033. At each encoder's PAN threshold macro-F1 is only 0.501-0.580 (predict-0 0.450), and there
  same_topic, not topic_style, is lowest (mean 0.511 vs 0.520). With one threshold tuned on random and transferred,
  the other constructions score 0.595-0.629. Reading: on unseen-domain text the PAN-trained detector loses most when
  the sources share a topic. A style-similarity splice lowered AUC less than topic matching in each encoder
  (0.032-0.045 vs 0.065-0.070); how much less is not established (ratios 0.45-0.70 by encoder, no CIs yet, and each
  drop depends on how strongly each construction was selected). The detector does not do better on the construction
  that reproduces PAN Hard's reversed signs: its AUC there is about as low as on same_topic (0.009-0.016 lower, within
  sampling noise), so most of that loss is what topic matching alone costs. The C50 random row equals the earlier C50
  multi-author rescore for every encoder (AUC 0.7783 / 0.7592 / 0.7616; @PAN 0.5608 / 0.5477 / 0.5800; @rand 0.6629 /
  0.6458 / 0.6542), so both scripts built the same random documents; their same_topic documents differ (change rate
  0.181 vs 0.182; AUC 0.7136 / 0.6943 / 0.6916 vs 0.7175 / 0.6950 / 0.6894), reason not recorded.
- Lag test (PAN 2025 Hard validation; only boundaries whose previous three sentences share an author: 5,905, of
  which 1,039 are changes). Content at lag 1 / 2 / 3 = 0.283 / 0.482 / 0.482; trigram 0.286 / 0.487 / 0.478. The
  extra similarity at Hard changes is concentrated at the junction pair; one and two sentences further back it is
  near chance, though still slightly reversed (content CIs [0.469, 0.496] and [0.469, 0.495]). The known splice
  constructions show the same drop after lag 1 (Reddit topic_style content 0.343 -> 0.505 / 0.526; C50 topic_style
  0.304 -> 0.576 / 0.577; C50 content_splice 0.155 -> 0.547 / 0.552). Topic matching without a splice (same_topic) is
  not reversed at any lag. Near-duplicate first sentences are rare (Hard changes 0.9%, no-changes 0.5%). Easy and
  Medium are not reversed at any lag. Wording: "The extra similarity sits at the junction pair, as a filter on that
  pair would produce. This argues against an echo of the previous author's whole run at equal strength. There is no
  natural-reply calibration, so the test cannot separate such a filter from replies that answer mainly the
  immediately preceding sentence." A shared topic alone would put lags 2-3 at about 0.5, and a known junction filter
  also leaves some residual, so the small residual at lags 2-3 does not discriminate.
- Mechanism profile (fw / trigram / punct; unmatched -> exact length -> length x topic). PAN 2025 Hard validation:
  fw 0.411 -> 0.490 -> 0.533; trigram 0.259 -> 0.333 -> 0.427; punct 0.419 -> 0.546 -> 0.607. Reddit topic_style: fw
  0.388 -> 0.393 -> 0.404; trigram 0.381 -> 0.358 -> 0.416; punct 0.514 -> 0.552 -> 0.548. C50 topic_style: fw 0.302
  -> 0.301 -> 0.327; trigram 0.305 -> 0.294 -> 0.385; punct 0.445 -> 0.457 -> 0.467. In PAN Hard, matching length
  brings fw to near chance and turns punct normal; in the constructions the fw reversal survives length matching, as
  expected if it is selected directly through the style vector (mostly function words). After joint matching the
  trigram residual is similar for Reddit topic_style (0.416) and PAN Hard (0.427), but length matching moves PAN Hard
  (0.259 -> 0.333) and not the construction. Share of changes in informative cells at length x topic: PAN Hard 62%,
  constructions 64-87%. This supports (J): topic_style reproduces the direction of PAN Hard's content, trigram and
  function-word reversals but not how they arise (length-led in PAN Hard, selected directly in the construction), and
  on Reddit text it does not reproduce PAN Hard's punctuation reversal (0.514 vs 0.419).
- One content TF-IDF fitted on both editions: overlap at Hard changes differs little between editions (mean 0.165 vs
  0.183; medians 0.130 vs 0.171); overlap between consecutive same-author sentences differs about threefold (means
  0.067 vs 0.213; medians 0.000 vs 0.202). PAN 2025 Hard changes share far more content than same-author neighbours
  (ratio 2.45, AUC 0.275); PAN 2026 Hard changes share slightly less (ratio 0.86, AUC 0.569); per-edition TF-IDF
  ratios were 2.39 and 0.84. The within-author level also varies by tier inside an edition (PAN 2026: Easy 0.221,
  Medium 0.073, Hard 0.213), so it reflects how each tier's documents were built. The shared vectorizer equalises IDF
  weights, not sentence length (median shorter sentence 12 vs 22 words) or vocabulary. This fits "the sign depends on
  where cross-author similarity sits relative to the tier's within-author similarity" but cannot establish it.
- Boilerplate (PAN 2025 test). 42 distinct test lines occur >= 10 times; all 42 also occur in the train text and in
  the validation text. Boundaries next to such a line are 20.8% of Easy, 26.5% of Medium and 4.2% of Hard test
  boundaries. On them an edge rule (change iff exactly one of the two sentences is a repeated line) reaches macro-F1
  0.828 / 0.901 / 0.207. Entering or leaving a block of repeated lines is usually a change (75% in Easy, 83% in
  Medium), and a boundary inside such a block rarely is (6% / 5%), so in Easy and Medium these boundaries are largely
  predictable from which lines are boilerplate. DeSAF is still higher there (0.994-0.997 / 0.984-0.986). All repeated
  test lines occur in the training text, so a model could learn them; whether DeSAF relies on that is untested. In
  Hard the pattern is reversed (both-repeated boundaries are mostly changes, 68%; exactly-one-side 20%) and the rule
  fails.
- Notebook only (not for the paper): in the first desaf_on_constructions run the reddit2025test files were not found
  although the construction run reported saving them; cause not recorded. They were present at the next run.

---

**(M) User's runs pasted 2026-11-09: DeSAF on the Reddit TEST constructions, and tier_ranking_check.py (ORIGINAL version,
thresholds pooled over folds). Numbers transcribed and derived values recomputed; NOT yet adversarially verified.
PROVISIONAL until the patched tier_ranking_check rerun and bootstrap_ci_check.py (CIs) are in.
[2026-11-09: SUPERSEDED by (N) for every reading; keep (M) only for its transcribed numbers.]**

- DeSAF on Reddit constructions built from PAN 2025 TEST text (3 gated encoders, no retraining, official metric).
  Mean ROC-AUC: random 0.809 (0.802-0.818), same_topic 0.798, content_splice 0.784, length_splice 0.790, style_splice
  0.781, topic_style 0.787. Mean drop vs random: same_topic -0.011, length_splice -0.019, topic_style -0.022,
  content_splice -0.025, style_splice -0.028 (every construction below random in every encoder; the order of the
  other five differs by encoder). Macro-F1 at the PAN threshold: random 0.680 (0.672-0.695), the others 0.654-0.671
  (predict-0 0.430-0.438). So on held-out Reddit text the C50 pattern does NOT hold: topic matching alone costs least
  (-0.011 vs -0.067 on C50), and every drop is small (0.011-0.028; CIs pending).
  Pre-registered check (critic, 2026-11-09): P1 "random AUC >= 0.90" FAILS (0.809); P2 "same_topic and topic_style are
  the two lowest" FAILS; rival "topic_style >= random - 0.01" FAILS (0.787 vs 0.799). Per the pre-registration: report
  the numbers, no mechanism sentence.
  Also: DeSAF's AUC on randomly re-assembled Reddit test text (0.80-0.82) is far below its AUC on PAN's own test
  documents (original DeSAF per tier 0.991-0.992 Easy, 0.928-0.931 Medium, 0.933-0.940 Hard). Re-assembly removes
  boilerplate segments, single-sentence runs and thread structure; which of these DeSAF relies on is untested.
- tier_ranking_check (original version). "As run" rows reproduce tier_mechanism.log exactly (e.g. seed43 paper head
  0.8610 -> 0.8781). Threshold-free results (unaffected by the patch):
  * Validation CV, ROC-AUC within folds, paper head: flag gain +0.0037 / +0.0081 / +0.0272 / +0.0130 (Easy / Medium /
    Hard / avg; mean of 3 encoders; Hard +0.026 to +0.029 in every encoder). [CLS]-only head: +0.0142 / +0.0145 /
    +0.0158 avg. Stylometric-only head: +0.0367 avg (Hard +0.0668).
  * Official test, ROC-AUC, concat_tier vs concat: +0.0027 / +0.0131 / +0.0195 / +0.0118 (mean of 3 encoders; Hard
    +0.0175 to +0.0228, positive in every encoder and tier).
  * Official test, no-flag head with per-tier thresholds TUNED ON TEST (an unfair upper bound): 0.8681 / 0.8697 /
    0.8670, i.e. +0.003 to +0.004 over its as-run 0.8643 / 0.8665 / 0.8643, closing 16-21% of the gap to the flag head
    (as run 0.8822 / 0.8823 / 0.8814).
  * Validation CV threshold rows (pooled version, to be replaced by the patched per-fold rows): per-tier thresholds
    alone add +0.0010 to +0.0016 to the no-flag paper and [CLS]-only heads (5-9% of the flag gain), +0.0064 to the
    stylometric-only head (22%).
  Pre-registered decision (critic): (ii) within-fold AUC gain on Hard > 0 for all encoders and avg >= +0.005 HOLDS;
  (iii) test AUC gain on Hard > 0 for all encoders HOLDS; (iv) no-flag head with test-tuned per-tier thresholds stays
  below 0.8820 HOLDS; (i) per-tier thresholds give the no-flag paper head <= +0.005 HOLDS in the pooled version,
  to be confirmed by the patched per-fold rows. If (i) holds there too, the verified sentence will be: "the oracle
  flag changes the ranking within a tier, not only the operating point."

---

**(N) User's runs pasted 2026-11-09: tier_ranking_check.py (PATCHED: thresholds chosen inside each CV fold; fold check
passed), bootstrap_ci_check.py (2,000 document resamples), length_split_check.py. VERIFIED 2026-11-09: every number checked
against the log (logs_user/trc_v2_bootstrap_lengthsplit.log) and every derived figure recomputed by an adversarial verifier;
the wording below is the verifier's corrected text. This section supersedes the provisional readings in (M).**

- Tier flag: ranking, not only threshold (all four pre-registered criteria hold). "The oracle tier flag changes the ranking
  within a tier, not only the operating point. In 5-fold CV (thresholds chosen in each fold, official macro-F1), per-tier
  thresholds add at most about +0.003 to the head without the flag over one tuned threshold (paper head +0.0032 / +0.0031 /
  +0.0032; [CLS]-only +0.0030 / +0.0032 / +0.0027; stylometric-only +0.0084). This is in-sample and therefore an upper
  bound: threshold optimism alone gives about +0.0025 at this cell size. With both heads at per-tier thresholds, the flag
  keeps 96-99% of its gain (+0.016 to +0.018; [CLS]-only 95-102%, stylometric-only 89%). The flag also raises within-tier
  ROC-AUC, which no threshold can change: on Hard by 0.026-0.029 in CV and by 0.018-0.023 on the official test (Medium
  0.009-0.018, Easy 0.002-0.003). On the official test, the head without the flag stays 0.0125-0.0142 below the flag head
  even with per-tier thresholds tuned on test (0.8681 / 0.8698 / 0.8672 vs 0.8822 / 0.8823 / 0.8814)." Against the as-run
  threshold (not one tuned threshold) per-tier thresholds add +0.006 to +0.007 in CV; the share line, where the optimism
  cancels between heads, is the fair comparison. Do not set the CV macro-F1 figure against the test ROC-AUC figure in one
  sentence as if comparable. This replaces the section 4c "NOT calibration" sentence (which used positive-class F1).
- Level with the PAN 2025 winners. "Tier-free DeSAF is level with the two best PAN 2025 systems within our test-sampling
  uncertainty. The 3-seed mean's gaps to wqd and xxsu-team are -0.005 to +0.001, each smaller than the 95% CI half-width of
  that mean on the same tier (Easy 0.006, Medium 0.010, Hard 0.009; means 0.9530 [0.9472, 0.9586], 0.8242 [0.8137,
  0.8343], 0.8261 [0.8173, 0.8352]). wqd's Easy score (0.958) lies at the upper edge of that CI and above all three of our
  seeds (0.948-0.956). The CI covers test sampling only, not training randomness: across seeds, scores range 0.948-0.956 on
  Easy, 0.822-0.826 on Medium and 0.819-0.830 on Hard. The winners' predictions are not available, so no paired test is
  possible." Per-run CI half-widths: Easy 0.006-0.007, Medium 0.011, Hard 0.009-0.010; for single runs the winner falls
  outside the run's CI in 3 comparisons (seed45 Easy vs wqd and xxsu; seed44 Hard vs wqd). One Hard test document whose
  label count does not match its sentence count is truncated in the direction scripts (11,077 boundaries) and kept in full
  by the prediction loader (11,080).
- DeSAF on constructed documents, with CIs (3 gated encoders, no retraining). "C50: every construction is below random
  (drops 0.033-0.079, all CIs exclude 0). The two topic-matched constructions drop most (topic_style -0.079, same_topic
  -0.067). A style splice costs less than topic matching (+0.030 [+0.013, +0.045]), and topic_style vs same_topic is not
  resolved (-0.013 [-0.029, +0.004]). Reddit test: all four junction splices are below random (-0.019 to -0.028; CI upper
  ends -0.003 to -0.011), are not resolved from one another, and all but style_splice are not resolved from same_topic.
  same_topic itself is not resolved from random (-0.011 [-0.027, +0.006]). The C50 order does not hold on Reddit:
  style_splice - same_topic is -0.017 [-0.034, -0.0005], its CI only just excluding 0. How much topic matching costs
  differs between the two sources (-0.067 vs -0.011)." Numbers only, no mechanism sentence (pre-registered). DELETED: "a
  style splice costs about half as much" (even on C50 only "less than" is supported). Mean AUC [CI]: C50 random 0.7663
  [0.7561, 0.7766] ... topic_style 0.6873 [0.6763, 0.6980]; Reddit random 0.8090 [0.7974, 0.8201] ... topic_style 0.7870
  [0.7751, 0.7985] (all in the log).
- Where the PAN 2025 Hard length effect comes from (validation / test; gate: unmatched |log length ratio| 0.287 equals the
  earlier value; test 0.283 vs earlier 0.281, cause not recorded).
  * Placement: Hard changes sit between longer sentences (AUC that the shorter sentence is longer at a change 0.718 /
    0.704). Change rate by the shorter sentence's word count: 0.056-0.078 at 4-11 words, 0.169 at 12-19, 0.352-0.356 at
    20-29, 0.468-0.486 at 30+ (1-3 words 0.130 / 0.134). Easy is the opposite (0.388 / 0.414 at 1-3 words down to 0.000
    at 30+; AUC 0.272 / 0.267). PAN 2026 Hard leans the other way (AUC 0.447; change rate 0.241 / 0.218 / 0.146 at 12-19 /
    20-29 / 30+); PAN 2026 has no pairs under 8 words and only 2 at 8-11, so that comparison covers 12+ words only.
  * "Placement and closeness in length each produce most of the reversal on their own. With the shorter sentence's length
    held fixed, the two sentences at a change are still closer in length than at a same-author boundary (0.356 [0.336,
    0.374] validation / 0.338 [0.319, 0.356] test). With the word-count difference held fixed, changes still sit between
    longer sentences (0.690 / 0.674). Fixing placement removes about a quarter to a third of the unmatched reversal (s =
    0.32 / 0.25; pre-registered verdict 'both' on validation, 'similar length beyond placement' on test, with s near the 0.3
    cutoff on both). Fixing the difference removes less (0.08 / 0.18)." Paper sentence: "Hard changes join sentences of
    similar length beyond placement: holding the shorter sentence's length fixed removes only about a quarter to a third
    of the length reversal."
  * By cue: trigram's length component is placement (G_min 0.332 / 0.323 within 0.001 / 0.006 of exact 0.333 / 0.329;
    G_diff 0.285 / 0.285; meets the pre-registered rule). style_L2 (unscaled, length-dominated): the length difference
    moves it most of the way (0.348 / 0.330 to 0.396 / 0.388), placement hardly at all (0.354 / 0.332); G_diff stays 0.025
    short of exact (0.421 / 0.413), so no driver is named. fw: neither grouping alone comes within 0.02 of exact (G_min
    0.453 / 0.465, G_diff 0.430 / 0.429 vs 0.490 / 0.492); placement moves fw more (+0.042 on both splits vs +0.019 /
    +0.006); no driver is named. punct: placement alone removes the reversal on both splits (0.509 ~ / 0.527 N);
    difference alone only on test (0.468 R / 0.495 ~); neither within 0.02 of exact (0.546 / 0.569).
  * Constructions: "Random assemblies show placement only within the near-chance band (0.475-0.515) and no length reversal
    (0.522-0.555). topic_style shifts changes toward longer sentences compared with random assembly from the same source
    (Reddit 0.475 to 0.550, Hard-only 0.515 to 0.585, C50 0.492 to 0.549). Only Hard-only is clearly outside the
    near-chance band. Its length reversal does not depend on placement (s -0.15 / 0.10 / -0.01), and its fw reversal barely
    moves under any grouping (Reddit 0.386-0.394)." Signs, not mechanism, again.
  * Pre-registered predictions: P1 (|log len| under G_min 0.36-0.46) fails (validation 0.356, just below 0.36; test
    0.338); P2 (punct >= 0.50 under G_min) holds; P3 (fw driven by difference, not placement) fails; P4 (trigram <= 0.36
    under both) holds; P5 (Hard change rate rises with length, Easy does not) holds except the 1-3-word bin; P6 (PAN 2026
    Hard |log len| under G_min within 0.05 of chance) holds (0.479). The optional topic_length / topic_rawstyle
    constructions are permitted by the pre-registration (s <= 0.6).

---

**(O) User's runs pasted 2026-11-16: hard_error_profile.py, construction_experiment.py (topic_length, topic_rawstyle),
construction_profile_check.py, boilerplate_mask_test.py. Logs: logs_user/hard_error_profile.log,
construction_tl_and_profile_check.log, boilerplate_mask_test.log. VERIFIED 2026-11-16 by four lenses (numbers,
pre-registration, inference, code): every transcribed number matches the logs and every derived figure recomputes; the
wording below is the verifiers' corrected text.**

- Error profile, PAN 2025 test (hard_error_profile.py). Junction similarity = mean of three within-tier percentile ranks
  (content overlap, -|log length ratio|, -trigram dissimilarity); tertile cut points per tier, so tertiles are relative
  to each tier's own distribution. Composite AUC P(change more similar): Easy 0.182, Medium 0.492, Hard 0.789. Changes
  per tertile (least -> most similar): Easy 1572 / 597 / 69; Medium 813 / 1176 / 808; Hard 100 / 427 / 1384 (72% of Hard
  changes in the most-similar third).
  * P1 (pre-registered: original DeSAF's recall at least 0.10 lower in the most-similar than the least-similar tertile on
    all 3 encoders): Hard +0.092 / +0.126 / +0.145, mean +0.121 [+0.053, +0.184]. FAILS (seed43 below 0.10). The sentence
    "DeSAF misses Hard changes where the junction is most alike" is NOT written. (Easy +0.044 [-0.018, +0.112]; Medium
    +0.002 [-0.038, +0.045].)
  * P2 (pre-registered: tier head's recall gain over the no-flag head larger in the most- than the least-similar tertile,
    all 3 encoders): holds numerically on Hard (+0.076 vs +0.010; +0.049 vs -0.060; +0.081 vs +0.023), but its sentence is
    conditional on P1 and is NOT written. Seed44 passes partly through a -0.060 recall change in a 100-change cell (SE of a
    recall difference there about 0.05-0.06).
  * Post hoc, not pre-registered, no CIs (allowed wording): "The pre-registered test that DeSAF misses more Hard changes at
    the most alike junctions (recall at least 0.10 lower on every encoder) failed narrowly (gaps +0.092 / +0.126 /
    +0.145). That gap comes from the least-alike third, which holds only 100 of 1,911 Hard changes; recall in the middle
    and most-alike thirds is the same (0.705-0.738 vs 0.714-0.728). DeSAF's within-tertile ROC-AUC on Hard is lowest among
    the most alike third of junctions (0.881-0.884, vs 0.943-0.951 in the middle third and 0.970-0.981 among the least
    alike), which holds 72% of Hard changes. The weaker separation there shows mainly as more false positives on
    look-alike same-author pairs (FPR 0.138-0.149 vs 0.041-0.049 in the middle third), not as lower recall." On Medium the
    AUC fall is smaller (0.941-0.942 -> 0.924-0.928 -> 0.916-0.922); on Easy there is no consistent fall (0.990 -> 0.980 on
    two encoders, 0.988 -> 0.993 on the third; the most-similar Easy tertile holds only 69 changes).
  * Tier head vs no-flag head, post hoc, no CIs: in the most-similar Hard tertile the within-tertile ROC-AUC gain is the
    smallest of the three tertiles on every encoder (+0.0125 / +0.0093 / +0.0059 vs +0.0184 / +0.0199 / +0.0153 in the
    least similar), and the tier head's false-positive rate there is higher (+0.027 / +0.008 / +0.043). So its recall gain
    there is not only better ranking within that tertile; it also calls more changes there (between tertiles it shifts its
    change calls from unlike to look-alike junctions, which within-tertile AUC cannot see). In the least-similar tertile
    (100 changes, about 3,600 same-author pairs) it makes fewer false positives (FPR 0.026-0.027 -> 0.005-0.009), raising
    macro-F1 there by 0.097-0.115, while recall there changes by -0.060 to +0.023. Not to be used as a claim.
  * NOT allowed: "the tier flag fixes look-alike junctions"; "the flag helps where similarity misleads"; "the gain at
    look-alike junctions is a threshold artefact"; "look-alike junctions explain most of Hard's misses" (misses are about
    proportional to where the changes are).
- Length-aware constructions (topic_length = content rank + length-closeness rank; topic_rawstyle = content rank + rank
  of the UNSCALED 182-feature style distance). Pre-registered criteria C1-C7 (construction_profile_check.py docstring).
  * Reproduction: reddit2025test_tl random and topic_style equal the earlier reddit2025test run in point AUCs (all 13),
    boundary counts (7,541 / 8,101) and change rates (0.244 / 0.227); topic_style bootstrap CIs differ slightly (e.g.
    punct [0.491, 0.516] vs [0.490, 0.517]; the RNG is shared across constructions). On C50 the random and topic_style
    unmatched and exact-length profile values and all lag-1 values equal extra_checks.log; the length x topic cells differ
    by 0.001-0.008 (topic_style fw 0.335 vs 0.327, trigram 0.390 vs 0.385, punct 0.470 vs 0.467; random fw 0.470 vs
    0.469) and one lag value by 0.001 (topic_style content lag 3, 0.576 vs 0.577), probably because the content TF-IDF is
    now fitted on four constructions instead of five; not checked by refitting. C50 placement equals (N) (0.492 / 0.549).
  * Verdict: no construction meets C1-C5 on either source, so no construction "reproduces PAN Hard's profile". Criteria
    met. Reddit test text: topic_length C2, C4, C6; topic_rawstyle C2, C4, C6; topic_style C5, C6; random none. C50:
    topic_style C1, C5; topic_rawstyle C1, C2; topic_length C2; random none. No construction meets more than two of C1-C5;
    none meets C3 on either source; none meets C7 (short/long ratio; closest Reddit topic_length 0.59 vs PAN Hard 0.39).
  * Pre-stated predictions not met: the critic's "topic_rawstyle meets C1, C2 and C5 but fails C7" was partly wrong (C2
    on Reddit, C1 and C2 on C50, never C5; it did fail C7); "topic_length fails C3" and "no full match" held. The generic
    expectation in construction_experiment.py ("cues it does not select on stay normal or go to chance") is not met for
    topic_length: punctuation (0.442 Reddit / 0.449 C50) and readability (0.418 / 0.445) are also reversed, plausibly
    through their correlation with length, which was not pre-specified.
  * Allowed reading: "Each of the three topic recipes meets some of the pre-registered criteria, but none meets more than
    two of C1-C5, and none meets C3 (the length-led function-word pattern) on either source. Selecting on length
    (topic_length, topic_rawstyle) gives a length reversal stronger than PAN Hard's on both sources (|log length ratio|
    0.155 / 0.073 and 0.218 / 0.161 vs 0.287) and, on Reddit text only, PAN Hard's pattern where the punctuation reversal
    disappears at matched length (0.442 -> 0.567, 0.437 -> 0.539 vs 0.419 -> 0.546; on C50 0.449 -> 0.494 and 0.415 ->
    0.467). Selecting on the standardised style vector (topic_style) gives a trigram reversal that survives length and
    topic matching on both sources (0.424 / 0.390 vs 0.427); selecting on the unscaled vector does not (0.564 / 0.539). We
    did not search further: a recipe tuned until it matches would no longer be an independent test."
    Descriptive: the length-aware recipes' function-word cue is near chance from the start on Reddit (0.489 [0.475,
    0.501] / 0.507 [0.494, 0.518]; PAN Hard 0.411) and weakly reversed on C50 (0.463 / 0.465); topic_style's length
    reversal is weak (0.450 / 0.405, C2 not met) and its function-word reversal survives exact length (0.399 / 0.301; PAN
    Hard 0.490). Placement moves toward PAN Hard (topic_length 0.641 / 0.635, topic_rawstyle 0.635 / 0.607; random 0.463
    / 0.492; PAN Hard 0.718).
    NOT allowed: "reproduces (part of) PAN Hard's profile"; "PAN Hard was built by selecting on topic, length and style";
    "a combined recipe would reproduce it".
  * topic_rawstyle: meets the same criteria as topic_length on Reddit (C2, C4, C6). Its length (0.243 / 0.138) and
    readability (0.298 / 0.218) families are strongly reversed, while the summary without length and readability is near
    chance on Reddit (0.508 [0.496, 0.523]) and weakly reversed on C50 (0.474 [0.462, 0.486]). Consistent with the unscaled
    distance being dominated by length and readability features, as the script assumed and as (N) found for unscaled
    style_L2; the feature weights themselves were not measured. Compared with topic_length it reverses readability and
    other_surface more and |log length ratio| less.
  * Lag (C6): on Reddit all three topic recipes meet C6 (lag-1 content 0.368-0.389, reversed but weaker than PAN Hard's
    0.283; lags 2-3 0.522-0.544). Their lags 2-3 lie slightly on the normal side (every CI above 0.5), PAN Hard's slightly
    on the reversed side (0.482, CIs below 0.5). On C50 lags 2-3 are 0.564-0.577, outside the band.
  * Definitions: the construction |log length ratio| uses +1-smoothed word counts, the PAN reference does not (about
    0.002 apart where both were computed; no criterion changes). The recipes' rank sums use ordinal ranks, so ties
    (frequent at zero content overlap) are broken by source order, the same for all three topic recipes.
- Boilerplate masking, PAN 2025 test (boilerplate_mask_test.py). The 42 test lines with >= 10 copies were replaced by one
  placeholder ("This sentence has been replaced."); the 575 / 553 / 172 of 900 documents containing such a line were
  re-scored; all other boundaries keep their saved predictions.
  * Reproduction: 30 unchanged documents per tier re-scored, prediction agreement 1.0000 and max probability difference
    0.0000 in all 9 runs; unmasked tier scores equal the per-seed official numbers (43: 0.9546 / 0.8249 / 0.8301; 44:
    0.9564 / 0.8259 / 0.8190; 45: 0.9479 / 0.8218 / 0.8293); unmasked near-repeated scores equal extra_checks.log.
  * EDGE boundaries (exactly one side a repeated line; the pre-registered subset; 12.0% / 9.8% / 3.6% of boundaries):
    Easy 0.990-0.992 -> 0.507-0.550; Medium 0.936-0.942 -> 0.254-0.394; Hard 0.889-0.925 -> 0.615-0.653. On EDGE the edge
    rule predicts a change everywhere; at EDGE change rates 0.749 / 0.834 / 0.204 its macro-F1 is r/(1+r) = 0.428 / 0.455
    / 0.169 (computed after the run).
  * Pre-registered verdict: "in between", numbers only, NEITHER sentence written. The first sentence ("reads the block
    structure", EDGE >= 0.95 after masking on Easy and Medium) is not met (Easy 0.507-0.550, Medium 0.254-0.394); on Medium
    this bar could not have been met, since unmasked Medium EDGE is already 0.936-0.942. The second ("depends on the
    strings it saw in training") requires Easy and Medium EDGE to fall to the edge rule's level on those boundaries:
    Medium falls below it (0.254-0.394 vs 0.455), Easy does not (0.507-0.550 vs 0.428). The amended pre-registration is
    ambiguous about this comparator: the script printed the edge rule only over all near-repeated boundaries (0.828 /
    0.901), and under that reading both tiers fall below it and the second sentence would be written. We chose the
    same-subset reading after seeing the results, because the rule says "on those boundaries" and because it does not
    trigger the claim; the verdict depends on this choice.
  * Allowed wording: "On boundaries where exactly one side is a repeated line (12.0% of Easy and 9.8% of Medium
    boundaries), replacing the 42 repeated lines with one neutral placeholder sentence lowers DeSAF's macro-F1 from
    0.990-0.992 to 0.507-0.550 (Easy) and from 0.936-0.942 to 0.254-0.394 (Medium). Over all boundaries touching a
    repeated line it falls from 0.996 to 0.643-0.683 (Easy) and from 0.985-0.986 to 0.505-0.635 (Medium), and the Easy and
    Medium tier scores fall by 0.12 and 0.14-0.19. Part of this is forced by the design: 57 (Easy) and 100 (Medium)
    changes between two different repeated lines become pairs of identical placeholders that no model can detect. The
    pre-registered outcome is 'in between', so we draw no conclusion about whether DeSAF relies on having seen these
    strings." Do NOT set the masked scores against the near-boundary edge rule (0.83 / 0.90) in the paper; that is the
    comparison under which the untriggered sentence would have been written. Hard: near 0.908-0.940 -> 0.545-0.576, tier
    0.819-0.830 -> 0.798-0.808.
  * At boundaries not next to a repeated line (tier-wide; only those in the 575 / 553 / 172 re-scored documents can change,
    so the effect inside the masked documents is diluted and was not measured separately): -0.024 to -0.025 Easy, -0.010
    to -0.019 Medium, -0.002 to -0.004 Hard. These pairs are unchanged, so the shifts come from the changed document
    context (BiLSTM).
  * INSIDE boundaries (both sides repeated): masked scores 0.4838 / 0.4882 / 0.2414 equal an all-no-change prediction
    exactly (57 / 100 / 43 changes); bounded by the design, completeness only.
  * Caveats: one placeholder sentence, unlike real boilerplate and assumed (not checked) to be absent from the training
    text; masking also changes the document context the model reads; thresholds stayed at their unmasked values and no
    ranking measure was reported. The test cannot separate "relies on having seen these strings" from "is thrown by this
    placeholder or by the changed context".
  * The official scores and ranking are unchanged (every system was scored on the same test text, and the 42 lines also
    occur in the shared training text). Whether the winners' Easy and Medium scores are as sensitive to these lines is
    untested (no winner predictions), so "level with the winners" does not show that DeSAF and the winners are level on
    Easy and Medium for the same reasons. Lead the parity claim with Hard.

---

**(P) User's GPU chain pasted 2026-11-17: tier_mechanism.py --skip-part1 (forced-flag ROC-AUC), tier_router.py, the three
router-fed gate_lab heads, tier_router.py --score-only. Log: logs_user/gpu_forcedauc_router.log. VERIFIED 2026-11-17 by
two lenses (numbers + pre-registration, inference); every number recomputes; the wording below is the verifiers'
corrected text. Only the --score-only lines are official-metric numbers; gate_lab's "macro=" lines are positive-class F1.**

- Reproduction. Part 2 retrained the 9 concat_tier heads; their true-flag predictions match the saved files on 100.00% of
  boundaries (as printed, 2 decimals); per-encoder true-flag avgs 0.8822 / 0.8823 / 0.8814. All 27 forced-flag macro-F1
  cells, all 39 accuracy-curve points and the break-even points (0.86 / 0.86 / 0.82) equal (T) digit for digit. The
  diagonal of the forced ROC-AUC matrix equals the concat_tier test ROC-AUC of tier_ranking_check (0.9965 / 0.9331 / 0.9559;
  0.9962 / 0.9394 / 0.9537; 0.9958 / 0.9333 / 0.9532).
- Forced-flag ROC-AUC within the true tier (threshold-free; mean of 3 heads). Wording: "A wrong tier flag does not only
  shift the threshold. In all 18 cells (3 encoders x 6) it lowers within-tier ROC-AUC compared with the same heads given
  the true flag (by 0.006-0.176), to below the AUC of a separately trained head without the flag input (by 0.004-0.158;
  point estimates, no CIs). The loss is largest for Hard documents given the Easy flag (0.778-0.801 vs 0.953-0.956 with the
  true flag and 0.933-0.936 without a flag). How much of the macro-F1 loss is ranking and how much is threshold is not
  separated." Smallest against the true flag: Easy given the Medium flag (0.982-0.990 vs 0.996); against the no-flag head
  the smallest gaps are 0.004-0.012 (in encoder 44 Medium given Easy, 0.008).
- Learned router (tier_router.py): DeBERTa-v3-small, head + tail 512 tokens, trained on the 12,600 PAN 2025 TRAIN documents
  (11,340 fit, 1,260 held out for the epoch choice; epoch 2 chosen, held-out accuracy 0.8452). Validation accuracy 0.8344;
  TEST accuracy 0.8544 (recall Easy 0.884, Medium 0.808, Hard 0.871). 393 test errors; most common Medium read as Hard
  (155, 39%), then Hard read as Medium (83). The docstring's first paragraph says "model chosen by VALIDATION accuracy";
  the code and the run choose the epoch on the held-out TRAIN slice (validation stays out of sample); docstring fixed.
  * Pre-registered prediction (critic): test accuracy 0.88-0.95. FAILED (0.854).
- Router-fed tier heads (the same 9 heads, flag = the router's most likely tier), official metric: 0.9489 / 0.8182 / 0.8215,
  avg 0.8629 (heads 0.8603-0.8653). Pre-registered decision: below 0.8678 -> "a learned router does not recover the gain".
  Wording: "With a learned router (85% test accuracy) in place of the true tier, the tier-aware head scores 0.863 (9 heads,
  0.860-0.865). That is below tier-free DeSAF (0.868; every head is below its encoder's original) and no better than a
  separately trained head without the tier input (0.865; lower at each encoder by 0.001-0.003, not tested)." Against that
  no-flag head per tier: Easy -0.009, Medium +0.001, Hard +0.002. Against the oracle tier head -0.019. Against the earlier
  hard-argmax predicted-tier heads (0.8532; feeding classifier not recorded, 0.762 or 0.853-0.863): +0.010. The earlier
  soft-tier (0.8644) and retrained (0.8641) runs scored slightly above the router-fed heads.
- Router errors vs the accuracy curve. Wording: "Its errors cost more than the accuracy curve's uniformly random errors.
  The router-fed heads (0.860-0.865) fall below the curve even at 82% accuracy (0.866-0.868), although the router is right
  on 85.4% of documents. The kind of confusion does not explain this: weighted by the forced-flag matrix, its confusion
  mix is no costlier than uniform errors, because its many Medium-as-Hard errors are offset by few Hard-as-Easy ones (33
  vs about 66 expected). Which documents it misroutes (for example longer ones carrying more boundaries) is untested. This
  rests on one router and one test set." (Verifier's approximation, not in a log: the router performs like random flags at
  about 76-79% document accuracy.)
- Correction to (T): the tier accuracy needed to keep half the gain over tier-free DeSAF, interpolated on the curve, is
  0.906 / 0.914 / 0.923 by encoder (about 91-92%), not "about 92-95%" (those were the bracketing grid points). Break-even
  with tier-free DeSAF stays about 81-85% (interpolated 0.85 / 0.83 / 0.81); against the no-flag head it is about 78-82%.
- Reading for the paper (finding 3). "The tier flag is worth +0.017 over the same architecture without it (+0.014 over
  tier-free DeSAF) when the tier is known. Under random errors, a tier guess must be right for about 81-85% of documents
  just to match tier-free DeSAF, and about 91-92% to keep half the gain over it. A dedicated learned router
  (DeBERTa-v3-small, 85% test accuracy, no more accurate than our earlier encoder-feature classifiers at 85-86%) did not
  reach tier-free DeSAF (0.863 vs 0.868): its errors cost more than uniformly random errors at that accuracy. ULILLE
  reports 0.9867 router accuracy on PAN 2026 validation; whether that is reachable on PAN 2025 is untested." The paper's
  main system is tier-free DeSAF; the tier-aware head is an oracle analysis.
  NOT allowed: "our best router"; "tier routing does not work" (one router, one dataset); "PAN 2025 tiers cannot be
  identified"; "the router's Medium-as-Hard errors explain the loss"; "the same head given no tier".

---

**(Q) Clue-only and position-only baselines, clue_baselines.py version 2, run by the user 2026-11-17. Log:
logs_user/clue_baselines.log (results/clue_baselines.json on the user's machine). CPU, 550 s. Official metric
(pooled two-class macro-F1) unless stated; thresholds chosen on VALIDATION with the official metric; TEST scored once.**

- History of the rules. Version 1 (written earlier on 2026-11-17, never run on real data) measured gains from predict-0.
  A code review on synthetic data showed predict-0 is below the 0.5 that a guess at the change rate reaches, so a
  no-signal model gained about +0.05 for free, and that v1's position features revealed document length. Version 2
  (gains from 0.5; relative position only; sign vectors printed; Q4r added) was fixed in the docstring before this run.
  A second check of v2 was still running when the log arrived; the rules were not changed after the log was seen.
  Its code lens (finished after the run) found no wrong number and no crash (also run under Python 3.10 / sklearn 1.3).
  Its rule-stress lens was lost in a container restart and was not rerun (the results were already seen).
  Two rule-level caveats from the code lens: (a) the 'relative position' feature takes fine fractions k/(n-1), so it
  still partly encodes document length; on Easy and Medium position alone (AUC 0.633 / 0.618) beats document length
  alone (0.529 / 0.554), but on Hard position (0.586) is below document length (0.614), so the Hard position signal may
  be length. Paper wording: "relative position alone, which partly encodes document length, reaches ROC-AUC 0.59 to
  0.63". (b) gbt_rev signs come from point TRAIN AUCs without CIs; irrelevant here, since all five Hard TRAIN AUCs are
  far from 0.5 (0.258-0.430 on 51,062 boundaries).
- Sanity checks, all pass. Boundaries train 48,402 / 58,817 / 51,062, validation 10,247 / 12,759 / 10,648, test
  10,328 / 12,966 / 11,080 (test change rates 21.67 / 21.57 / 17.25%, as in section 7). predict-0 0.439 / 0.440 / 0.453
  (= (1-p)/(2-p) exactly). Random score through the same threshold search 0.489 / 0.492 / 0.505 (about 0.5). 42 test lines
  occur >= 10 times; boundaries not touching them 79.2 / 73.5 / 95.8% (touching 20.8 / 26.5 / 4.2%, as in the paper).
  DeSAF files reproduce their verified Hard macro-F1: seed43 0.8301, seed44 0.8190, seed45 0.8293.
- Clue AUCs on TRAIN (content, length, function words, trigrams, punctuation): Easy 0.624 / 0.784 / 0.416 / 0.771 / 0.713;
  Medium 0.563 / 0.476 / 0.591 / 0.505 / 0.452; Hard 0.282 / 0.289 / 0.421 / 0.258 / 0.430 (all five reversed on Hard;
  close to the validation values in the paper's Table II and text). gbt_rev signs: Easy [+,+,-,+,+], Medium [+,-,+,+,-],
  Hard [-,-,-,-,-].
- Test macro-F1 Easy / Medium / Hard [95% CI, 2,000 document resamples], and test ROC-AUC Easy / Medium / Hard:
  * random 0.489 / 0.492 / 0.505; AUC Hard 0.504.
  * best single clue (Easy length, Medium function words, Hard trigrams read reversed) 0.713 / 0.535 / 0.638
    [Hard 0.627, 0.648]; AUC 0.779 / 0.577 / 0.739.
  * logistic regression 0.713 / 0.565 / 0.666; AUC 0.839 / 0.625 / 0.800.
  * gbt (five clues, per level) 0.794 / 0.649 / 0.696 [Hard 0.685, 0.706]; AUC 0.913 / 0.761 / 0.843.
  * gbt_rev (each clue monotone in its own TRAIN direction; on Hard all five reversed) 0.740 / 0.585 / 0.677
    [Hard 0.666, 0.686]; AUC 0.863 / 0.670 / 0.815.
  * gbt_cls (all five monotone in the classical direction) 0.717 / 0.549 / 0.497 [Hard 0.488, 0.506]; AUC 0.847 / 0.637 /
    0.515.
  * pooled gbt, no level input, one threshold 0.711 / 0.568 / 0.615 [Hard 0.604, 0.625]; AUC 0.819 / 0.672 / 0.685.
    With per-level thresholds 0.721 / 0.574 / 0.615 (Hard threshold identical, gap 0.000).
  * pooled logistic, one threshold 0.704 / 0.460 / 0.453; Hard AUC 0.412 (below chance).
  * relative position only 0.560 / 0.561 / 0.534; AUC 0.633 / 0.618 / 0.586. Structure (index, index from end,
    relative position, number of sentences) 0.577 / 0.598 / 0.574; AUC 0.659 / 0.677 / 0.641. Number of sentences only
    0.505 / 0.512 / 0.564; AUC 0.529 / 0.554 / 0.614.
  * Hard without repeated-line boundaries: gbt 0.684, gbt_rev 0.672, gbt_cls 0.455, pooled gbt 0.602 (same picture).
- Cross-level transfer (logistic fitted on one level's TRAIN, ROC-AUC on another level's TEST [95% CI, 500 document
  resamples]): Easy -> Hard 0.221 [0.212, 0.232]; Hard -> Easy 0.203 [0.194, 0.213]; Easy -> Medium 0.499; Medium -> Easy
  0.363; Medium -> Hard 0.399; Hard -> Medium 0.507. Own level: 0.839 / 0.625 / 0.800.
- Logistic partial coefficients on Hard TRAIN (standardised): content -0.44 [-0.50, -0.36], length -1.44 [-1.49, -1.39],
  function words +0.08 [+0.04, +0.13], trigrams -0.13 [-0.41, +0.13], punctuation +0.70 [+0.64, +0.77]. With the other
  clues held fixed, function words and punctuation point the usual way on Hard, in line with the length-led pattern in
  the paper (both lose their reversal at matched length). Descriptive only; no rule.
- Paired gaps on Hard (2,000 document resamples): gbt - single +0.058 [+0.048, +0.068]; gbt - pooled gbt (per-level
  threshold) +0.081 [+0.070, +0.092] (one threshold: the same); gbt_rev - gbt_cls +0.180 [+0.166, +0.193]; logistic -
  pooled logistic +0.206. DeSAF - gbt: seed43 +0.134 [+0.121, +0.147], seed44 +0.123 [+0.110, +0.136], seed45 +0.134
  [+0.119, +0.146]. DeSAF - pooled gbt: +0.215 / +0.204 / +0.215 (all CIs exclude 0). ROC-AUC gaps (500 resamples): gbt -
  pooled gbt +0.158 [+0.148, +0.168]; logistic - pooled logistic +0.388 [+0.369, +0.403].
- Fixed-before-running checks (version 2): Q1 holds (+0.058). Q2 holds narrowly: gains over 0.5 on Hard gbt +0.196,
  gbt_rev +0.177 (90.3% of gbt's; the bar was 90%), gbt_cls -0.003 (no gain); all five Hard signs reversed. Q3 holds
  (0.221 and 0.203, upper bounds 0.232 and 0.213). Q4 holds (+0.081). Q4r holds (+0.158). Q5 FAILS: relative position
  alone has test ROC-AUC 0.633 / 0.618 / 0.586 (predicted <= 0.55 on every level).
- Share of the gain over 0.5 on Hard: per-level gbt 0.570 of the true-level head's (0.696 vs 0.844) -> "roughly half to
  two-thirds"; pooled gbt 0.353 of DeSAF's (0.615 vs 0.826) -> "a minority of the gain". Caveat: the clue models'
  thresholds are chosen with the official metric, DeSAF's with positive-class F1, which if anything favours the clue models.
- Allowed wording (rules in the docstring): "combining the five clues adds to the best single clue"; "on Hard the clues
  help only when read in the reversed direction"; "a clue model fitted on Easy ranks Hard boundaries worse than chance,
  and the reverse"; "without the level as an input, the clues lose 0.081 on Hard"; "without the level, the clue model
  ranks Hard boundaries worse, consistent with clues that point opposite ways across levels"; with the level the clues
  give "roughly half to two-thirds" of the true-level head's gain over chance on Hard, without it "a minority" of DeSAF's;
  "DeSAF is 0.12 to 0.13 above the per-level clue model on Hard on every seed (paired CIs exclude 0)".
  NOT allowed: "position in the document alone carries little signal" (Q5 failed; report AUC 0.59-0.63 instead);
  "the clues explain DeSAF" or "DeSAF learns the clues" (not tested); "the reversal causes DeSAF's errors".

---

**Literature read in full, 2026-11-17 (uploaded PDFs; one reader per paper, claims checked against the text with page
numbers).**

- Wegmann, Schraagen and Nguyen (2022), "Same Author or Just Same Topic? Towards Content-Independent Style
  Representations", Proc. 7th Workshop on Representation Learning for NLP (RepL4NLP), pp. 249-268, ACL. Reddit
  authorship verification with siamese RoBERTa; different-author texts drawn from the same conversation, the same
  subreddit, or at random. Citable: style and content are correlated, so representations trained for authorship
  verification can encode content instead of style (p.249); how the different-author pairs are sampled changes what the
  model learns and how well it scores: the model trained on random pairs reaches AUC .79 on random test pairs but .58 on
  conversation-matched ones (Table 2, p.253), and training on conversation-matched pairs gives the most
  content-independent representations (STEL-Or-Content 0.05 -> 0.42, p.255). Do NOT cite it for anything about style
  change detection, PAN, cue reversal, or its own test of function words (it only restates Litvinova 2020, which we
  have not read).
- Mikros and Argiri (2007), "Investigating topic influence in authorship attribution", Proc. SIGIR '07 Workshop on
  Plagiarism Analysis, Authorship Identification, and Near-Duplicate Detection (PAN), Amsterdam. 200 Greek newspaper
  articles, 2 authors x 2 topics. Citable: stylometric variables commonly assumed topic-neutral correlate with topic
  (topic classified at 79.5% from stylometric variables alone, p.4); several of the 10 most frequent function words and
  12 of 32 letter frequencies vary with topic (p.5); "authorship attribution could become a by-product of the
  correlation of authors with specific topics" (p.6); word and sentence length are among the oldest authorship features
  (p.2). Do NOT cite it for character n-grams, punctuation (not measured), sentence length tracking topic (only an
  interaction), or stylometry failing (a topic-neutral subset still gives 93%).
- Koppel, Schler and Argamon (2011), "Authorship attribution in the wild", Language Resources and Evaluation 45:83-94,
  doi:10.1007/s10579-009-9111-2. Citable: character n-gram statistics capture both content and writing style (p.86);
  attribution gets much harder with many candidate authors and short texts (pp.84-89). Do NOT cite it for topic-control
  experiments, function words or how pairs are built. The uploaded paper is the 2011 one; "Koppel et al. (2007)"
  (unmasking) was not read and is not cited.
- Koppel, Schler, Argamon and Winter (2012), "The 'Fundamental Problem' of Authorship Attribution", English Studies
  93(3):284-291, doi:10.1080/0013838X.2012.668794 (read in full by the main session). Citable: the "fundamental problem"
  is "given two (possibly short) documents, determine if they were written by a single author or not" (p.284), a
  building block for other attribution problems; the unmasking principle "two texts are probably by different authors if
  the differences between them are robust to changes in the underlying feature set" (p.285); their impostor method picks
  impostors among the texts most similar to the input "to ensure that the impostors at least roughly resemble" it
  (p.289), with 90% precision and 83% recall on blog pairs. Do NOT cite it for topic confounds or style change detection.
- ICNLP 2027 submission rules (Submission Guide page, pasted by the user 2026-11-17): minimum 4 and maximum 5 printed
  pages including figures without extra page charges; papers over 10 pages are not accepted; official LaTeX template =
  IEEE conference template (IEEEtran V1.8b, conference_101719.tex; "Sub-titles are not captured in Xplore and should not
  be used"); submission via zmeeting.org/submission/icnlp2027; no parallel submission elsewhere during review. Review mode
  (anonymous or not) is not stated on the pasted page; the template has a named author block.

**Literature read in full, 2026-11-17, second batch (PDFs uploaded by the user; one reader per paper, Gururangan read
by the main session). Page numbers are PDF pages.**

- Zangerle, Mayerl, Potthast, Stein (2021), "Overview of the Style Change Detection Task at PAN 2021", CLEF 2021
  Working Notes, CEUR-WS Vol-2936 (volume from the margin stamp; no page range printed). Three subtasks: single vs
  multiple authors, change positions between paragraphs, paragraph-to-author assignment; 16,000 StackExchange documents,
  1 to 4 authors. Citable: documents take all paragraphs from one question/answer thread because "one of the goals for
  this year's edition of the task was to reduce the impact of topic changes within a document, which could
  inadvertently make the task easier" (p.4); best Task 2 F1 0.751 (Zhang et al., fine-tuned BERT). Do NOT cite it for
  difficulty levels (none in 2021), sentence-level changes, or evidence that models exploit topic (it states a design
  motive only).
- Zangerle et al. (2022), "Overview of the Style Change Detection Task at PAN 2022", CLEF 2022 Working Notes, CEUR-WS
  Vol-3180. Three subtasks; Subtask 3 is sentence level. Citable: "artificial documents are synthesized by drawing
  paragraphs from a single question thread to ensure that topic changes cannot be exploited for detecting style changes"
  (p.4); history 2016 (clustering) / 2017 (positions) / 2018 (single vs multi) / 2019 (number of authors) / 2020-2021
  (paragraph level) (p.1). Do NOT cite it for difficulty levels or later editions.
- Zangerle et al. (2023), "Overview of the Multi-Author Writing Style Analysis Task at PAN 2023", CLEF 2023 Working
  Notes, CEUR-WS Vol-3497. Citable: earlier datasets "exhibited a high topical diversity", which "allows participants to
  leverage topic information as a style change signal" (p.1); Reddit, three levels (easy/medium/hard), documents from
  the paragraphs of a single Reddit post, "the paragraphs were then shuffled based on the similarity of these
  properties" (semantic and stylistic), similarities configured "large" for easy, "moderate" for medium, "small" for hard
  (p.3; quote, do not paraphrase the direction); hard: "All paragraphs of a document have the same topic" (p.3); best
  hard F1 0.821 (Ye et al., DeBERTaV3 with contrastive learning); the organisers conclude "topical signals indeed mask
  the detection of authorship style changes" from the score drops (p.5). No artifact analysis.
- Zangerle et al. (2024), "Overview of the Multi-Author Writing Style Analysis Task at PAN 2024", CLEF 2024 Working
  Notes. The PDF prints no volume; CEUR-WS Vol-3740, paper-222 from the CEUR listing (web search, 2026-11-17). Citable:
  "The paragraphs are arranged based on these pair-wise paragraph similarities" (semantic and stylistic) with the same
  large/moderate/small wording (p.2); 6,000 documents per level, 2 to 4 authors, 70/15/15; best hard F1 0.863.
- McCoy, Pavlick, Linzen (2019), "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language
  Inference", Proc. ACL 2019, pp. 3428-3448. Citable: "A machine learning system can score well on a given test set by
  relying on heuristics that are effective for frequent example types but break down in more challenging cases"
  (abstract, p.3428); "test sets drawn from the same distribution as the training set may be inadequate for assessing
  whether a model has learned to perform the intended task" (p.3436). Do NOT cite it for "annotation artifacts" (that is
  Gururangan) or for below-chance accuracy overall.
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith (2018), "Annotation Artifacts in Natural Language Inference
  Data", Proc. NAACL-HLT 2018, pp. 107-112. Citable: a hypothesis-only fastText classifier reaches 67.0% on SNLI and
  53.9 / 52.3% on MultiNLI matched / mismatched, against majority baselines of 34.3 / 35.4 / 35.2% (Table 2, p.108);
  the cues come from annotator strategies (negation -> contradiction, generic words -> entailment, purpose clauses ->
  neutral; hypothesis length) (pp.108-109); "the success of natural language inference models to date has been
  overestimated" (abstract).
- Used in the paper (Related Work), 2026-11-17: PAN history and topic control (2021-2024 overviews), and one sentence
  each for Gururangan (67.0% vs 34.3%) and McCoy (heuristics that fail on harder cases).

**Implementation details used in the paper, confirmed from the code (not from a log), 2026-11-17:** supervised
contrastive loss weight 0.3 and temperature 0.1 (train_deberta_fusion.py defaults, passed explicitly in the run
scripts); function-word distance over 135 function words and punctuation distance over 11 marks
(pan_feature_direction.py FUNCTION_WORDS / PUNCT); constructions use 2 to 6 consecutive sentences per author
(construction_experiment.py), same_topic draws from the 5 sources closest in content, length and style splices draw
from 20 random sources; curriculum = Easy, then Easy + Medium, then all three levels (train_deberta_fusion.py
--curriculum help text: "Easy -> Easy+Medium -> All tiers"), phases 2, 2, 8 epochs.

---

## Still open / not yet done

Added 2026-11-17:
- DONE and verified in (Q): scripts/clue_baselines.py version 2 (clue-only and position-only baselines).
- Literature: all papers cited in the paper are now read in full (the 2007 Koppel item below was never cited).

Added 2026-11-16:
- DONE and verified in (O): hard_error_profile.py, topic_length / topic_rawstyle constructions with construction_profile_check.py,
  boilerplate_mask_test.py.
- DONE and verified in (P): forced-flag ROC-AUC, tier_router.py, the router-fed heads (0.8629: "a learned router does not
  recover the gain").
- ALL EXPERIMENTS ARE COMPLETE. Next: clean the repo (update_repo.ps1), then write the paper from FINDINGS.md and this file.
- Literature: Koppel et al. (2007), Mikros & Argiri (2007), Wegmann et al. (2022) still need the PDFs; cite only if read.

Added 2026-11-09:
- DONE: desaf_on_constructions.py on the Reddit TEST constructions (see (M), provisional).
- DONE: patched tier_ranking_check.py, bootstrap_ci_check.py, length_split_check.py; verified in (N).
- OPTIONAL (permitted by the pre-registration, s <= 0.6): topic_length / topic_rawstyle constructions with the fixed
  criteria C1-C7 from the critic (LT_lean_verdicts.json). Otherwise the experiments are complete; next step is the paper.
- DONE: adversarial check of (T) and (L) and a completeness critic (2026-11-09). Critic's ESSENTIAL items: run
  tier_ranking_check.py (patched version); split PAN 2025 Hard's length effect into absolute length vs length
  difference (length_split_check.py); score the Reddit TEST constructions with DeSAF; document-bootstrap CIs for the
  detector numbers that carry claims (construction AUC drops; PAN 2025 test macro-F1 per seed). OPTIONAL, only if the
  length split leaves the question open: topic_rawstyle / topic_length constructions with pre-registered criteria.
- Paper: write only after every run above is in and verified.

Earlier list (kept as it was):
- Gate vs concat: DONE for 3 seeds, with per-seed significance for all 3 (section 4b).
- Gate lab: DONE. Overlap gating does not help; the tier-aware head is the robust win on all 3
  encoders (section 4c). Rerun feature_direction_by_tier.py with the AUC version for the figure.
- Optional: test whether the tier can be predicted from the document itself (removes the "tier is
  given by PAN" caveat); optional end-to-end tier-aware DeSAF trained on the train set.
- REVISE finding C (pooled overlap trend) in the paper: it is a Simpson's-paradox composition effect (G).
  Cache built 2026-10-31 as `seed43_gate`; it reproduced the seed43 test F1 exactly
  (0.9286/0.7198/0.7198 at threshold 0.50). Gated seed43 VALIDATION F1 at 0.50:
  0.9213/0.7343/0.7116. Topic overlap (TF-IDF fit on train): validation mean 0.0699,
  median 0.0270, 30% of boundaries above 0.075; test mean 0.0693, median 0.0262, 29% above.
- Check whether the Hard ensemble gain (0.7097 -> 0.7350) holds on seeds 44/45; the
  original number is single-seed and the seed noise on Hard is ~0.018.
- Do NOT present the contrastive Hard gain (+0.003) as an improvement; it is far below
  seed noise.
- Primary-source verification of Koppel et al. (2007), Mikros & Argiri (2007), and
  Wegmann et al. (2022); needs you to upload the PDFs.
- `true_nostyle_nocontrastive` has no saved predictions file yet, so style-vs-no-style
  significance can't be tested until `save_test_predictions.py` is run on it.
