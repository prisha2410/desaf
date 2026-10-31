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

---

## Still open / not yet done

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
