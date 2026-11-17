#!/usr/bin/env python3
"""
Is the tier flag's gain a threshold (calibration) effect, or does the flag change the RANKING of boundaries
within a tier?  CPU only, reads files that already exist (seconds); nothing is retrained.

Inputs (written by gate_lab.py / tier_mechanism.py):
  results/gate_lab/{name}/cv_{variant}_s{seed}.npz               validation 5-fold out-of-fold probabilities
  results/gate_lab/{name}/style_only/cv_{variant}_s{seed}.npz    same, [CLS] zeroed (tier_mechanism part 1)
  results/gate_lab/{name}/final_{variant}_s{seed}_test.npz       official-test probabilities of the final heads

For each pair (head without flag, same head with flag):
  as run          official macro-F1 with the thresholds the heads actually used (reproduces tier_mechanism)
  best 1 thr      one threshold for all tiers, chosen IN-SAMPLE to maximise the tier-averaged official macro-F1
  best per tier   one threshold per tier, chosen IN-SAMPLE (uses the true tier, like the flag)
  ROC-AUC         per tier, threshold-free. Validation: computed inside each CV fold (one model) and averaged.
The in-sample thresholds are optimistic for BOTH heads, so their difference is a fair comparison. If per-tier
thresholds give the no-flag head most of the flag's gain, the flag is mainly calibration; if the flag head is
still clearly better at per-tier thresholds and has a higher within-tier ROC-AUC, the flag changes the ranking.
Test: ROC-AUC per tier (threshold-free), plus test-tuned per-tier thresholds labelled as an UPPER BOUND only.

Run from D:\\sass-tad:
    python scripts/tier_ranking_check.py --names seed43_gate,seed44_gate,seed45_gate
"""
import argparse
import json
from pathlib import Path

import numpy as np

TIERS = ["easy", "medium", "hard"]
PAIRS = [("[CLS] only", "", "cls_only", "cls_tier"),
         ("stylometric only", "style_only", "concat", "concat_tier"),
         ("both (paper head)", "", "concat", "concat_tier")]


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def macro_curve(prob, y, thr):
    """Official macro-F1 for every threshold in thr (predict 1 iff prob >= t), vectorised."""
    order = np.argsort(prob)
    ps, ys = prob[order], y[order]
    n, n1 = len(y), int(y.sum()); n0 = n - n1
    pos_above = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])     # positives with index >= i
    i = np.searchsorted(ps, thr, side="left")                        # first index with prob >= t
    k = n - i                                                         # predicted 1
    tp1 = pos_above[i]; fp1 = k - tp1; fn1 = n1 - tp1
    tp0 = n0 - fp1; fp0 = fn1; fn0 = fp1
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp1 + fp1 + fn1 > 0, 2 * tp1 / (2 * tp1 + fp1 + fn1), 0.0)
        f0 = np.where(2 * tp0 + fp0 + fn0 > 0, 2 * tp0 / (2 * tp0 + fp0 + fn0), 0.0)
    return (f1 + f0) / 2


def candidates(prob):
    # every distinct probability: exact, and macro_curve is vectorised, so no grid is needed
    return np.concatenate([np.unique(prob), [np.inf]])


def roc_auc(prob, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(prob)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(path):
    z = np.load(path)
    return {t: (z[f"{t}__probs"].astype(np.float64), z[f"{t}__labels"].astype(np.int64),
                z[f"{t}__preds"].astype(np.int64), z[f"{t}__lengths"].astype(np.int64)) for t in TIERS}


def threshold_stats(d, fb=None):
    """In-sample thresholds. With fb (CV folds per tier) they are chosen INSIDE each fold, i.e. per trained model,
    like the heads' own as-run thresholds; without fb (official test: one model per file) over the whole file."""
    as_run = [macro(d[t][1], d[t][2]) for t in TIERS]
    pf = {t: (np.repeat(fb[t], d[t][3]) if fb is not None else np.zeros(len(d[t][0]), np.int64)) for t in TIERS}
    thr1 = {t: np.empty(len(d[t][0])) for t in TIERS}
    thr_t = {t: np.empty(len(d[t][0])) for t in TIERS}
    for k in np.unique(np.concatenate([pf[t] for t in TIERS])):
        sub = {t: (d[t][0][pf[t] == k], d[t][1][pf[t] == k]) for t in TIERS}
        cand = candidates(np.concatenate([sub[t][0] for t in TIERS]))
        j = int(np.argmax(np.mean([macro_curve(sub[t][0], sub[t][1], cand) for t in TIERS], axis=0)))
        for t in TIERS:
            thr1[t][pf[t] == k] = cand[j]
            c = candidates(sub[t][0])
            thr_t[t][pf[t] == k] = c[int(np.argmax(macro_curve(sub[t][0], sub[t][1], c)))]
    one = [macro(d[t][1], (d[t][0] >= thr1[t]).astype(np.int64)) for t in TIERS]
    per_tier = [macro(d[t][1], (d[t][0] >= thr_t[t]).astype(np.int64)) for t in TIERS]
    return {"as_run": as_run + [float(np.mean(as_run))], "best_one_thr": one + [float(np.mean(one))],
            "best_per_tier": per_tier + [float(np.mean(per_tier))]}


def fold_auc(d, folds_by_tier):
    """Per-tier ROC-AUC computed inside each CV fold (one trained model), averaged over folds."""
    res = []
    for t in TIERS:
        prob, y, _, lengths = d[t]
        pair_fold = np.repeat(folds_by_tier[t], lengths)
        res.append(float(np.nanmean([roc_auc(prob[pair_fold == k], y[pair_fold == k]) for k in np.unique(pair_fold)])))
    return res + [float(np.mean(res))]


def make_folds(doc_tier, k, seed):
    """Copy of gate_lab.make_folds (document folds per tier, same RandomState use)."""
    folds = np.zeros(len(doc_tier), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for t in range(3):
        idx = np.where(doc_tier == t)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % k
    return folds


def folds_by_tier(cache_doc_tier, cache_lengths, seed, n_folds, split_seed, d, path=""):
    """gate_lab's CV folds for head seed `seed`, split by tier in the order save_sig_npz writes documents.
    Verified from the file itself: run_cv predicts each held-out fold with ONE model and ONE threshold shared by
    all tiers, so inside every correctly rebuilt fold the as-run predictions must be separable by one threshold."""
    folds = make_folds(cache_doc_tier, n_folds, split_seed + seed)
    out = {t: folds[cache_doc_tier == i] for i, t in enumerate(TIERS)}
    for i, t in enumerate(TIERS):
        if not np.array_equal(cache_lengths[cache_doc_tier == i], d[t][3]):
            raise ValueError(f"{path} {t}: document lengths differ from the cache (count or order)")
    for k in range(n_folds):
        p = np.concatenate([d[t][0][np.repeat(out[t], d[t][3]) == k] for t in TIERS])
        q = np.concatenate([d[t][2][np.repeat(out[t], d[t][3]) == k] for t in TIERS])
        if (q == 1).any() and (q == 0).any() and p[q == 1].min() <= p[q == 0].max():
            raise ValueError(f"{path}: rebuilt fold {k} is not one model + one threshold; --folds/--split-seed "
                             "do not match the run that wrote this file")
    return out


def f4(v, signed=False):
    return " / ".join((f"{x:+.4f}" if signed else f"{x:.4f}") for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="seed43_gate,seed44_gate,seed45_gate")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--gate-lab-dir", default="results/gate_lab")
    ap.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--out", default="results/tier_ranking_check.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    report = {}
    for name in [n.strip() for n in a.names.split(",") if n.strip()]:
        root = Path(a.gate_lab_dir) / name
        report[name] = {"cv": {}, "test": {}}
        cache = Path(a.cache_dir) / f"{name}_validation.npz"
        cz = np.load(cache) if cache.exists() else None
        cache_doc_tier = cz["doc_tier"].astype(np.int64) if cz is not None else None
        cache_lengths = cz["doc_lengths"].astype(np.int64) if cz is not None else None
        auc_txt = "ROC-AUC per tier (within folds)" if cz is not None else "ROC-AUC per tier (POOLED over folds)"
        thr_txt = "in each fold" if cz is not None else "POOLED over folds"
        print(f"\n{'=' * 110}\n[{name}] VALIDATION 5-fold CV (out-of-fold probabilities), official macro-F1 easy / medium / hard / avg,"
              f" mean over head seeds {seeds}\n{'=' * 110}")
        if cache_doc_tier is None:
            print(f"  NOTE: {cache} not found; ROC-AUC pooled over folds instead of computed within folds")
        for label, sub, v0, v1 in PAIRS:
            stats = {v: [] for v in (v0, v1)}
            aucs = {v: [] for v in (v0, v1)}
            missing = []
            for s in seeds:
                for v in (v0, v1):
                    f = root / sub / f"cv_{v}_s{s}.npz"
                    if not f.exists():
                        missing.append(str(f)); continue
                    d = load(f)
                    if cache_doc_tier is not None:
                        fb = folds_by_tier(cache_doc_tier, cache_lengths, s, a.folds, a.split_seed, d, str(f))
                        stats[v].append(threshold_stats(d, fb))
                        aucs[v].append(fold_auc(d, fb))
                    else:
                        stats[v].append(threshold_stats(d))
                        aucs[v].append([roc_auc(d[t][0], d[t][1]) for t in TIERS])
                        aucs[v][-1].append(float(np.mean(aucs[v][-1])))
            if missing:
                print(f"  {label}: missing {len(missing)} files, e.g. {missing[0]} (skipped)")
                continue
            m = {v: {k: np.mean([x[k] for x in stats[v]], axis=0) for k in stats[v][0]} for v in (v0, v1)}
            A = {v: np.mean(aucs[v], axis=0) for v in (v0, v1)}
            print(f"\n  {label}  ({v0} vs {v1})")
            for k, txt in (("as_run", "as run (thresholds the heads used)"), ("best_one_thr", f"1 threshold {thr_txt}, in-sample"),
                           ("best_per_tier", f"per-tier thr. {thr_txt}, in-sample")):
                print(f"    {txt:38s} no flag {f4(m[v0][k])} | flag {f4(m[v1][k])} | flag gain {f4(m[v1][k] - m[v0][k], True)}")
            cal = m[v0]["best_per_tier"] - m[v0]["best_one_thr"]
            print(f"    {'per-tier thresholds alone (no flag)':38s} {f4(cal, True)}   (per tier - single, no-flag head; "
                  "3 vs 1 in-sample thresholds, slightly optimistic)")
            g1 = m[v1]["best_one_thr"][3] - m[v0]["best_one_thr"][3]
            gt = m[v1]["best_per_tier"][3] - m[v0]["best_per_tier"][3]
            if abs(g1) >= 0.005:
                print(f"    {'':38s} flag gain at per-tier thresholds = {gt / g1:.0%} of the flag gain at one threshold "
                      f"({gt:+.4f} of {g1:+.4f}, same in-sample scheme)")
            else:
                print(f"    {'':38s} flag gain at one in-sample threshold {g1:+.4f}: too small for a share")
            print(f"    {auc_txt:38s} no flag {f4(A[v0])} | flag {f4(A[v1])} | gain {f4(A[v1] - A[v0], True)}")
            report[name]["cv"][label] = {"no_flag": {k: m[v0][k].tolist() for k in m[v0]}, "flag": {k: m[v1][k].tolist() for k in m[v1]},
                                         "auc_no_flag": A[v0].tolist(), "auc_flag": A[v1].tolist(),
                                         "per_tier_thr_gain_no_flag": cal.tolist()}
        # official test, threshold-free
        print(f"\n[{name}] OFFICIAL TEST, final heads (validation-chosen thresholds; nothing tuned on test except the row marked UPPER BOUND)")
        rows = {}
        for v, ss in (("original", [0]), ("concat", seeds), ("concat_tier", seeds)):
            fs = [root / f"final_{v}_s{s}_test.npz" for s in ss]
            fs = [f for f in fs if f.exists()]
            if not fs:
                print(f"  {v}: no final test files found under {root} (skipped)"); continue
            ds = [load(f) for f in fs]
            auc = np.mean([[roc_auc(d[t][0], d[t][1]) for t in TIERS] for d in ds], axis=0)
            st = [threshold_stats(d) for d in ds]
            rows[v] = {"n_files": len(fs), "auc": auc.tolist() + [float(auc.mean())],
                       "as_run": np.mean([x["as_run"] for x in st], axis=0).tolist(),
                       "upper_bound_per_tier_test_thr": np.mean([x["best_per_tier"] for x in st], axis=0).tolist()}
            print(f"  {v:12s} ({len(fs)} file{'s' if len(fs) > 1 else ''})  ROC-AUC {f4(rows[v]['auc'])}")
            print(f"  {'':12s}  as run       {f4(rows[v]['as_run'])}")
            print(f"  {'':12s}  UPPER BOUND, per-tier thresholds tuned on test: {f4(rows[v]['upper_bound_per_tier_test_thr'])}")
        if "concat" in rows and "concat_tier" in rows:
            g = np.array(rows["concat_tier"]["auc"]) - np.array(rows["concat"]["auc"])
            print(f"  flag gain in test ROC-AUC (concat_tier - concat): {f4(g, True)}")
            ub = rows["concat"]["upper_bound_per_tier_test_thr"][3]
            print(f"  no-flag head even with test-tuned per-tier thresholds: {ub:.4f}  vs flag head as run: "
                  f"{rows['concat_tier']['as_run'][3]:.4f}")
        report[name]["test"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
