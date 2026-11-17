#!/usr/bin/env python3
"""
Document-bootstrap 95% CIs for the detector numbers that carry claims. CPU only; reads saved predictions.

1. DeSAF on constructed documents (results/construction_desaf/{encoder}_{set}__{construction}.npz, written by
   desaf_on_constructions.py): mean-over-encoders ROC-AUC per construction, its drop vs 'random', style_splice minus
   same_topic, topic_style minus same_topic, and pooled two-class macro-F1 at each encoder's PAN threshold.
   Documents are resampled independently for each construction (they are different documents) and identically for
   the three encoders (they score the same documents).
2. PAN 2025 official test (checkpoints/*/test_predictions.npz): pooled two-class macro-F1 per tier with a CI for each
   saved run, and for the 3-seed mean of the original DeSAF (same resampled documents for every seed), next to the
   PAN 2025 winners (wqd 0.958 / 0.823 / 0.830; xxsu-team 0.955 / 0.825 / 0.829).

Pre-registered (completeness critic, 2026-11-09): the topic_style and same_topic drop CIs exclude 0; if the
style_splice - same_topic CI includes 0, delete the sentence that a style splice costs less than topic matching;
PAN Medium/Hard CI half-widths 0.010-0.020, wider than every gap to wqd/xxsu, so the paper says "level with".
CIs cover test-sampling noise only, not training randomness.

Run from D:\\sass-tad:
    python scripts/bootstrap_ci_check.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

KINDS = ["random", "same_topic", "content_splice", "length_splice", "style_splice", "topic_style"]
TIERS = ["easy", "medium", "hard"]
WINNERS = {"wqd": [0.958, 0.823, 0.830], "xxsu-team": [0.955, 0.825, 0.829]}


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def auc(prob, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(prob)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def doc_slices(lengths):
    off = np.concatenate([[0], np.cumsum(lengths)])
    return [np.arange(off[i], off[i + 1]) for i in range(len(lengths))]


def resample_index(slices, rng):
    pick = rng.randint(0, len(slices), len(slices))
    return np.concatenate([slices[i] for i in pick])


def ci(v):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def fmt(point, c):
    tag = "excl. 0" if (c[0] > 0 or c[1] < 0) else "incl. 0"
    return f"{point:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] {tag}"


def constructions(a, rng, report):
    encs = [x.split(":") for x in a.encoders.split(",")]
    for s in [x.strip() for x in a.sets.split(",") if x.strip()]:
        boot, point, missing = {}, {}, []
        for k in KINDS:
            files = [Path(a.desaf_dir) / f"{n}_{s}__{k}.npz" for n, _ in encs]
            if not all(f.exists() for f in files):
                missing.append(k); continue
            Z = [np.load(f) for f in files]
            y = Z[0]["labels"].astype(int); L = Z[0]["lengths"].astype(int)
            for z in Z[1:]:
                if not (np.array_equal(z["labels"], Z[0]["labels"]) and np.array_equal(z["lengths"], Z[0]["lengths"])):
                    raise SystemExit(f"{s}/{k}: encoders were not scored on the same documents")
            if L.sum() != len(y):
                raise SystemExit(f"{s}/{k}: lengths do not add up to the number of boundaries")
            P = [z["probs"].astype(np.float64) for z in Z]
            thr = [float(t) for _, t in encs]
            point[k] = {"auc": float(np.mean([auc(p, y) for p in P])),
                        "macro_pan": float(np.mean([macro(y, (p >= t).astype(int)) for p, t in zip(P, thr)])),
                        "auc_per_encoder": [auc(p, y) for p in P], "docs": len(L), "boundaries": int(len(y))}
            sl = doc_slices(L)
            A, M = [], []
            for _ in range(a.boot):
                i = resample_index(sl, rng)
                A.append(np.mean([auc(p[i], y[i]) for p in P]))
                M.append(np.mean([macro(y[i], (p[i] >= t).astype(int)) for p, t in zip(P, thr)]))
            boot[k] = {"auc": np.array(A), "macro_pan": np.array(M)}
        if not point:
            print(f"\n{s}: no scored constructions found in {a.desaf_dir} (skipped)"); continue
        print(f"\n{'=' * 110}\n{s}: DeSAF on constructed documents, mean of {len(encs)} encoders, "
              f"{a.boot} document resamples per construction\n{'=' * 110}")
        if missing:
            print(f"  (not scored yet: {', '.join(missing)})")
        print(f"  {'construction':16s}{'docs':>6s}{'mean AUC':>10s}{'95% CI':>20s}{'macro@PAN':>11s}{'95% CI':>20s}")
        rep = {"point": point, "ci": {}, "differences": {}}
        for k in point:
            ca, cm = ci(boot[k]["auc"]), ci(boot[k]["macro_pan"])
            rep["ci"][k] = {"auc": ca, "macro_pan": cm}
            print(f"  {k:16s}{point[k]['docs']:6d}{point[k]['auc']:10.4f}   [{ca[0]:.4f}, {ca[1]:.4f}]"
                  f"{point[k]['macro_pan']:11.4f}   [{cm[0]:.4f}, {cm[1]:.4f}]")
        pairs = [(k, "random") for k in KINDS if k != "random"] + [("style_splice", "same_topic"),
                                                                   ("topic_style", "same_topic")]
        print("  differences in mean AUC (independent resamples of the two constructions):")
        for k1, k0 in pairs:
            if k1 in point and k0 in point:
                d = point[k1]["auc"] - point[k0]["auc"]
                c = ci(boot[k1]["auc"] - boot[k0]["auc"])
                rep["differences"][f"{k1} - {k0}"] = {"diff": d, "ci": c}
                print(f"    {k1 + ' - ' + k0:30s} {fmt(d, c)}")
        report[s] = rep


def pan_test(a, rng, report):
    runs = {}
    for spec in [x.strip() for x in a.pan_files.split(",") if x.strip()]:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            runs[name] = np.load(path)
    if not runs:
        print("\nPAN 2025 test: no prediction files found (skipped)"); return
    print(f"\n{'=' * 110}\nPAN 2025 official test: pooled two-class macro-F1 per tier, {a.boot} document resamples\n{'=' * 110}")
    rep = {}
    for t in TIERS:
        L = next(iter(runs.values()))[f"{t}__lengths"].astype(int)
        y = next(iter(runs.values()))[f"{t}__labels"].astype(int)
        for n, z in runs.items():
            if not (np.array_equal(z[f"{t}__lengths"], L) and np.array_equal(z[f"{t}__labels"], y)):
                raise SystemExit(f"{n} {t}: different documents or labels from the other runs")
        sl = doc_slices(L)
        P = {n: z[f"{t}__preds"].astype(int) for n, z in runs.items()}
        B = {n: [] for n in P}
        for _ in range(a.boot):
            i = resample_index(sl, rng)
            for n in P:
                B[n].append(macro(y[i], P[n][i]))
        rep[t] = {}
        print(f"\n  {t.upper()}  ({len(L)} documents, {len(y)} boundaries)")
        for n in P:
            pt, c = macro(y, P[n]), ci(B[n])
            rep[t][n] = {"macro": pt, "ci": c, "half_width": (c[1] - c[0]) / 2}
            print(f"    {n:16s} {pt:.4f} [{c[0]:.4f}, {c[1]:.4f}]  half-width {(c[1] - c[0]) / 2:.4f}")
        orig = [n for n in P if n.startswith("gate")]
        if len(orig) > 1:
            mean_pt = float(np.mean([macro(y, P[n]) for n in orig]))
            mean_b = np.mean([B[n] for n in orig], axis=0)
            c = ci(mean_b)
            rep[t]["original_mean"] = {"macro": mean_pt, "ci": c}
            print(f"    {'original mean':16s} {mean_pt:.4f} [{c[0]:.4f}, {c[1]:.4f}]  (same documents for every seed)")
            for w, v in WINNERS.items():
                x = v[TIERS.index(t)]
                inside = c[0] <= x <= c[1]
                print(f"      {w:10s} {x:.3f}: gap {mean_pt - x:+.4f}, {'inside' if inside else 'OUTSIDE'} the CI of the original mean")
                rep[t]["original_mean"][f"vs_{w}"] = {"gap": mean_pt - x, "inside_ci": bool(inside)}
    report["pan2025_test"] = rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desaf-dir", default="results/construction_desaf")
    ap.add_argument("--sets", default="c50,reddit2025test")
    ap.add_argument("--encoders", default="seed43_gate:0.50,seed44_gate:0.52,seed45_gate:0.64",
                    help="name:PAN threshold, as used by desaf_on_constructions.py")
    ap.add_argument("--pan-files", default="gate43=checkpoints/seed43/test_predictions.npz,"
                    "gate44=checkpoints/seed44/test_predictions.npz,gate45=checkpoints/seed45/test_predictions.npz,"
                    "concat43=checkpoints/concat_seed43/test_predictions.npz,"
                    "concat44=checkpoints/concat_seed44/test_predictions.npz,"
                    "concat45=checkpoints/concat_seed45/test_predictions.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bootstrap_ci_check.json")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    report = {}
    constructions(a, rng, report)
    pan_test(a, rng, report)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
