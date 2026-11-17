#!/usr/bin/env python3
"""
Re-score the saved DeSAF-on-C50 predictions (results/c50_multiauthor/<name>_<construction>.npz, written by
c50_multiauthor_test.py) with the official PAN metric (pooled two-class macro-F1), next to both trivial
baselines, under three threshold rules:
  PAN threshold      the checkpoint's own PAN-validation threshold (no C50 information)
  tuned on random    one threshold chosen on the 'random' construction (fine grid, incl. below 0.02),
                     then applied unchanged to every other construction
  oracle             best threshold on the same documents (optimistic; reported only as an upper bound)
ROC-AUC (threshold-free) is printed too. No model, no GPU.

Run from D:\\sass-tad:
    python scripts/c50_rescore_official.py
    python scripts/c50_rescore_official.py --thresholds seed43_gate=0.50,seed44_gate=0.52,seed45_gate=0.64
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_THR = {"43": 0.50, "44": 0.52, "45": 0.64}   # PAN-validation thresholds of the three gated encoders
GRID = np.unique(np.concatenate([np.arange(0.0005, 0.02, 0.0005), np.arange(0.02, 0.905, 0.01)]))


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def roc_auc(probs, y):
    from scipy.stats import rankdata
    r = rankdata(probs)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best(probs, y):
    scores = [macro(y, (probs >= t).astype(int)) for t in GRID]
    i = int(np.argmax(scores))
    return float(GRID[i]), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/c50_multiauthor")
    ap.add_argument("--thresholds", default="", help="name=thr,... (default: by seed number 43/44/45)")
    a = ap.parse_args()
    thr_map = dict(kv.split("=") for kv in a.thresholds.split(",") if "=" in kv)
    files = sorted(Path(a.dir).glob("*.npz"))
    runs = {}
    for f in files:
        m = re.match(r"(.+?)_(random|same_topic|similar_splice|splice_ge_[0-9.]+)$", f.stem)
        if m:
            runs.setdefault(m.group(1), {})[m.group(2)] = f
    if not runs:
        raise SystemExit(f"No <name>_<construction>.npz files in {a.dir}")
    report = {}
    for name, regs in runs.items():
        if name in thr_map:
            pan_thr = float(thr_map[name])
        else:
            seed = re.search(r"4[345]", name)
            pan_thr = DEFAULT_THR.get(seed.group(0)) if seed else None
        if "random" not in regs:
            print(f"{name}: no 'random' construction; skipped")
            continue
        zr = np.load(regs["random"])
        tuned_thr, _ = best(zr["probs"].astype(np.float64), zr["labels"].astype(int))
        print("=" * 112)
        print(f"{name}: PAN threshold {pan_thr if pan_thr is not None else 'unknown'}; threshold tuned on 'random' {tuned_thr:.4f}")
        print("=" * 112)
        print(f"  {'construction':16s}{'chg rate':>9s}{'predict0':>9s}{'predict1':>9s}{'@PAN thr':>10s}"
              f"{'@random thr':>12s}{'oracle (thr)':>17s}{'ROC-AUC':>9s}")
        report[name] = {"pan_threshold": pan_thr, "tuned_threshold": tuned_thr, "constructions": {}}
        order = ["random", "same_topic"] + sorted(r for r in regs if r.startswith("splice_ge_")) + ["similar_splice"]
        for reg in [r for r in order if r in regs]:
            z = np.load(regs[reg])
            pr, y = z["probs"].astype(np.float64), z["labels"].astype(int)
            p = y.mean()
            at_pan = macro(y, (pr >= pan_thr).astype(int)) if pan_thr is not None else float("nan")
            at_tuned = macro(y, (pr >= tuned_thr).astype(int))
            o_thr, o_sc = best(pr, y)
            auc = roc_auc(pr, y)
            print(f"  {reg:16s}{p:9.3f}{(1 - p) / (2 - p):9.3f}{p / (1 + p):9.3f}{at_pan:10.4f}{at_tuned:12.4f}"
                  f"{o_sc:10.4f} ({o_thr:.3f}){auc:9.4f}")
            report[name]["constructions"][reg] = {"change_rate": p, "predict0": (1 - p) / (2 - p),
                                                  "predict1": p / (1 + p), "macro_at_pan_thr": at_pan,
                                                  "macro_at_random_thr": at_tuned, "oracle_macro": o_sc,
                                                  "oracle_thr": o_thr, "roc_auc": auc}
    out = Path(a.dir) / "rescore_official.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
