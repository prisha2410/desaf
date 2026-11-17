#!/usr/bin/env python3
"""
Re-score every saved PAN 2025 TEST prediction file with the metric the PAN leaderboard uses.

Our scripts report POSITIVE-CLASS F1 (F1 of the "change" class only). The PAN 2025 leaderboard reports
TWO-CLASS MACRO-F1 (mean of the F1 of "change" and of "no change"). Evidence: the leaderboard's
"predict 0" baseline is 0.439 / 0.440 / 0.453; positive-class F1 of all-zero predictions is 0.

This script needs NO model and NO GPU and changes NO prediction: it only re-scores saved predictions.
It first recomputes the leaderboard's own baseline rows from your test labels. If they come out as
    predict 1: 0.178 / 0.177 / 0.147      predict 0: 0.439 / 0.440 / 0.453
under one of the two macro variants, that variant is the official metric and the test set is the same.

Run from D:\\sass-tad:
    python scripts/rescore_official.py
    python scripts/rescore_official.py path\\to\\one.npz path\\to\\other.npz     (specific files)
It scans checkpoints\\ and results\\ for .npz files with "easy__preds"/"easy__labels"/"easy__lengths" keys
(written by save_test_predictions.py and gate_lab.py) and writes results\\rescore_official.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

TIERS = ["easy", "medium", "hard"]
LEADERBOARD_BASELINES = {"predict 1": (0.178, 0.177, 0.147), "predict 0": (0.439, 0.440, 0.453)}


def class_f1(y, p, c):
    tp = int(np.sum((p == c) & (y == c)))
    fp = int(np.sum((p == c) & (y != c)))
    fn = int(np.sum((p != c) & (y == c)))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0   # sklearn zero_division=0


def macro_pooled(y, p):
    return (class_f1(y, p, 0) + class_f1(y, p, 1)) / 2


def macro_per_doc(y, p, lengths):
    out, s = [], 0
    for n in lengths:
        out.append(macro_pooled(y[s:s + n], p[s:s + n]))
        s += n
    return float(np.mean(out)) if out else float("nan")


def find_files(args):
    if args:
        return [Path(a) for a in args]
    files = []
    for root in ("checkpoints", "results"):
        if Path(root).exists():
            files += [f for f in Path(root).rglob("*.npz") if not f.name.endswith(".partial.npz")]
    keep = []
    for f in sorted(files):
        try:
            with np.load(f, allow_pickle=False) as z:
                if all(f"{t}__{k}" in z.files for t in TIERS for k in ("preds", "labels", "lengths")):
                    keep.append(f)
        except Exception:
            pass
    return keep


def main():
    files = find_files(sys.argv[1:])
    if not files:
        raise SystemExit("No prediction files found (expected keys easy__preds / easy__labels / easy__lengths).")
    print(f"{len(files)} prediction files found.\n")

    # ---- 1. metric check with the leaderboard's own baselines, from the test labels of the first file
    with np.load(files[0], allow_pickle=False) as z:
        lab = {t: z[f"{t}__labels"].astype(int) for t in TIERS}
        lens = {t: z[f"{t}__lengths"].astype(int) for t in TIERS}
    print("METRIC CHECK on the test labels (must match the PAN 2025 leaderboard baseline rows)")
    print(f"  test boundaries: " + ", ".join(f"{t} {len(lab[t])} (change rate {lab[t].mean():.4f}, "
                                            f"{len(lens[t])} docs)" for t in TIERS))
    for name, const in (("predict 1", 1), ("predict 0", 0)):
        pooled = [macro_pooled(lab[t], np.full_like(lab[t], const)) for t in TIERS]
        perdoc = [macro_per_doc(lab[t], np.full_like(lab[t], const), lens[t]) for t in TIERS]
        posf1 = [class_f1(lab[t], np.full_like(lab[t], const), 1) for t in TIERS]
        print(f"  {name}: leaderboard {LEADERBOARD_BASELINES[name]} | pooled macro "
              f"({pooled[0]:.3f}, {pooled[1]:.3f}, {pooled[2]:.3f}) | per-document macro "
              f"({perdoc[0]:.3f}, {perdoc[1]:.3f}, {perdoc[2]:.3f}) | positive-class F1 "
              f"({posf1[0]:.3f}, {posf1[1]:.3f}, {posf1[2]:.3f})")
    print()

    # ---- 2. re-score every file
    rows = []
    hdr = f"{'file':70s} {'metric':16s} {'easy':>7s} {'medium':>7s} {'hard':>7s} {'avg':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            res = {"pos": [], "macro": [], "macro_doc": []}
            same_labels = True
            for t in TIERS:
                y, p, n = z[f"{t}__labels"].astype(int), z[f"{t}__preds"].astype(int), z[f"{t}__lengths"].astype(int)
                same_labels &= (len(y) == len(lab[t]) and np.array_equal(y, lab[t]))
                res["pos"].append(class_f1(y, p, 1))
                res["macro"].append(macro_pooled(y, p))
                res["macro_doc"].append(macro_per_doc(y, p, n))
        name = str(f)
        if len(name) > 70:
            name = "..." + name[-67:]
        for key, label in (("pos", "positive-class"), ("macro", "macro pooled"), ("macro_doc", "macro per-doc")):
            v = res[key]
            print(f"{name if key == 'pos' else '':70s} {label:16s} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} {np.mean(v):7.4f}")
            rows.append([str(f), label] + [f"{x:.4f}" for x in v] + [f"{np.mean(v):.4f}", int(same_labels)])
        if not same_labels:
            print(f"{'':70s} !! labels differ from the first file: not the same test set or order")
    out = Path("results") / "rescore_official.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "metric", "easy", "medium", "hard", "avg", "same_test_labels"])
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
