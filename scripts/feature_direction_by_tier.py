#!/usr/bin/env python3
"""
Data check for what the tier-aware gates learned: for each tier, does each surface
feature separate author changes from non-changes in the NORMAL direction (more
dissimilar at a change) or REVERSED (more similar at a change)? And within a tier,
does that direction change between low and high topic overlap?

Reads the caches from cache_stage2_features.py (no model, no GPU, seconds).
Effect size = AUC, the probability that a random change boundary is MORE dissimilar than a
random no-change boundary (rank-based, so robust to the skewed feature distributions):
above 0.5 = normal, below 0.5 = reversed. Cohen's d (mean-based) is shown alongside.

Usage:
    python scripts/feature_direction_by_tier.py --name seed43_gate
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

TIERS = ["easy", "medium", "hard"]
CROSSOVER = 0.075


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    p.add_argument("--splits", default="validation,test")
    return p.parse_args()


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def cell(x, y):
    change, same = x[y == 1], x[y == 0]
    if len(change) < 5 or len(same) < 5:
        return "n/a"
    res = stats.mannwhitneyu(change, same, alternative="two-sided")
    auc = res.statistic / (len(change) * len(same))
    d = cohens_d(change, same)
    p = res.pvalue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    direction = "N" if auc > 0.5 else "R"
    return f"{auc:.3f}{stars:3s} {direction} (d{d:+.2f})"


def main():
    args = parse_args()
    parts = {k: [] for k in ["surf", "style", "overlap", "labels", "tier"]}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        z = np.load(Path(args.cache_dir) / f"{args.name}_{split}.npz")
        parts["surf"].append(z["surf"])
        parts["style"].append(np.linalg.norm(z["style"], axis=1))
        parts["overlap"].append(z["overlap"])
        parts["labels"].append(z["labels"])
        parts["tier"].append(np.repeat(z["doc_tier"], z["doc_lengths"]))
    surf = np.concatenate(parts["surf"])
    feats = {
        "fw_dist": surf[:, 0],
        "trigram_dissim": surf[:, 1],
        "punct_dist": surf[:, 2],
        "style_L2 (193)": np.concatenate(parts["style"]),
    }
    overlap = np.concatenate(parts["overlap"])
    y = np.concatenate(parts["labels"])
    tier = np.concatenate(parts["tier"])

    print(f"Splits: {args.splits} | {len(y)} boundaries")
    print("AUC = P(a change boundary is more dissimilar than a no-change one): N = normal (>0.5),")
    print("R = reversed (<0.5); * p<.05, ** p<.01, *** p<.001 (Mann-Whitney); d = Cohen's d\n")
    print(f"{'tier':8s}{'subset':26s}" + "".join(f"{n:>22s}" for n in feats))
    for i, t in enumerate(TIERS):
        for label, m in [
            ("all", tier == i),
            (f"overlap <= {CROSSOVER}", (tier == i) & (overlap <= CROSSOVER)),
            (f"overlap >  {CROSSOVER}", (tier == i) & (overlap > CROSSOVER)),
        ]:
            row = f"{t:8s}{label + f' (n={int(m.sum())})':26s}"
            row += "".join(f"{cell(x[m], y[m]):>22s}" for x in feats.values())
            print(row)
        print()
    hi = overlap > CROSSOVER
    print("Share of boundaries with overlap > %.3f, by tier: " % CROSSOVER
          + ", ".join(f"{t} {hi[tier == i].mean():.2f}" for i, t in enumerate(TIERS)))
    print("Share of Hard among low-overlap / high-overlap boundaries: "
          f"{(tier[~hi] == 2).mean():.2f} / {(tier[hi] == 2).mean():.2f}\n")
    print("Reading it: if a feature's letter is the same in both overlap rows of a tier, its")
    print("direction belongs to the tier, not to topic overlap. If Medium's features point in")
    print("different directions, an overall distance can cancel out on Medium.")


if __name__ == "__main__":
    main()
