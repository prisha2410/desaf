#!/usr/bin/env python3
"""
Construction-artifact diagnostic (Test 1 of 3).

Question: is the Hard-tier surface-similarity inversion a real property of
authorship under topic control, or an artifact of PAN's sentence-mixing
algorithm, which actively searches for a *compatible* cross-author sentence
to splice in at each change boundary, while same-author interior sentences
are simply the author's own original, unfiltered text?

Logic: if the inversion is a selection artifact of the splicing step, its
signature should be strongest exactly at the change boundaries themselves
(the actively-filtered junctions), and same-author (no-change) pairs should
look the same regardless of how close they sit to a change boundary --
since interior same-author sentences were never filtered against anything,
proximity to a splice point shouldn't matter for them. If instead same-author
pairs near a splice point look measurably different from same-author pairs
deep inside a stable run, that's a sign the splicing process is leaking into
nearby text too, which would also point toward a construction effect rather
than a pure property of authorship.

Run from the repo root (same place you run train_deberta_fusion.py):
    python scripts/analysis_construction_artifact.py --tier hard
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="hard", choices=["easy", "medium", "hard"])
    p.add_argument("--near-threshold", type=int, default=1,
                    help="no-change boundaries within this many positions of a change "
                         "boundary are 'near'; boundaries at least --far-threshold "
                         "positions away are 'far'")
    p.add_argument("--far-threshold", type=int, default=3)
    p.add_argument("--no-char-ngrams", action="store_true",
                    help="match whatever feature-dim config your main model uses")
    return p.parse_args()


def dist_to_nearest_change(boundary_idx, change_positions):
    if not change_positions:
        return 10_000
    return min(abs(boundary_idx - c) for c in change_positions)


def main():
    args = parse_args()
    extractor = StylometricExtractor(use_char_ngrams=not args.no_char_ngrams)

    print(f"Loading PAN 2025 {args.tier} (train + validation + test -- this is a data")
    print("characterization study, not a model-performance claim, so combining all")
    print("splits for statistical power does not compromise any held-out evaluation.)")
    docs = []
    for split in ["train", "validation", "test"]:
        docs.extend(load_pan2025_full("data/raw", tier=args.tier, split=split))
    print(f"  {len(docs)} documents loaded\n")

    change_dists = []   # ||delta|| at change boundaries
    near_dists = []      # ||delta|| at no-change boundaries near a change
    far_dists = []       # ||delta|| at no-change boundaries far from any change

    for doc_idx, doc in enumerate(docs):
        sents = doc.sentences
        labels = doc.boundary_labels
        n = len(sents) - 1
        if n < 1:
            continue

        vecs = [extractor.extract_vector(s) for s in sents]
        change_positions = [i for i in range(n) if labels[i] == 1]

        for i in range(n):
            diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
            overall_dist = float(np.linalg.norm(diff))

            if labels[i] == 1:
                change_dists.append(overall_dist)
            else:
                d = dist_to_nearest_change(i, change_positions)
                if d <= args.near_threshold:
                    near_dists.append(overall_dist)
                elif d >= args.far_threshold:
                    far_dists.append(overall_dist)
                # boundaries strictly between near/far thresholds are dropped
                # (ambiguous zone, not needed for the comparison)

        if (doc_idx + 1) % 500 == 0:
            print(f"  processed {doc_idx + 1}/{len(docs)} docs")

    change_dists = np.array(change_dists)
    near_dists = np.array(near_dists)
    far_dists = np.array(far_dists)

    print("\n" + "=" * 70)
    print(f"CONSTRUCTION-ARTIFACT DIAGNOSTIC -- {args.tier.upper()} tier")
    print("=" * 70)
    print(f"\nChange boundaries (n={len(change_dists)}):")
    print(f"  mean stylometric distance = {change_dists.mean():.4f}  (std={change_dists.std():.4f})")
    print(f"\nNo-change, NEAR a change boundary (<= {args.near_threshold} away, n={len(near_dists)}):")
    print(f"  mean stylometric distance = {near_dists.mean():.4f}  (std={near_dists.std():.4f})")
    print(f"\nNo-change, FAR from any change boundary (>= {args.far_threshold} away, n={len(far_dists)}):")
    print(f"  mean stylometric distance = {far_dists.mean():.4f}  (std={far_dists.std():.4f})")

    # Headline comparison: change vs all no-change (near+far combined) --
    # this is the original inversion claim, reproduced here for context.
    all_nochange = np.concatenate([near_dists, far_dists])
    u_stat, p_headline = stats.mannwhitneyu(change_dists, all_nochange, alternative="two-sided")
    direction = "MORE similar (inverted)" if change_dists.mean() < all_nochange.mean() else "LESS similar (normal direction)"
    print(f"\n--- Headline check: change vs. all no-change ---")
    print(f"  change mean={change_dists.mean():.4f} vs no-change mean={all_nochange.mean():.4f}")
    print(f"  change boundaries are, on average, {direction}")
    print(f"  Mann-Whitney U p-value = {p_headline:.6f}")

    # The actual diagnostic: near vs far, BOTH same-author (no-change).
    # If these differ significantly, that's a signature of the splicing
    # process leaking into nearby same-author text -- evidence FOR the
    # construction-artifact explanation. If they don't differ, that
    # specific leak mechanism is not supported (though it doesn't fully
    # rule out artifact effects concentrated only at the splice point
    # itself, which this near/far decomposition can't directly isolate
    # further without the actual generator source code).
    u_stat2, p_proximity = stats.mannwhitneyu(near_dists, far_dists, alternative="two-sided")
    print(f"\n--- Diagnostic check: near-boundary vs far-interior, BOTH same-author ---")
    print(f"  near mean={near_dists.mean():.4f} vs far mean={far_dists.mean():.4f}")
    print(f"  Mann-Whitney U p-value = {p_proximity:.6f}")
    if p_proximity < 0.05:
        print("  SIGNIFICANT difference: same-author pairs near a splice point look")
        print("  different from same-author pairs deep in a stable run. This is")
        print("  consistent with (not proof of) the splicing process leaking into")
        print("  nearby text -- a signature that supports the construction-artifact")
        print("  explanation, at least in part.")
    else:
        print("  NOT significant: same-author pairs look the same regardless of")
        print("  proximity to a splice point. This weakens (does not rule out) the")
        print("  specific 'leakage into nearby text' artifact mechanism. It does NOT")
        print("  by itself prove the change-boundary inversion is real, since the")
        print("  filtering could still act narrowly, only at the exact splice point.")

    print("\nHonest limitation: this test cannot fully isolate the artifact question")
    print("without the actual PAN mixing-algorithm source/thresholds, which were not")
    print("reachable. It's the best internal signal available from the data alone.")
    print("Combine with the C50 cross-corpus test (zero PAN filtering at all) for the")
    print("strongest available evidence either way.")


if __name__ == "__main__":
    main()
