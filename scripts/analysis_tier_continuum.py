#!/usr/bin/env python3
"""
Continuous topic-overlap diagnostic (Test 2 of 3).

Question: instead of trusting PAN's own discrete Easy/Medium/Hard labels
(which are themselves a product of the construction process under
investigation), compute an independent, continuous topic-overlap score for
every boundary across ALL THREE tiers, and see how the surface-similarity-
to-authorship relationship behaves as a smooth function of that score.

This directly tests the Medium-tier hypothesis: is Medium a clean midpoint
on a smooth interpolation between Easy's normal direction and Hard's
inverted direction, or does it sit in a genuinely unstable, high-variance
zone that explains why no technique has been able to improve it?

Topic overlap is measured with TF-IDF cosine similarity between the two
sentences -- cheap, CPU-only, no GPU, no new model downloads, and
deliberately independent of PAN's own tier construction.

Run from the repo root:
    python scripts/analysis_tier_continuum.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor


N_BINS = 10


def main():
    extractor = StylometricExtractor(use_char_ngrams=False)

    print("Loading PAN 2025, all three tiers, all splits (data characterization,")
    print("not a model-performance claim -- combining splits is fine here).\n")

    all_docs = {}
    for tier in ["easy", "medium", "hard"]:
        docs = []
        for split in ["train", "validation", "test"]:
            docs.extend(load_pan2025_full("data/raw", tier=tier, split=split))
        all_docs[tier] = docs
        print(f"  {tier}: {len(docs)} documents")

    # Fit TF-IDF once on every sentence across all tiers, so similarity
    # scores are on a comparable scale everywhere.
    print("\nFitting TF-IDF vectorizer on all sentences (this may take a minute)...")
    all_sentences = []
    for tier_docs in all_docs.values():
        for doc in tier_docs:
            all_sentences.extend(doc.sentences)
    vectorizer = TfidfVectorizer(max_features=20000, stop_words=None)
    vectorizer.fit(all_sentences)
    print(f"  vocabulary size: {len(vectorizer.vocabulary_)}\n")

    def cosine_sim(a, b):
        va = vectorizer.transform([a])
        vb = vectorizer.transform([b])
        num = (va.multiply(vb)).sum()
        denom = (np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum()))
        return float(num / denom) if denom > 0 else 0.0

    records = []  # (tier, topic_overlap, stylometric_dist, label)
    for tier, docs in all_docs.items():
        print(f"Processing {tier}...")
        for doc_idx, doc in enumerate(docs):
            sents = doc.sentences
            labels = doc.boundary_labels
            n = len(sents) - 1
            if n < 1:
                continue
            vecs = [extractor.extract_vector(s) for s in sents]
            for i in range(n):
                overlap = cosine_sim(sents[i], sents[i + 1])
                diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
                dist = float(np.linalg.norm(diff))
                records.append((tier, overlap, dist, labels[i]))
            if (doc_idx + 1) % 500 == 0:
                print(f"  {doc_idx + 1}/{len(docs)} docs")

    records = np.array(records, dtype=object)
    tiers = records[:, 0]
    overlap = records[:, 1].astype(float)
    dist = records[:, 2].astype(float)
    label = records[:, 3].astype(int)

    print("\n" + "=" * 70)
    print("SANITY CHECK: does self-measured topic overlap track PAN's own tiers?")
    print("=" * 70)
    for tier in ["easy", "medium", "hard"]:
        mask = tiers == tier
        print(f"  {tier:6s}: mean measured overlap = {overlap[mask].mean():.4f}  "
              f"(std={overlap[mask].std():.4f})")

    print("\n" + "=" * 70)
    print("MAIN RESULT: surface-similarity-vs-authorship direction, by overlap decile")
    print("=" * 70)
    print(f"{'bin':>4} {'overlap range':>18} {'n':>7} {'change_mean':>12} "
          f"{'nochange_mean':>14} {'direction':>20} {'p-value':>10}")

    bin_edges = np.quantile(overlap, np.linspace(0, 1, N_BINS + 1))
    bin_results = []
    for b in range(N_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == N_BINS - 1:
            mask = (overlap >= lo) & (overlap <= hi)
        else:
            mask = (overlap >= lo) & (overlap < hi)

        change_d = dist[mask & (label == 1)]
        nochange_d = dist[mask & (label == 0)]
        if len(change_d) < 5 or len(nochange_d) < 5:
            print(f"{b:>4} {f'{lo:.3f}-{hi:.3f}':>18} {mask.sum():>7}   (too few samples, skipped)")
            continue

        direction = "inverted" if change_d.mean() < nochange_d.mean() else "normal"
        _, p = stats.mannwhitneyu(change_d, nochange_d, alternative="two-sided")
        print(f"{b:>4} {f'{lo:.3f}-{hi:.3f}':>18} {mask.sum():>7} {change_d.mean():>12.4f} "
              f"{nochange_d.mean():>14.4f} {direction:>20} {p:>10.4f}")
        bin_results.append({
            "bin": b, "overlap_lo": lo, "overlap_hi": hi,
            "change_mean": change_d.mean(), "nochange_mean": nochange_d.mean(),
            "direction": direction, "p_value": p,
            "effect_size": nochange_d.mean() - change_d.mean(),
        })

    print("\n" + "=" * 70)
    print("MEDIUM-TIER INSTABILITY CHECK")
    print("=" * 70)
    print("If Medium sits in a genuinely unstable zone (not a clean midpoint),")
    print("its within-tier spread of (nochange_mean - change_mean) across overlap")
    print("bins should be larger / less monotonic than Easy's or Hard's.\n")

    effect_sizes_by_tier = {"easy": [], "medium": [], "hard": []}
    for tier in ["easy", "medium", "hard"]:
        mask = tiers == tier
        tier_overlap = overlap[mask]
        tier_dist = dist[mask]
        tier_label = label[mask]
        tbin_edges = np.quantile(tier_overlap, np.linspace(0, 1, 6))  # 5 bins, tier-local
        for b in range(5):
            lo, hi = tbin_edges[b], tbin_edges[b + 1]
            bmask = (tier_overlap >= lo) & (tier_overlap <= hi if b == 4 else tier_overlap < hi)
            cd = tier_dist[bmask & (tier_label == 1)]
            nd = tier_dist[bmask & (tier_label == 0)]
            if len(cd) >= 5 and len(nd) >= 5:
                effect_sizes_by_tier[tier].append(nd.mean() - cd.mean())

    for tier, effects in effect_sizes_by_tier.items():
        effects = np.array(effects)
        print(f"  {tier:6s}: effect sizes across 5 internal overlap-bins = "
              f"{np.round(effects, 4).tolist()}")
        print(f"           mean={effects.mean():.4f}  std={effects.std():.4f}  "
              f"(higher std = less consistent direction within this tier)")

    print("\nInterpretation: compare the 'std' values above. If Medium's std is")
    print("clearly larger than both Easy's and Hard's, that supports the hypothesis")
    print("that Medium's signal is genuinely inconsistent rather than just weaker --")
    print("a real mechanistic reason no single fusion technique has moved it.")


if __name__ == "__main__":
    main()
