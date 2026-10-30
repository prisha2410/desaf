#!/usr/bin/env python3
"""
C50 overlap-binned diagnostic (refinement of Test 3).

Script 3 (analysis_c50_crosscorpus.py) compared the OVERALL average direction
on C50 against PAN Hard's overall average -- but that's not a fair,
apples-to-apples comparison, because C50's topic control is much broader
than PAN Hard's, and the two corpora's pairs likely sit at very different
points on the topic-overlap spectrum.

This script redoes the comparison the right way: bin ALL C50 pairs (same-
author and different-author together) by their OWN measured topic overlap
(same TF-IDF method as analysis_tier_continuum.py, fit on C50 text only --
deliberately NOT compared to PAN's absolute overlap numbers, since TF-IDF
vocabularies/scales differ across corpora and aren't directly comparable in
raw value; what matters is the TREND within C50 as its own overlap
increases, mirroring the within-tier analysis already done on PAN).

If C50 shows the SAME trend as PAN -- change-boundary dissimilarity falling
as measured overlap rises, even if it doesn't fully cross into inversion
within C50's own overlap range -- that's real supporting evidence the
mechanism is general, just not reaching the same absolute intensity in this
broader-topic corpus. If C50 shows a FLAT or opposite trend, that's real
evidence the mechanism is PAN-specific.

Run from the repo root:
    python scripts/analysis_c50_overlap_binned.py --c50-dir "C:\\path\\to\\reuter+50+50"
"""
import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.stylometric import StylometricExtractor

N_BINS = 8


def split_sentences(text):
    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if s.strip() and len(s.strip().split()) >= 3]
    return sents


def load_c50(c50_dir):
    c50_dir = Path(c50_dir)
    authors = {}
    for split_dir in ["C50train", "C50test"]:
        split_path = c50_dir / split_dir
        if not split_path.exists():
            continue
        for author_dir in sorted(split_path.iterdir()):
            if not author_dir.is_dir():
                continue
            author = author_dir.name
            authors.setdefault(author, [])
            for txt_file in sorted(author_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8", errors="replace")
                sents = split_sentences(text)
                if len(sents) >= 2:
                    authors[author].append(sents)
    return authors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c50-dir", required=True)
    p.add_argument("--n-crossauthor-pairs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    extractor = StylometricExtractor(use_char_ngrams=False)

    print(f"Loading C50 from {args.c50_dir} ...")
    authors = load_c50(args.c50_dir)
    print(f"  {len(authors)} authors, {sum(len(d) for d in authors.values())} documents\n")

    # Collect same-author pairs with their sentence texts (need text for TF-IDF)
    same_author_records = []  # (sent_a, sent_b, stylometric_dist)
    for author, docs in authors.items():
        for sents in docs:
            vecs = [extractor.extract_vector(s) for s in sents]
            for i in range(len(sents) - 1):
                diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
                same_author_records.append((sents[i], sents[i + 1], float(np.linalg.norm(diff))))
    print(f"Same-author pairs: {len(same_author_records)}")

    pool = []
    for author, docs in authors.items():
        for sents in docs:
            for s in sents:
                pool.append((author, s))

    cross_author_records = []
    attempts = 0
    max_attempts = args.n_crossauthor_pairs * 5
    while len(cross_author_records) < args.n_crossauthor_pairs and attempts < max_attempts:
        attempts += 1
        (author_a, sent_a), (author_b, sent_b) = random.sample(pool, 2)
        if author_a == author_b:
            continue
        vec_a = extractor.extract_vector(sent_a)
        vec_b = extractor.extract_vector(sent_b)
        diff = np.abs(np.asarray(vec_b) - np.asarray(vec_a))
        cross_author_records.append((sent_a, sent_b, float(np.linalg.norm(diff))))
    print(f"Cross-author pairs: {len(cross_author_records)}\n")

    # Fit TF-IDF on ALL sentences involved (both pools), C50-only vocabulary
    print("Fitting TF-IDF on C50 sentences...")
    all_texts = []
    for a, b, _ in same_author_records:
        all_texts.extend([a, b])
    for a, b, _ in cross_author_records:
        all_texts.extend([a, b])
    vectorizer = TfidfVectorizer(max_features=20000)
    vectorizer.fit(all_texts)
    print(f"  vocabulary size: {len(vectorizer.vocabulary_)}\n")

    def cosine_sim(a, b):
        va = vectorizer.transform([a])
        vb = vectorizer.transform([b])
        num = (va.multiply(vb)).sum()
        denom = (np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum()))
        return float(num / denom) if denom > 0 else 0.0

    print("Computing topic-overlap scores for all pairs (this is the slow part)...")
    same_overlap = np.array([cosine_sim(a, b) for a, b, _ in same_author_records])
    same_dist = np.array([d for _, _, d in same_author_records])
    cross_overlap = np.array([cosine_sim(a, b) for a, b, _ in cross_author_records])
    cross_dist = np.array([d for _, _, d in cross_author_records])

    print(f"\nSame-author overlap:  mean={same_overlap.mean():.4f}  std={same_overlap.std():.4f}")
    print(f"Cross-author overlap: mean={cross_overlap.mean():.4f}  std={cross_overlap.std():.4f}")

    all_overlap = np.concatenate([same_overlap, cross_overlap])
    all_dist = np.concatenate([same_dist, cross_dist])
    all_label = np.concatenate([np.zeros(len(same_overlap)), np.ones(len(cross_overlap))])

    print("\n" + "=" * 70)
    print("C50 TREND: surface-dissimilarity vs. authorship, by WITHIN-C50 overlap bin")
    print("=" * 70)
    print(f"{'bin':>4} {'overlap range':>18} {'n':>7} {'change_mean':>12} "
          f"{'nochange_mean':>14} {'direction':>20} {'p-value':>10}")

    bin_edges = np.quantile(all_overlap, np.linspace(0, 1, N_BINS + 1))
    trend = []
    for b in range(N_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (all_overlap >= lo) & (all_overlap <= hi if b == N_BINS - 1 else all_overlap < hi)
        change_d = all_dist[mask & (all_label == 1)]
        nochange_d = all_dist[mask & (all_label == 0)]
        if len(change_d) < 5 or len(nochange_d) < 5:
            print(f"{b:>4} {f'{lo:.3f}-{hi:.3f}':>18} {mask.sum():>7}   (too few samples)")
            continue
        direction = "inverted" if change_d.mean() < nochange_d.mean() else "normal"
        _, pval = stats.mannwhitneyu(change_d, nochange_d, alternative="two-sided")
        gap = nochange_d.mean() - change_d.mean()
        print(f"{b:>4} {f'{lo:.3f}-{hi:.3f}':>18} {mask.sum():>7} {change_d.mean():>12.4f} "
              f"{nochange_d.mean():>14.4f} {direction:>20} {pval:>10.4f}")
        trend.append(gap)

    print("\n" + "=" * 70)
    print("TREND SUMMARY")
    print("=" * 70)
    trend = np.array(trend)
    print(f"Gap (nochange_mean - change_mean) across bins, low to high overlap:")
    print(f"  {np.round(trend, 2).tolist()}")
    if len(trend) >= 3:
        slope = np.polyfit(range(len(trend)), trend, 1)[0]
        print(f"\nLinear trend slope = {slope:.3f}")
        if slope < 0:
            print("NEGATIVE slope: as C50's own measured overlap increases, the gap")
            print("shrinks (moving toward / into inversion) -- the SAME direction of")
            print("trend seen on PAN. This supports the mechanism being general, even")
            print("if C50's overall average never fully crosses into inversion because")
            print("most of its pairs sit at lower overlap than PAN Hard's tightly")
            print("controlled single-topic documents.")
        else:
            print("FLAT or POSITIVE slope: the gap does not shrink as overlap rises in")
            print("C50. This does NOT support the mechanism generalizing to this corpus")
            print("-- real evidence leaning toward the construction-artifact explanation.")
    else:
        print("Not enough valid bins to fit a trend reliably.")


if __name__ == "__main__":
    main()
