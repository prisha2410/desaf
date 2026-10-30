#!/usr/bin/env python3
"""
Cross-corpus validation on C50 / Reuters-50-50 (Test 3 of 3).

Question: does the surface-similarity-to-authorship inversion show up in
completely different, naturally-written text -- real news articles by 50
named authors, all restricted to one broad Reuters topic category, with
ZERO sentence-splicing or selection filtering of any kind?

This is the decisive test for the construction-artifact question. PAN's
Hard tier is a synthetic document assembled by an algorithm that actively
filters which cross-author sentence gets spliced in. C50 has no such
process: every document is simply what one author actually wrote. If the
inversion still appears here, it is strong evidence the phenomenon is real,
not a PAN-specific artifact. If it does not appear (or the normal,
non-inverted direction holds instead), that is real evidence PAN's active
filtering step is what produces or inflates the effect.

Design:
  - "no-change" pairs: two CONSECUTIVE sentences from the SAME real article
    by the SAME author (natural, unfiltered, no splicing at all).
  - "change" pairs: one sentence randomly drawn from author A's article and
    one sentence randomly drawn from a DIFFERENT author B's article, both
    within the same corpus (so topic is roughly constant across the whole
    comparison, since all of C50 is pre-restricted to one Reuters category)
    -- picked with NO compatibility filtering whatsoever, i.e. the opposite
    of PAN's mixing algorithm.

Run from the repo root, pointing at wherever you extracted the C50 zip:
    python scripts/analysis_c50_crosscorpus.py --c50-dir /path/to/reuter5050_raw
"""
import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.stylometric import StylometricExtractor


def split_sentences(text):
    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if s.strip() and len(s.strip().split()) >= 3]
    return sents


def load_c50(c50_dir):
    """Returns {author_name: [list of sentence-lists, one per document]}."""
    c50_dir = Path(c50_dir)
    authors = {}
    for split_dir in ["C50train", "C50test"]:
        split_path = c50_dir / split_dir
        if not split_path.exists():
            print(f"  WARNING: {split_path} not found, skipping")
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
    p.add_argument("--c50-dir", required=True,
                    help="Directory containing C50train/ and C50test/ subfolders")
    p.add_argument("--n-crossauthor-pairs", type=int, default=20000,
                    help="Number of random cross-author pairs to sample")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    extractor = StylometricExtractor(use_char_ngrams=False)

    print(f"Loading C50 from {args.c50_dir} ...")
    authors = load_c50(args.c50_dir)
    author_names = sorted(authors.keys())
    print(f"  {len(author_names)} authors, "
          f"{sum(len(docs) for docs in authors.values())} documents total\n")

    if len(author_names) < 2:
        print("ERROR: need at least 2 authors with usable documents. Check --c50-dir.")
        sys.exit(1)

    # --- Same-author (no-change), natural consecutive sentence pairs -----
    print("Computing same-author (natural, unfiltered) sentence-pair distances...")
    same_author_dists = []
    for author, docs in authors.items():
        for sents in docs:
            vecs = [extractor.extract_vector(s) for s in sents]
            for i in range(len(sents) - 1):
                diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
                same_author_dists.append(float(np.linalg.norm(diff)))
    same_author_dists = np.array(same_author_dists)
    print(f"  {len(same_author_dists)} same-author pairs\n")

    # --- Different-author (change proxy), fully random, zero filtering ---
    print(f"Sampling {args.n_crossauthor_pairs} random cross-author sentence pairs "
          "(no compatibility filtering at all)...")
    # Pre-flatten (author, sentence) pool for fast random sampling
    pool = []
    for author, docs in authors.items():
        for sents in docs:
            for s in sents:
                pool.append((author, s))
    print(f"  sentence pool size: {len(pool)}")

    cross_author_dists = []
    attempts = 0
    max_attempts = args.n_crossauthor_pairs * 5
    while len(cross_author_dists) < args.n_crossauthor_pairs and attempts < max_attempts:
        attempts += 1
        (author_a, sent_a), (author_b, sent_b) = random.sample(pool, 2)
        if author_a == author_b:
            continue  # resample -- we want genuinely different authors
        vec_a = extractor.extract_vector(sent_a)
        vec_b = extractor.extract_vector(sent_b)
        diff = np.abs(np.asarray(vec_b) - np.asarray(vec_a))
        cross_author_dists.append(float(np.linalg.norm(diff)))
    cross_author_dists = np.array(cross_author_dists)
    print(f"  {len(cross_author_dists)} cross-author pairs collected\n")

    print("=" * 70)
    print("RESULT: C50 cross-corpus validation (zero splicing, zero filtering)")
    print("=" * 70)
    print(f"Same-author (natural, n={len(same_author_dists)}):")
    print(f"  mean distance = {same_author_dists.mean():.4f}  (std={same_author_dists.std():.4f})")
    print(f"Different-author (random, unfiltered, n={len(cross_author_dists)}):")
    print(f"  mean distance = {cross_author_dists.mean():.4f}  (std={cross_author_dists.std():.4f})")

    u_stat, p_value = stats.mannwhitneyu(
        same_author_dists, cross_author_dists, alternative="two-sided"
    )
    print(f"\nMann-Whitney U p-value = {p_value:.6f}")

    if same_author_dists.mean() < cross_author_dists.mean():
        print("\nDIRECTION: NORMAL -- same-author pairs are MORE similar than")
        print("different-author pairs, as classical stylometry would predict.")
        print("This is evidence the Hard-tier INVERSION does NOT replicate in")
        print("naturally-written, unfiltered text -- supporting the construction-")
        print("artifact explanation (the inversion may be specific to PAN's active")
        print("splice-selection process, not a general property of authorship).")
    else:
        print("\nDIRECTION: INVERTED -- different-author pairs are MORE similar than")
        print("same-author pairs, matching the Hard-tier pattern, even with ZERO")
        print("selection filtering of any kind. This is real, meaningful evidence")
        print("the phenomenon is NOT purely a PAN construction artifact -- it shows")
        print("up in independently-constructed, naturally-written text too.")

    print("\nHonest caveats:")
    print("- C50's topic control is broad (one Reuters category across the whole")
    print("  corpus), not as tight as PAN's single-document single-topic Hard tier.")
    print("  A null or weak result here doesn't fully rule out the phenomenon at")
    print("  tighter topic control; a positive result here is the stronger finding.")
    print("- Sentence splitting here is a simple regex, not the same pipeline PAN")
    print("  uses -- minor boundary artifacts possible, shouldn't change the")
    print("  direction of a real effect.")


if __name__ == "__main__":
    main()
