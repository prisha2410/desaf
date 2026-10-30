#!/usr/bin/env python3
"""
Broader Hard-tier signal probe, round 2.

Round 1 findings (probe_v1 + artifact check):
  - register/rhetorical/pronoun features: all near-chance (AUC 0.47-0.53)
  - length-diff features: real AUC but flips sign vs Easy tier -> benchmark
    construction artifact, not usable

This round adds features that are harder to accidentally correlate with
benchmark construction:
  - character-trigram distance (captures orthographic/punctuation habits,
    largely topic-invariant)
  - function-word frequency-vector distance (classic stylometry, cheap)
  - type-token ratio (vocabulary richness) diff
  - punctuation-profile distance (comma/period/semicolon/dash rates etc.)
  - GPT-2 per-token surprisal delta (subsampled -- slower, needs `transformers`)

Usage:
    python hard_tier_probe_v2.py            # skip LM surprisal (fast)
    python hard_tier_probe_v2.py --lm        # include GPT-2 surprisal (slower)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "raw" / "pan2025_full" / "hard" / "train"

FUNCTION_WORDS = [
    "the","a","an","of","to","in","on","at","for","with","by","from","as",
    "is","was","are","were","be","been","being","have","has","had","do",
    "does","did","will","would","shall","should","may","might","must","can",
    "could","and","but","or","nor","so","yet","if","because","although",
    "while","this","that","these","those","it","its","he","she","they",
    "we","you","i","not","no","very","just","also","only","even","more",
    "most","some","any","all","each","every",
]


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def char_trigrams(text):
    t = re.sub(r"\s+", " ", text.lower())
    return Counter(t[i:i+3] for i in range(len(t) - 2))


def cosine_from_counters(c1, c2):
    keys = set(c1) | set(c2)
    if not keys:
        return 0.0
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=np.float32)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def punctuation_profile(text):
    n = max(len(text), 1)
    return np.array([
        text.count(",") / n, text.count(".") / n, text.count(";") / n,
        text.count("-") / n, text.count(":") / n, text.count("(") / n,
    ], dtype=np.float32)


def para_stats(p):
    toks = tokenize(p)
    n = max(len(toks), 1)
    fw_vec = np.array([toks.count(w) / n for w in FUNCTION_WORDS], dtype=np.float32)
    ttr = len(set(toks)) / n
    trigrams = char_trigrams(p)
    punct = punctuation_profile(p)
    return {"fw_vec": fw_vec, "ttr": ttr, "trigrams": trigrams, "punct": punct}


def boundary_features(pa, pb):
    sa, sb = para_stats(pa), para_stats(pb)
    fw_dist = float(np.linalg.norm(sa["fw_vec"] - sb["fw_vec"]))
    ttr_diff = abs(sa["ttr"] - sb["ttr"])
    trigram_sim = cosine_from_counters(sa["trigrams"], sb["trigrams"])
    punct_dist = float(np.linalg.norm(sa["punct"] - sb["punct"]))
    return {
        "function_word_dist": fw_dist,
        "ttr_diff": ttr_diff,
        "char_trigram_dissim": 1.0 - trigram_sim,  # higher = more different
        "punctuation_dist": punct_dist,
    }


def load_hard_train():
    truth_files = {p.stem.replace("truth-problem-", ""): p
                  for p in TRAIN_DIR.glob("truth-problem-*.json")}
    prob_files = {p.stem.replace("problem-", ""): p
                 for p in TRAIN_DIR.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))

    pairs, labels = [], []
    for pid in common_ids:
        paragraphs = [p.strip() for p in
                     prob_files[pid].read_text(encoding="utf-8", errors="replace")
                     .strip().split("\n") if p.strip()]
        changes = json.loads(truth_files[pid].read_text(encoding="utf-8")).get("changes", [])
        n = len(paragraphs) - 1
        if n < 1 or len(changes) != n:
            continue
        for i in range(n):
            pairs.append((paragraphs[i], paragraphs[i + 1]))
            labels.append(int(changes[i]))
    return pairs, np.array(labels)


def compute_lm_surprisal_delta(pairs, sample_idx):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("  Loading GPT-2 (small) ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def mean_surprisal(text):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
        if ids.shape[1] < 2:
            return 0.0
        ids = ids.to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        return float(out.loss.item())  # mean cross-entropy = mean surprisal (nats/token)

    deltas = []
    for idx in sample_idx:
        pa, pb = pairs[idx]
        sa, sb = mean_surprisal(pa), mean_surprisal(pb)
        deltas.append(abs(sa - sb))
    return np.array(deltas, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", action="store_true", help="also compute GPT-2 surprisal delta (slower)")
    ap.add_argument("--lm-sample", type=int, default=1500, help="balanced sample size per class for LM probe")
    args = ap.parse_args()

    print(f"Loading Hard-tier train docs from {TRAIN_DIR} ...")
    pairs, y = load_hard_train()
    print(f"Boundaries: {len(y)}  (positive={y.sum()}, negative={(y==0).sum()})")

    print("\nComputing cheap features (char-trigram, function-word, TTR, punctuation)...")
    feats = {k: [] for k in ["function_word_dist", "ttr_diff", "char_trigram_dissim", "punctuation_dist"]}
    for pa, pb in pairs:
        f = boundary_features(pa, pb)
        for k, v in f.items():
            feats[k].append(v)

    print(f"\n{'Feature':<24}{'mean(change)':>14}{'mean(no-change)':>18}{'AUC':>8}")
    print("-" * 64)
    results = []
    for name, vals in feats.items():
        vals = np.array(vals)
        m1, m0 = vals[y == 1].mean(), vals[y == 0].mean()
        auc = roc_auc_score(y, vals)
        results.append((name, m1, m0, auc))
        print(f"{name:<24}{m1:>14.4f}{m0:>18.4f}{auc:>8.3f}")

    if args.lm:
        print(f"\nComputing GPT-2 surprisal delta on a balanced sample (n={args.lm_sample}/class)...")
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng = np.random.RandomState(0)
        pos_sample = rng.choice(pos_idx, size=min(args.lm_sample, len(pos_idx)), replace=False)
        neg_sample = rng.choice(neg_idx, size=min(args.lm_sample, len(neg_idx)), replace=False)
        sample_idx = np.concatenate([pos_sample, neg_sample])
        sample_y = y[sample_idx]

        lm_delta = compute_lm_surprisal_delta(pairs, sample_idx)
        auc_lm = roc_auc_score(sample_y, lm_delta)
        m1 = lm_delta[sample_y == 1].mean()
        m0 = lm_delta[sample_y == 0].mean()
        print(f"\n{'lm_surprisal_delta':<24}{m1:>14.4f}{m0:>18.4f}{auc_lm:>8.3f}  (n={len(sample_y)})")
        results.append(("lm_surprisal_delta", m1, m0, auc_lm))

    print("\nSorted by |AUC - 0.5| (discriminative power, either direction):")
    for name, m1, m0, auc in sorted(results, key=lambda r: -abs(r[3] - 0.5)):
        print(f"  {name:<24} AUC={auc:.3f}  (|delta from chance|={abs(auc-0.5):.3f})")


if __name__ == "__main__":
    main()
