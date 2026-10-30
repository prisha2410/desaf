#!/usr/bin/env python3
"""
Artifact check: is the avg_sent_len / avg_word_len discriminability on Hard
tier a genuine stylistic signal, or a benchmark-construction quirk?

Checks:
  1. Same AUC pattern computed on Easy and Medium tiers -- if the "smaller
     diff at true changes" effect is equally strong on Easy (where topic
     changes give tons of real signal already), it's more likely a generic
     dataset-construction property, not something specific to hard-tier style.
  2. Raw paragraph length (not diff) at the two paragraphs straddling a
     boundary vs. straddling a non-boundary -- if true-change paragraph pairs
     are just individually more length-matched by construction, that's the
     smoking gun.
  3. Boundary position in document (early/mid/late) vs label -- rules out a
     positional confound.

Usage:
    python length_artifact_check.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw" / "pan2025_full"


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_split(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def para_len_features(p):
    toks = tokenize(p)
    n = max(len(toks), 1)
    sents = sentence_split(p)
    n_sents = max(len(sents), 1)
    avg_sent_len = n / n_sents
    avg_word_len = sum(len(t) for t in toks) / n
    return avg_sent_len, avg_word_len, n  # n = word count


def load_tier(tier, split="train"):
    tier_dir = DATA_ROOT / tier / split
    if not tier_dir.exists():
        return []
    truth_files = {p.stem.replace("truth-problem-", ""): p
                  for p in tier_dir.glob("truth-problem-*.json")}
    prob_files = {p.stem.replace("problem-", ""): p
                 for p in tier_dir.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))

    docs = []
    for pid in common_ids:
        paragraphs = [p.strip() for p in
                     prob_files[pid].read_text(encoding="utf-8", errors="replace")
                     .strip().split("\n") if p.strip()]
        changes = json.loads(truth_files[pid].read_text(encoding="utf-8")).get("changes", [])
        n = len(paragraphs) - 1
        if n < 1 or len(changes) != n:
            continue
        docs.append((paragraphs, changes))
    return docs


def check1_cross_tier_auc(split="train"):
    print(f"\n=== Check 1: same length-diff AUC across tiers ({split}) ===")
    for tier in ["easy", "medium", "hard"]:
        docs = load_tier(tier, split)
        diffs_sent, diffs_word, labels = [], [], []
        for paragraphs, changes in docs:
            for i in range(len(changes)):
                a = para_len_features(paragraphs[i])
                b = para_len_features(paragraphs[i + 1])
                diffs_sent.append(abs(a[0] - b[0]))
                diffs_word.append(abs(a[1] - b[1]))
                labels.append(int(changes[i]))
        if not labels:
            print(f"  {tier}: no data")
            continue
        y = np.array(labels)
        auc_sent = roc_auc_score(y, diffs_sent)
        auc_word = roc_auc_score(y, diffs_word)
        print(f"  {tier:<8} n={len(y):<6} pos_rate={y.mean():.3f}  "
             f"AUC(sent_len_diff)={auc_sent:.3f}  AUC(word_len_diff)={auc_word:.3f}")


def check2_raw_paragraph_length(tier="hard", split="train"):
    print(f"\n=== Check 2: raw paragraph word-count at boundary ({tier}, {split}) ===")
    docs = load_tier(tier, split)
    len_a_change, len_b_change = [], []
    len_a_nochange, len_b_nochange = [], []
    for paragraphs, changes in docs:
        for i in range(len(changes)):
            wa = len(tokenize(paragraphs[i]))
            wb = len(tokenize(paragraphs[i + 1]))
            if changes[i]:
                len_a_change.append(wa); len_b_change.append(wb)
            else:
                len_a_nochange.append(wa); len_b_nochange.append(wb)
    print(f"  True-change boundaries:    para_a mean={np.mean(len_a_change):.1f}  "
         f"para_b mean={np.mean(len_b_change):.1f}  "
         f"abs(a-b) mean={np.mean(np.abs(np.array(len_a_change)-np.array(len_b_change))):.1f}")
    print(f"  Non-change boundaries:     para_a mean={np.mean(len_a_nochange):.1f}  "
         f"para_b mean={np.mean(len_b_nochange):.1f}  "
         f"abs(a-b) mean={np.mean(np.abs(np.array(len_a_nochange)-np.array(len_b_nochange))):.1f}")


def check3_position_confound(tier="hard", split="train"):
    print(f"\n=== Check 3: boundary position in doc vs label ({tier}, {split}) ===")
    docs = load_tier(tier, split)
    pos_change, pos_nochange = [], []
    for paragraphs, changes in docs:
        n = len(changes)
        if n < 2:
            continue
        for i in range(n):
            rel_pos = i / (n - 1) if n > 1 else 0.5
            if changes[i]:
                pos_change.append(rel_pos)
            else:
                pos_nochange.append(rel_pos)
    print(f"  True-change relative position:  mean={np.mean(pos_change):.3f}  "
         f"(0=start of doc, 1=end)")
    print(f"  Non-change relative position:   mean={np.mean(pos_nochange):.3f}")


def main():
    check1_cross_tier_auc("train")
    check2_raw_paragraph_length("hard", "train")
    check3_position_confound("hard", "train")


if __name__ == "__main__":
    main()
