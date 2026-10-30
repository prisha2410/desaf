#!/usr/bin/env python3
"""
Probe: do register/rhetorical/discourse features separate true style-change
boundaries from non-boundaries on Hard tier, where topic is held constant?

No model training here -- just per-feature mean(label=1) vs mean(label=0)
and AUC, on the Hard-tier PAN 2025 TRAIN split. Run this before committing to
a full architecture rebuild.

Usage:
    python hard_tier_feature_probe.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent  # this script's own folder
TRAIN_DIR = ROOT / "data" / "raw" / "pan2025_full" / "hard" / "train"

REPORTING_VERBS = {"said", "stated", "announced", "reported", "claimed",
                   "according", "noted", "added", "explained", "confirmed"}
FIRST_PERSON_OPINION = {"i think", "i believe", "in my view", "my read",
                        "imo", "imho", "personally", "honestly"}
HEDGE_INTENSIFY = {"obviously", "clearly", "literally", "basically",
                   "actually", "frankly", "surely", "definitely"}

PRONOUNS_1 = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
PRONOUNS_2 = {"you", "your", "yours"}
PRONOUNS_3 = {"he", "she", "it", "they", "him", "her", "them",
             "his", "hers", "its", "their", "theirs"}


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_split(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def para_features(p):
    toks = tokenize(p)
    n = max(len(toks), 1)
    sents = sentence_split(p)
    n_sents = max(len(sents), 1)

    f = {}
    f["reporting_verb_ratio"] = sum(t in REPORTING_VERBS for t in toks) / n
    joined = " " + p.lower() + " "
    f["first_person_opinion"] = sum(joined.count(" " + phrase + " ")
                                    for phrase in FIRST_PERSON_OPINION)
    f["hedge_intensify_ratio"] = sum(t in HEDGE_INTENSIFY for t in toks) / n
    f["pronoun1_ratio"] = sum(t in PRONOUNS_1 for t in toks) / n
    f["pronoun2_ratio"] = sum(t in PRONOUNS_2 for t in toks) / n
    f["pronoun3_ratio"] = sum(t in PRONOUNS_3 for t in toks) / n
    f["question_ratio"] = p.count("?") / n_sents
    f["exclaim_ratio"] = p.count("!") / n_sents
    f["quote_density"] = (p.count('"') + p.count("“") + p.count("”")) / n
    f["avg_sent_len"] = n / n_sents
    f["avg_word_len"] = sum(len(t) for t in toks) / n
    return f


FEATURE_NAMES = list(para_features("placeholder sentence.").keys())


def boundary_diff(pa, pb):
    fa, fb = para_features(pa), para_features(pb)
    return np.array([abs(fa[k] - fb[k]) for k in FEATURE_NAMES], dtype=np.float32)


def load_hard_train():
    truth_files = {p.stem.replace("truth-problem-", ""): p
                  for p in TRAIN_DIR.glob("truth-problem-*.json")}
    prob_files = {p.stem.replace("problem-", ""): p
                 for p in TRAIN_DIR.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))

    diffs, labels = [], []
    for pid in common_ids:
        paragraphs = [p.strip() for p in
                     prob_files[pid].read_text(encoding="utf-8", errors="replace")
                     .strip().split("\n") if p.strip()]
        changes = json.loads(truth_files[pid].read_text(encoding="utf-8")).get("changes", [])
        n = len(paragraphs) - 1
        if n < 1 or len(changes) != n:
            continue
        for i in range(n):
            diffs.append(boundary_diff(paragraphs[i], paragraphs[i + 1]))
            labels.append(int(changes[i]))
    return np.array(diffs), np.array(labels)


def main():
    print(f"Loading Hard-tier train docs from {TRAIN_DIR} ...")
    X, y = load_hard_train()
    print(f"Boundaries: {len(y)}  (positive={y.sum()}, negative={(y==0).sum()})")
    if len(y) == 0:
        print("No data found -- check TRAIN_DIR path.")
        return

    print(f"\n{'Feature':<22}{'mean(change)':>14}{'mean(no-change)':>18}{'AUC':>8}")
    print("-" * 62)
    results = []
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        m1 = col[y == 1].mean()
        m0 = col[y == 0].mean()
        try:
            auc = roc_auc_score(y, col)
        except ValueError:
            auc = float("nan")
        results.append((name, m1, m0, auc))
        print(f"{name:<22}{m1:>14.4f}{m0:>18.4f}{auc:>8.3f}")

    print("\nSorted by |AUC - 0.5| (discriminative power, either direction):")
    for name, m1, m0, auc in sorted(results, key=lambda r: -abs(r[3] - 0.5)):
        print(f"  {name:<22} AUC={auc:.3f}  (|delta from chance|={abs(auc-0.5):.3f})")


if __name__ == "__main__":
    main()