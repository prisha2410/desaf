#!/usr/bin/env python3
"""
Round 4: push the ceiling on the 3 winning features (function_word_dist,
char_trigram_dissim, punctuation_dist) plus a few cheap additions, using
gradient boosting. Trains on hard/train, evaluates on hard/validation
(your actual designated validation split) -- official test stays untouched.

Requires: pip install lightgbm  (falls back to sklearn GradientBoosting if
lightgbm isn't installed)

Usage:
    python hard_tier_probe_v4.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "raw" / "pan2025_full" / "hard" / "train"
VAL_DIR = ROOT / "data" / "raw" / "pan2025_full" / "hard" / "validation"

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


def sentence_split(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


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
    sents = sentence_split(p)
    n_sents = max(len(sents), 1)
    fw_vec = np.array([toks.count(w) / n for w in FUNCTION_WORDS], dtype=np.float32)
    trigrams = char_trigrams(p)
    punct = punctuation_profile(p)
    ttr = len(set(toks)) / n
    avg_sent_len = n / n_sents
    return {
        "fw_vec": fw_vec, "trigrams": trigrams, "punct": punct,
        "ttr": ttr, "avg_sent_len": avg_sent_len, "n_words": n, "n_sents": n_sents,
    }


def boundary_features(pa, pb):
    sa, sb = para_stats(pa), para_stats(pb)
    fw_dist = float(np.linalg.norm(sa["fw_vec"] - sb["fw_vec"]))
    trigram_dissim = 1.0 - cosine_from_counters(sa["trigrams"], sb["trigrams"])
    punct_dist = float(np.linalg.norm(sa["punct"] - sb["punct"]))
    ttr_diff = abs(sa["ttr"] - sb["ttr"])
    sent_len_diff = abs(sa["avg_sent_len"] - sb["avg_sent_len"])
    word_count_diff = abs(sa["n_words"] - sb["n_words"])
    sent_count_diff = abs(sa["n_sents"] - sb["n_sents"])
    return [fw_dist, trigram_dissim, punct_dist, ttr_diff,
           sent_len_diff, word_count_diff, sent_count_diff]


FEATURE_NAMES = ["function_word_dist", "char_trigram_dissim", "punctuation_dist",
                 "ttr_diff", "sent_len_diff", "word_count_diff", "sent_count_diff"]


def load_docs(split_dir):
    truth_files = {p.stem.replace("truth-problem-", ""): p
                  for p in split_dir.glob("truth-problem-*.json")}
    prob_files = {p.stem.replace("problem-", ""): p
                 for p in split_dir.glob("problem-*.txt")}
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


def docs_to_xy(docs):
    X, y = [], []
    for paragraphs, changes in docs:
        for i in range(len(changes)):
            X.append(boundary_features(paragraphs[i], paragraphs[i + 1]))
            y.append(int(changes[i]))
    return np.array(X, dtype=np.float32), np.array(y)


def get_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                              class_weight="balanced", verbosity=-1), "LightGBM"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                          learning_rate=0.05), "sklearn GBM"


def main():
    print(f"Loading Hard-tier train docs from {TRAIN_DIR} ...")
    train_docs = load_docs(TRAIN_DIR)
    print(f"  train docs: {len(train_docs)}")

    print(f"Loading Hard-tier validation docs from {VAL_DIR} ...")
    val_docs = load_docs(VAL_DIR)
    print(f"  validation docs: {len(val_docs)}")

    print("Extracting features (7-dim) ...")
    X_train, y_train = docs_to_xy(train_docs)
    X_val, y_val = docs_to_xy(val_docs)
    print(f"  train boundaries={len(y_train)}  val boundaries={len(y_val)}")

    clf, clf_name = get_classifier()
    print(f"\nUsing classifier: {clf_name}")

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"5-fold CV AUC on train: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    clf.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Held-out VAL AUC: {val_auc:.3f}")

    if hasattr(clf, "feature_importances_"):
        print("\nFeature importances:")
        for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                                key=lambda x: -x[1]):
            print(f"  {name:<22} {imp:.4f}")

    from sklearn.metrics import precision_recall_curve
    print("\n=== F1 at best threshold (held-out val) ===")
    val_probs = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = np.argmax(f1s[:-1])  # last point has no threshold
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    print(f"  Best threshold: {best_thresh:.3f}")
    print(f"  F1 at best threshold: {best_f1:.4f}")
    print(f"  Precision: {precisions[best_idx]:.4f}  Recall: {recalls[best_idx]:.4f}")

    print("\n  For comparison (from your paper):")
    print("    DeSAF Hard F1 (full model):          0.7252")
    print("    DeSAF Hard F1 (specialist routing):  0.7644")
    print("    Top leaderboard Hard F1:              0.830")


if __name__ == "__main__":
    main()