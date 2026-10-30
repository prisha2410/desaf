#!/usr/bin/env python3
"""
Round 3: does the inverted char-trigram/function-word/punctuation signal
found on Hard-tier TRAIN also hold on a held-out slice of the SAME train
data (doc-level 80/20 split)? And does combining the three features
(logistic regression) beat any one feature alone?

Deliberately does NOT touch the official test set -- using test labels to
decide which features to build an architecture around would be a form of
leakage/overfitting-to-test, even without training weights on it. Official
test stays untouched until a final architecture is locked in.

Usage:
    python hard_tier_probe_v3.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "raw" / "pan2025_full" / "hard" / "train"
VAL_SPLIT_SEED = 42
VAL_FRACTION = 0.2


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


FUNCTION_WORDS = [
    "the","a","an","of","to","in","on","at","for","with","by","from","as",
    "is","was","are","were","be","been","being","have","has","had","do",
    "does","did","will","would","shall","should","may","might","must","can",
    "could","and","but","or","nor","so","yet","if","because","although",
    "while","this","that","these","those","it","its","he","she","they",
    "we","you","i","not","no","very","just","also","only","even","more",
    "most","some","any","all","each","every",
]


def para_stats(p):
    toks = tokenize(p)
    n = max(len(toks), 1)
    fw_vec = np.array([toks.count(w) / n for w in FUNCTION_WORDS], dtype=np.float32)
    trigrams = char_trigrams(p)
    punct = punctuation_profile(p)
    return fw_vec, trigrams, punct


def boundary_features(pa, pb):
    fa, ta, pa_punct = para_stats(pa)
    fb, tb, pb_punct = para_stats(pb)
    fw_dist = float(np.linalg.norm(fa - fb))
    trigram_dissim = 1.0 - cosine_from_counters(ta, tb)
    punct_dist = float(np.linalg.norm(pa_punct - pb_punct))
    return [fw_dist, trigram_dissim, punct_dist]


def load_docs(split_dir):
    if not split_dir.exists():
        return []
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


def main():
    print(f"Loading Hard-tier train docs from {TRAIN_DIR} ...")
    all_docs = load_docs(TRAIN_DIR)
    print(f"  total docs: {len(all_docs)}")

    rng = np.random.RandomState(VAL_SPLIT_SEED)
    idx = rng.permutation(len(all_docs))
    n_val = int(len(all_docs) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_docs = [all_docs[i] for i in train_idx]
    val_docs = [all_docs[i] for i in val_idx]

    X_train, y_train = docs_to_xy(train_docs)
    X_test, y_test = docs_to_xy(val_docs)  # "test" var names kept for min diff below
    print(f"  train docs={len(train_docs)} boundaries={len(y_train)} (pos={y_train.sum()})")
    print(f"  held-out val docs={len(val_docs)} boundaries={len(y_test)} (pos={y_test.sum()})")

    names = ["function_word_dist", "char_trigram_dissim", "punctuation_dist"]

    print("\n=== Individual feature AUC: train vs held-out val (same train dir, doc-level split) ===")
    for j, name in enumerate(names):
        auc_train = roc_auc_score(y_train, X_train[:, j])
        line = f"  {name:<22} train_AUC={auc_train:.3f}"
        if len(y_test):
            auc_test = roc_auc_score(y_test, X_test[:, j])
            line += f"   val_AUC={auc_test:.3f}"
        print(line)

    print("\n=== Combined logistic regression (3 features) ===")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  5-fold CV AUC on train: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    clf.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    print(f"  Train AUC (refit on full train): {train_auc:.3f}")
    print(f"  Coefficients: {dict(zip(names, clf.coef_[0].round(3)))}")

    if len(y_test):
        val_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"  Held-out VAL AUC (doc-level split of train, NOT official test): {val_auc:.3f}")


if __name__ == "__main__":
    main()