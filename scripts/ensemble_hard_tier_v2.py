#!/usr/bin/env python3
"""
Ensemble v2: upgrades over ensemble_hard_tier.py --
  1. Richer feature set for LightGBM (adds question/exclaim/quote-density
     diffs back in -- weak alone, may help in combination).
  2. Stacker sees more than 2 inputs: [p_deberta, p_lgbm] PLUS the 3
     strongest raw features (char_trigram_dissim, function_word_dist,
     punctuation_dist), so it can learn cases where LightGBM's own
     confidence under/overstates things.
  3. Stacker itself is a small LightGBM (not logistic regression) --
     captures non-linear interactions between DeBERTa/LightGBM confidence.

Still PAN 2025_full ONLY, real splits, ONE final test evaluation.

Usage:
    python scripts/ensemble_hard_tier_v2.py --checkpoint checkpoints/curriculum_228_2025only_193dim.pt
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset

from train_deberta_fusion import encode_document, build_style_diffs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Part A: DeBERTa+BiLSTM probabilities (unchanged from v1)
# --------------------------------------------------------------------------- #

class EvalArgs:
    def __init__(self, max_length=256, sub_batch=8, no_style=False):
        self.max_length = max_length
        self.sub_batch = sub_batch
        self.no_style = no_style


def load_established_model(checkpoint_path):
    extractor = StylometricExtractor(use_char_ngrams=False)
    style_dim = extractor.feature_dim
    assert style_dim == 193, f"Expected 193-dim extractor, got {style_dim}"

    model = DeBERTaFusionE2E(
        style_dim=style_dim, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, extractor


def deberta_probs_for_split(model, extractor, tokenizer, docs):
    sents_list = [d.sentences for d in docs]
    labels_list = [d.boundary_labels for d in docs]
    doc_ids = [d.doc_id for d in docs]
    style_diffs = build_style_diffs(docs, extractor)
    ds = DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

    eval_args = EvalArgs()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            doc = ds[idx]
            fused = encode_document(model, doc, tokenizer, DEVICE, True,
                                    eval_args.max_length, eval_args.sub_batch)
            from torch.amp import autocast
            with autocast("cuda", dtype=torch.float16):
                logits = model.forward_document(fused.unsqueeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(doc["labels"])
    return np.array(all_probs), np.array(all_labels, dtype=int)


# --------------------------------------------------------------------------- #
#  Part B: EXPANDED feature set (7 original + 3 round-1 weak features)
# --------------------------------------------------------------------------- #

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
    question_ratio = p.count("?") / n_sents
    exclaim_ratio = p.count("!") / n_sents
    quote_density = (p.count('"') + p.count(chr(8220)) + p.count(chr(8221))) / n
    return {
        "fw_vec": fw_vec, "trigrams": trigrams, "punct": punct,
        "ttr": ttr, "avg_sent_len": avg_sent_len, "n_words": n, "n_sents": n_sents,
        "question_ratio": question_ratio, "exclaim_ratio": exclaim_ratio,
        "quote_density": quote_density,
    }


FEATURE_NAMES = [
    "function_word_dist", "char_trigram_dissim", "punctuation_dist", "ttr_diff",
    "sent_len_diff", "word_count_diff", "sent_count_diff",
    "question_ratio_diff", "exclaim_ratio_diff", "quote_density_diff",
]


def boundary_features(pa, pb):
    sa, sb = para_stats(pa), para_stats(pb)
    fw_dist = float(np.linalg.norm(sa["fw_vec"] - sb["fw_vec"]))
    trigram_dissim = 1.0 - cosine_from_counters(sa["trigrams"], sb["trigrams"])
    punct_dist = float(np.linalg.norm(sa["punct"] - sb["punct"]))
    ttr_diff = abs(sa["ttr"] - sb["ttr"])
    sent_len_diff = abs(sa["avg_sent_len"] - sb["avg_sent_len"])
    word_count_diff = abs(sa["n_words"] - sb["n_words"])
    sent_count_diff = abs(sa["n_sents"] - sb["n_sents"])
    question_ratio_diff = abs(sa["question_ratio"] - sb["question_ratio"])
    exclaim_ratio_diff = abs(sa["exclaim_ratio"] - sb["exclaim_ratio"])
    quote_density_diff = abs(sa["quote_density"] - sb["quote_density"])
    return [fw_dist, trigram_dissim, punct_dist, ttr_diff,
           sent_len_diff, word_count_diff, sent_count_diff,
           question_ratio_diff, exclaim_ratio_diff, quote_density_diff]


def lgbm_features_for_docs(docs):
    X, y = [], []
    for d in docs:
        n = len(d.boundary_labels)
        for i in range(n):
            X.append(boundary_features(d.sentences[i], d.sentences[i + 1]))
            y.append(int(d.boundary_labels[i]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/curriculum_228_2025only_193dim.pt")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading Hard-tier docs from pan2025_full (real official splits) ONLY...")
    train_docs = load_pan2025_full("data/raw", tier="hard", split="train")
    val_docs = load_pan2025_full("data/raw", tier="hard", split="validation")
    test_docs = load_pan2025_full("data/raw", tier="hard", split="test")
    print(f"  train={len(train_docs)}  val={len(val_docs)}  test={len(test_docs)}")

    print(f"\nLoading established DeBERTa+BiLSTM checkpoint: {args.checkpoint}")
    model, extractor = load_established_model(args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    print("\nComputing DeBERTa probabilities (train/val/test)...")
    p_deberta_train, y_train = deberta_probs_for_split(model, extractor, tokenizer, train_docs)
    p_deberta_val, y_val = deberta_probs_for_split(model, extractor, tokenizer, val_docs)
    p_deberta_test, y_test = deberta_probs_for_split(model, extractor, tokenizer, test_docs)
    print(f"  train boundaries={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    print("\nComputing EXPANDED (10-feature) surface-similarity features...")
    X_train, y_train2 = lgbm_features_for_docs(train_docs)
    X_val, y_val2 = lgbm_features_for_docs(val_docs)
    X_test, y_test2 = lgbm_features_for_docs(test_docs)
    assert np.array_equal(y_train, y_train2) and np.array_equal(y_val, y_val2) and np.array_equal(y_test, y_test2), \
        "Label ordering mismatch between DeBERTa and feature streams!"

    lgbm = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                          class_weight="balanced", verbosity=-1)
    lgbm.fit(X_train, y_train)
    p_lgbm_train = lgbm.predict_proba(X_train)[:, 1]
    p_lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    p_lgbm_test = lgbm.predict_proba(X_test)[:, 1]

    # Indices of the 3 strongest raw features (from earlier probes):
    # char_trigram_dissim(1), function_word_dist(0), punctuation_dist(2)
    strong_idx = [0, 1, 2]

    def build_stack_input(p_deberta, p_lgbm, X):
        return np.column_stack([p_deberta, p_lgbm, X[:, strong_idx]])

    stack_X_train = build_stack_input(p_deberta_train, p_lgbm_train, X_train)
    stack_X_val = build_stack_input(p_deberta_val, p_lgbm_val, X_val)
    stack_X_test = build_stack_input(p_deberta_test, p_lgbm_test, X_test)

    print("\nFitting LightGBM stacker (5 inputs: p_deberta, p_lgbm, "
         "char_trigram_dissim, function_word_dist, punctuation_dist) on VALIDATION only...")
    stacker = LGBMClassifier(n_estimators=60, num_leaves=7, max_depth=3,
                             learning_rate=0.05, class_weight="balanced", verbosity=-1)
    stacker.fit(stack_X_val, y_val)

    stack_probs_val = stacker.predict_proba(stack_X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, stack_probs_val)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = np.argmax(f1s[:-1])
    best_thresh = thresholds[best_idx]
    print(f"  Best threshold (on validation): {best_thresh:.3f}  val_F1={f1s[best_idx]:.4f}")
    print("  NOTE: this val_F1 is optimistic (stacker fit AND thresholded on the same "
         "validation set) -- the test number below is the one that matters.")

    print("\n" + "=" * 60)
    print("FINAL: OFFICIAL TEST SET (ONE evaluation)")
    print("=" * 60)
    stack_probs_test = stacker.predict_proba(stack_X_test)[:, 1]
    stack_preds_test = (stack_probs_test >= best_thresh).astype(int)
    test_f1 = f1_score(y_test, stack_preds_test)

    print(f"  Ensemble v2 (LightGBM stacker, 5 inputs) Hard test F1: {test_f1:.4f}")
    print(f"\n  For comparison:")
    print(f"    DeBERTa alone (tuned threshold, earlier run):        0.7067")
    print(f"    Ensemble v1 (logistic, 2 inputs):                    0.7203")
    print(f"    Your paper's existing 2024+2025 combined model:      0.7276")


if __name__ == "__main__":
    main()