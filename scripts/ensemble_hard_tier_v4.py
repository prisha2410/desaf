#!/usr/bin/env python3
"""
Ensemble v4: Tier A combined -- logistic-regression stacker with calibrated
inputs (A1), expanded surface-similarity features incl. z-scored Delta,
NCD, POS-trigram distance (A2), and out-of-fold surface-model predictions
to avoid stacker leakage (A3).

Still PAN 2025_full ONLY, real splits, ONE final test evaluation.

Usage:
    python scripts/ensemble_hard_tier_v4.py \
        --checkpoint checkpoints/curriculum_228_2025only_193dim_seed42.pt \
        --contrastive-checkpoint checkpoints/curriculum_228_2025only_contrastive.pt
"""
import argparse
import bz2
import lzma
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset

from train_deberta_fusion import encode_document, build_style_diffs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import spacy
    _NLP = None
    def get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])
        return _NLP
    SPACY_OK = True
except ImportError:
    SPACY_OK = False
    print("[WARN] spaCy not available -- POS-trigram distance feature will be all-zero.")


# --------------------------------------------------------------------------- #
#  Part A: DeBERTa probabilities AND raw logits (logits needed for calibration)
# --------------------------------------------------------------------------- #

class EvalArgs:
    def __init__(self, max_length=256, sub_batch=8, no_style=False):
        self.max_length = max_length
        self.sub_batch = sub_batch
        self.no_style = no_style


def load_model(checkpoint_path, use_contrastive=False):
    extractor = StylometricExtractor(use_char_ngrams=False)
    style_dim = extractor.feature_dim
    assert style_dim == 193, f"Expected 193-dim extractor, got {style_dim}"

    model = DeBERTaFusionE2E(
        style_dim=style_dim, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
        use_contrastive=use_contrastive,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, extractor


def deberta_logits_for_split(model, extractor, tokenizer, docs):
    """Returns (logits, labels) -- RAW logits (pre-sigmoid), needed for
    temperature scaling. Same doc/boundary ordering as the surface
    features below."""
    sents_list = [d.sentences for d in docs]
    labels_list = [d.boundary_labels for d in docs]
    doc_ids = [d.doc_id for d in docs]
    style_diffs = build_style_diffs(docs, extractor)
    ds = DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

    eval_args = EvalArgs()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            doc = ds[idx]
            fused = encode_document(model, doc, tokenizer, DEVICE, True,
                                    eval_args.max_length, eval_args.sub_batch)
            from torch.amp import autocast
            with autocast("cuda", dtype=torch.float16):
                logits = model.forward_document(fused.unsqueeze(0))
            all_logits.extend(logits.float().cpu().numpy())
            all_labels.extend(doc["labels"])
    return np.array(all_logits), np.array(all_labels, dtype=int)


def fit_temperature(logits, labels, n_iters=200, lr=0.05):
    """1-parameter temperature scaling: find T minimizing BCE of
    sigmoid(logits / T). Simple gradient descent, no extra deps."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T), starts at T=1
    optimizer = torch.optim.Adam([log_T], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(n_iters):
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = bce(logits_t / T, labels_t)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_T).item())


def apply_temperature(logits, T):
    return 1.0 / (1.0 + np.exp(-logits / T))


# --------------------------------------------------------------------------- #
#  Part B: Expanded surface-similarity features
#    Original 7 (function_word_dist, char_trigram_dissim, punctuation_dist,
#    ttr_diff, sent_len_diff, word_count_diff, sent_count_diff)
#    NEW: cosine_delta, manhattan_delta (z-scored function words),
#         ncd_gzip, ncd_bz2, ncd_lzma,
#         pos_trigram_cosine, pos_trigram_js
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


def fw_counts_raw(toks):
    """Raw (not ratio) function-word counts -- needed for z-scoring against
    corpus mean/std computed separately."""
    return np.array([toks.count(w) for w in FUNCTION_WORDS], dtype=np.float32)


def pos_trigrams(text):
    if not SPACY_OK:
        return Counter()
    doc = get_nlp()(text)
    tags = [t.pos_ for t in doc]
    if len(tags) < 3:
        return Counter()
    return Counter(tuple(tags[i:i+3]) for i in range(len(tags) - 2))


def js_divergence(c1, c2):
    keys = set(c1) | set(c2)
    if not keys:
        return 0.0
    n1 = sum(c1.values()) or 1
    n2 = sum(c2.values()) or 1
    p = np.array([c1.get(k, 0) / n1 for k in keys], dtype=np.float64)
    q = np.array([c2.get(k, 0) / n2 for k in keys], dtype=np.float64)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ncd(bytes_a, bytes_b, compressor):
    ca = len(compressor(bytes_a))
    cb = len(compressor(bytes_b))
    cab = len(compressor(bytes_a + b" " + bytes_b))
    return (cab - min(ca, cb)) / max(ca, cb, 1)


def para_stats(p):
    toks = tokenize(p)
    n = max(len(toks), 1)
    sents = sentence_split(p)
    n_sents = max(len(sents), 1)
    fw_ratio_vec = np.array([toks.count(w) / n for w in FUNCTION_WORDS], dtype=np.float32)
    fw_raw_vec = fw_counts_raw(toks)
    trigrams = char_trigrams(p)
    pos_tri = pos_trigrams(p)
    punct = punctuation_profile(p)
    ttr = len(set(toks)) / n
    avg_sent_len = n / n_sents
    return {
        "fw_ratio_vec": fw_ratio_vec, "fw_raw_vec": fw_raw_vec,
        "trigrams": trigrams, "pos_tri": pos_tri, "punct": punct,
        "ttr": ttr, "avg_sent_len": avg_sent_len, "n_words": n, "n_sents": n_sents,
        "raw_bytes": p.encode("utf-8"),
    }


def compute_fw_zscore_stats(train_docs):
    """Corpus mean/std of raw function-word counts, computed ONCE from
    train paragraphs only (never validation/test) -- used for z-scored
    Delta distances."""
    all_counts = []
    for d in train_docs:
        for s in d.sentences:
            toks = tokenize(s)
            all_counts.append(fw_counts_raw(toks))
    all_counts = np.stack(all_counts)
    mean = all_counts.mean(axis=0)
    std = all_counts.std(axis=0) + 1e-6  # avoid div-by-zero
    return mean, std


FEATURE_NAMES = [
    "function_word_dist", "char_trigram_dissim", "punctuation_dist", "ttr_diff",
    "sent_len_diff", "word_count_diff", "sent_count_diff",
    "cosine_delta", "manhattan_delta",
    "ncd_gzip", "ncd_bz2", "ncd_lzma",
    "pos_trigram_cosine_dissim", "pos_trigram_js",
]


def boundary_features(pa, pb, fw_mean, fw_std):
    sa, sb = para_stats(pa), para_stats(pb)

    fw_dist = float(np.linalg.norm(sa["fw_ratio_vec"] - sb["fw_ratio_vec"]))
    trigram_dissim = 1.0 - cosine_from_counters(sa["trigrams"], sb["trigrams"])
    punct_dist = float(np.linalg.norm(sa["punct"] - sb["punct"]))
    ttr_diff = abs(sa["ttr"] - sb["ttr"])
    sent_len_diff = abs(sa["avg_sent_len"] - sb["avg_sent_len"])
    word_count_diff = abs(sa["n_words"] - sb["n_words"])
    sent_count_diff = abs(sa["n_sents"] - sb["n_sents"])

    za = (sa["fw_raw_vec"] - fw_mean) / fw_std
    zb = (sb["fw_raw_vec"] - fw_mean) / fw_std
    manhattan_delta = float(np.mean(np.abs(za - zb)))
    za_n, zb_n = np.linalg.norm(za), np.linalg.norm(zb)
    cosine_delta = 1.0 - float(np.dot(za, zb) / (za_n * zb_n)) if za_n > 0 and zb_n > 0 else 0.0

    ncd_gzip = ncd(sa["raw_bytes"], sb["raw_bytes"], zlib.compress)
    ncd_bz2_ = ncd(sa["raw_bytes"], sb["raw_bytes"], bz2.compress)
    ncd_lzma_ = ncd(sa["raw_bytes"], sb["raw_bytes"], lzma.compress)

    pos_cos_dissim = 1.0 - cosine_from_counters(sa["pos_tri"], sb["pos_tri"])
    pos_js = js_divergence(sa["pos_tri"], sb["pos_tri"])

    return [fw_dist, trigram_dissim, punct_dist, ttr_diff,
           sent_len_diff, word_count_diff, sent_count_diff,
           cosine_delta, manhattan_delta,
           ncd_gzip, ncd_bz2_, ncd_lzma_,
           pos_cos_dissim, pos_js]


def features_for_docs(docs, fw_mean, fw_std):
    X, y = [], []
    for d in docs:
        n = len(d.boundary_labels)
        for i in range(n):
            X.append(boundary_features(d.sentences[i], d.sentences[i + 1], fw_mean, fw_std))
            y.append(int(d.boundary_labels[i]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def doc_boundary_counts(docs):
    """For mapping k-fold document splits to boundary-level index ranges."""
    counts = [len(d.boundary_labels) for d in docs]
    return counts


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/curriculum_228_2025only_193dim_seed42.pt")
    ap.add_argument("--contrastive-checkpoint", default="checkpoints/curriculum_228_2025only_contrastive.pt")
    ap.add_argument("--oof-folds", type=int, default=5)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading Hard-tier docs from pan2025_full (real official splits) ONLY...")
    train_docs = load_pan2025_full("data/raw", tier="hard", split="train")
    val_docs = load_pan2025_full("data/raw", tier="hard", split="validation")
    test_docs = load_pan2025_full("data/raw", tier="hard", split="test")
    print(f"  train={len(train_docs)}  val={len(val_docs)}  test={len(test_docs)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    print(f"\nLoading plain checkpoint: {args.checkpoint}")
    model_plain, extractor = load_model(args.checkpoint, use_contrastive=False)
    print(f"Loading contrastive checkpoint: {args.contrastive_checkpoint}")
    model_contrastive, _ = load_model(args.contrastive_checkpoint, use_contrastive=True)

    print("\nComputing plain DeBERTa logits (train/val/test)...")
    logit_plain_train, y_train = deberta_logits_for_split(model_plain, extractor, tokenizer, train_docs)
    logit_plain_val, y_val = deberta_logits_for_split(model_plain, extractor, tokenizer, val_docs)
    logit_plain_test, y_test = deberta_logits_for_split(model_plain, extractor, tokenizer, test_docs)

    print("Computing contrastive DeBERTa logits (train/val/test)...")
    logit_contrast_train, y_train2 = deberta_logits_for_split(model_contrastive, extractor, tokenizer, train_docs)
    logit_contrast_val, y_val2 = deberta_logits_for_split(model_contrastive, extractor, tokenizer, val_docs)
    logit_contrast_test, y_test2 = deberta_logits_for_split(model_contrastive, extractor, tokenizer, test_docs)
    assert np.array_equal(y_train, y_train2) and np.array_equal(y_val, y_val2) and np.array_equal(y_test, y_test2)

    # --- A1: temperature scaling, fit on VALIDATION only ---
    print("\n[A1] Fitting temperature scaling on validation logits...")
    T_plain = fit_temperature(logit_plain_val, y_val)
    T_contrast = fit_temperature(logit_contrast_val, y_val)
    print(f"  T_plain={T_plain:.3f}  T_contrastive={T_contrast:.3f}")

    p_plain_train = apply_temperature(logit_plain_train, T_plain)
    p_plain_val = apply_temperature(logit_plain_val, T_plain)
    p_plain_test = apply_temperature(logit_plain_test, T_plain)
    p_contrast_train = apply_temperature(logit_contrast_train, T_contrast)
    p_contrast_val = apply_temperature(logit_contrast_val, T_contrast)
    p_contrast_test = apply_temperature(logit_contrast_test, T_contrast)

    # --- A2: expanded surface-similarity features ---
    print("\n[A2] Computing expanded surface-similarity features "
         f"({len(FEATURE_NAMES)} columns, incl. z-scored Delta / NCD / POS-trigram)...")
    fw_mean, fw_std = compute_fw_zscore_stats(train_docs)  # fit on TRAIN only
    X_train, y_train3 = features_for_docs(train_docs, fw_mean, fw_std)
    X_val, y_val3 = features_for_docs(val_docs, fw_mean, fw_std)
    X_test, y_test3 = features_for_docs(test_docs, fw_mean, fw_std)
    assert np.array_equal(y_train, y_train3) and np.array_equal(y_val, y_val3) and np.array_equal(y_test, y_test3), \
        "Label ordering mismatch across streams!"

    # --- A3: out-of-fold surface-model predictions on validation ---
    print(f"\n[A3] Out-of-fold ({args.oof_folds}-fold, document-level) surface-model "
         "predictions on validation (avoids stacker leakage)...")
    val_counts = doc_boundary_counts(val_docs)
    val_boundaries_total = sum(val_counts)
    doc_start = np.cumsum([0] + val_counts[:-1])

    kf = KFold(n_splits=args.oof_folds, shuffle=True, random_state=42)
    p_surface_val_oof = np.zeros(val_boundaries_total, dtype=np.float32)
    doc_idx_arr = np.arange(len(val_docs))
    for fold_i, (train_doc_idx, held_doc_idx) in enumerate(kf.split(doc_idx_arr)):
        train_boundary_mask = np.zeros(val_boundaries_total, dtype=bool)
        held_boundary_mask = np.zeros(val_boundaries_total, dtype=bool)
        for di in train_doc_idx:
            train_boundary_mask[doc_start[di]:doc_start[di] + val_counts[di]] = True
        for di in held_doc_idx:
            held_boundary_mask[doc_start[di]:doc_start[di] + val_counts[di]] = True

        fold_model = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                    class_weight="balanced", verbosity=-1)
        # Fold's own "train" is TRAIN split + val-minus-held-fold, to keep folds reasonably sized
        fold_X = np.concatenate([X_train, X_val[train_boundary_mask]], axis=0)
        fold_y = np.concatenate([y_train, y_val[train_boundary_mask]], axis=0)
        fold_model.fit(fold_X, fold_y)
        p_surface_val_oof[held_boundary_mask] = fold_model.predict_proba(X_val[held_boundary_mask])[:, 1]
        print(f"  fold {fold_i+1}/{args.oof_folds} done")

    # Surface model refit on ALL of train+val for the actual test prediction
    surface_final = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                   class_weight="balanced", verbosity=-1)
    surface_final.fit(np.concatenate([X_train, X_val], axis=0),
                      np.concatenate([y_train, y_val], axis=0))
    p_surface_train = surface_final.predict_proba(X_train)[:, 1]  # for reference only, unused in stacker fit
    p_surface_test = surface_final.predict_proba(X_test)[:, 1]

    # Calibrate the surface stream too (isotonic, using the OOF val predictions -- honest, no leakage)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_surface_val_oof, y_val)
    p_surface_val_cal = iso.transform(p_surface_val_oof)
    p_surface_test_cal = iso.transform(p_surface_test)

    # --- A1 continued: logistic-regression stacker on calibrated inputs ---
    strong_idx = [7, 8, 0, 1, 2]  # cosine_delta, manhattan_delta, fw_dist, trigram_dissim, punct_dist
    stack_X_val = np.column_stack([p_plain_val, p_contrast_val, p_surface_val_cal, X_val[:, strong_idx]])
    stack_X_test = np.column_stack([p_plain_test, p_contrast_test, p_surface_test_cal, X_test[:, strong_idx]])

    print("\nFitting L2-logistic-regression stacker on VALIDATION (OOF surface + calibrated DeBERTa)...")
    stacker = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    stacker.fit(stack_X_val, y_val)
    print(f"  Stacker coefficients: {dict(zip(['p_plain','p_contrast','p_surface']+[FEATURE_NAMES[i] for i in strong_idx], stacker.coef_[0].round(3)))}")

    stack_probs_val = stacker.predict_proba(stack_X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, stack_probs_val)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = np.argmax(f1s[:-1])
    best_thresh = thresholds[best_idx]
    print(f"  Best threshold (on validation): {best_thresh:.3f}  val_F1={f1s[best_idx]:.4f}")
    print("  NOTE: this val_F1 is optimistic -- the test number below is the one that matters.")

    print("\n" + "=" * 60)
    print("FINAL: OFFICIAL TEST SET (ONE evaluation)")
    print("=" * 60)
    stack_probs_test = stacker.predict_proba(stack_X_test)[:, 1]
    stack_preds_test = (stack_probs_test >= best_thresh).astype(int)
    test_f1 = f1_score(y_test, stack_preds_test)

    print(f"  Ensemble v4 (Tier A: logistic stacker + calibration + expanded features + OOF) "
         f"Hard test F1: {test_f1:.4f}")
    print(f"\n  For comparison:")
    print(f"    Plain DeBERTa alone:                       0.7067")
    print(f"    Contrastive DeBERTa alone:                 0.7097")
    print(f"    Ensemble v1 (2-way, logistic):              0.7203")
    print(f"    Ensemble v2 (2-way, LightGBM stack):        0.7232")
    print(f"    Ensemble v3 (3-way, LightGBM stack):        0.7350")


if __name__ == "__main__":
    main()