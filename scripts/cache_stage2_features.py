#!/usr/bin/env python3
"""
Runs a trained DeSAF checkpoint ONCE over the official PAN 2025 validation and
test sets and caches, for every sentence boundary:

  h        DeBERTa [CLS] vector of the pair (768, from the checkpoint's own encoder)
  style    |f(s_i+1) - f(s_i)|, the same 193-dim stylometric diff the model uses
  surf     3 surface-similarity distances (function words, char trigrams, punctuation)
  overlap  TF-IDF cosine topic overlap of the two sentences (vectorizer fit on TRAIN text)
  probs    the full checkpoint's own probability (sanity check vs. its known F1)
  labels, doc_lengths, doc_tier, doc_ids

gate_lab.py then trains small fusion heads on these caches in minutes instead of
retraining DeBERTa for every gate idea.

Usage (gated headline checkpoint, threshold from its training log):
    python scripts/cache_stage2_features.py \
        --checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt \
        --threshold 0.50 --attention-fusion --bilstm --no-char-ngrams --contrastive \
        --name seed43_gate
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset
from train_deberta_fusion import encode_document, build_style_diffs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS = ["easy", "medium", "hard"]

FUNCTION_WORDS = sorted(set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in
into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your
yours yourself yourselves also however though although yet still even quite rather
""".split()))
FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}
PUNCT = ".,;:!?'\"-()"
WORD_RE = re.compile(r"[a-z']+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, required=True,
                   help="the checkpoint's validation-tuned threshold (from its training log)")
    p.add_argument("--name", required=True, help="cache name, e.g. seed43_gate")
    p.add_argument("--out-dir", default="checkpoints/stage2_cache")
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--attention-fusion", action="store_true")
    p.add_argument("--film-gate", action="store_true")
    p.add_argument("--bilstm", action="store_true")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--no-char-ngrams", action="store_true")
    p.add_argument("--contrastive", action="store_true")
    return p.parse_args()


def fw_vector(text):
    words = WORD_RE.findall(text.lower())
    v = np.zeros(len(FUNCTION_WORDS), dtype=np.float32)
    for w in words:
        j = FW_INDEX.get(w)
        if j is not None:
            v[j] += 1
    return v / max(len(words), 1)


def punct_vector(text):
    v = np.array([text.count(c) for c in PUNCT], dtype=np.float32)
    return v / max(len(text), 1)


def trigrams(text):
    t = text.lower()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def surface_distances(a, b):
    fw = float(np.abs(fw_vector(a) - fw_vector(b)).sum())
    ta, tb = trigrams(a), trigrams(b)
    union = len(ta | tb)
    tri = 1.0 - (len(ta & tb) / union) if union else 0.0
    pu = float(np.abs(punct_vector(a) - punct_vector(b)).sum())
    return fw, tri, pu


def pair_overlap(vectorizer, a_list, b_list):
    # TfidfVectorizer rows are L2-normalised, so the row-wise dot product is the cosine.
    va = vectorizer.transform(a_list)
    vb = vectorizer.transform(b_list)
    return np.asarray(va.multiply(vb).sum(axis=1)).ravel().astype(np.float32)


def f1_of(preds, labels):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


class Args:
    def __init__(self, max_length, sub_batch):
        self.max_length = max_length
        self.sub_batch = sub_batch
        self.no_style = False


def main():
    args = parse_args()
    if not args.bilstm:
        print("ERROR: only BiLSTM checkpoints are supported (pass --bilstm).")
        sys.exit(1)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}  (threshold {args.threshold})")

    print("\nFitting TF-IDF topic-overlap vectorizer on TRAIN sentences only...")
    t0 = time.time()
    train_sents = []
    for t in TIERS:
        for d in load_pan2025_full("data/raw", tier=t, split="train"):
            train_sents.extend(d.sentences)
    vectorizer = TfidfVectorizer(max_features=20000)
    vectorizer.fit(train_sents)
    print(f"  {len(train_sents)} train sentences, vocab {len(vectorizer.vocabulary_)}, {time.time() - t0:.0f}s")
    del train_sents

    extractor = StylometricExtractor(use_char_ngrams=not args.no_char_ngrams)
    style_dim = extractor.feature_dim
    print(f"Stylometric feature_dim = {style_dim}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)
    model = DeBERTaFusionE2E(
        style_dim=style_dim,
        unfreeze_layers=args.unfreeze_layers,
        use_style=True,
        use_attention_fusion=args.attention_fusion,
        use_film=args.film_gate,
        use_bilstm=True,
        bilstm_hidden=args.bilstm_hidden,
        dropout=args.dropout,
        use_adversarial=False,
        use_contrastive=args.contrastive,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    eval_args = Args(args.max_length, args.sub_batch)

    for split in splits:
        print(f"\n=== {split} ===")
        t0 = time.time()
        H, STYLE, SURF, OVL, PROBS, LABELS = [], [], [], [], [], []
        doc_lengths, doc_tier, doc_ids = [], [], []

        for tier_id, t in enumerate(TIERS):
            docs = load_pan2025_full("data/raw", tier=t, split=split)
            style_diffs = build_style_diffs(docs, extractor)
            ds = DocLevelDataset([d.sentences for d in docs], [d.boundary_labels for d in docs],
                                 style_diffs, doc_ids=[d.doc_id for d in docs])
            tier_probs, tier_labels = [], []
            for i in range(len(ds)):
                doc = ds[i]
                n = len(doc["labels"])
                with torch.no_grad():
                    fused = encode_document(model, doc, tokenizer, DEVICE, True,
                                            eval_args.max_length, eval_args.sub_batch)
                    with autocast("cuda", dtype=torch.float16):
                        logits = model.forward_document(fused.unsqueeze(0))
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1).astype(np.float32)
                H.append(fused[:, :DeBERTaFusionE2E.DEBERTA_DIM].float().cpu().numpy())
                STYLE.append(np.asarray(doc["style_diffs"], dtype=np.float32).reshape(n, style_dim))
                a_list = [p[0] for p in doc["pairs"]]
                b_list = [p[1] for p in doc["pairs"]]
                SURF.append(np.array([surface_distances(a, b) for a, b in zip(a_list, b_list)],
                                     dtype=np.float32).reshape(n, 3))
                OVL.append(pair_overlap(vectorizer, a_list, b_list))
                PROBS.append(probs.astype(np.float32))
                LABELS.append(np.asarray(doc["labels"], dtype=np.int64))
                doc_lengths.append(n)
                doc_tier.append(tier_id)
                doc_ids.append(str(doc["doc_id"]))
                tier_probs.append(probs)
                tier_labels.append(doc["labels"])
            tp_ = np.concatenate(tier_probs)
            tl_ = np.concatenate([np.asarray(x) for x in tier_labels])
            # The eval scripts compare float16 probabilities; save_test_predictions compares float32.
            f1_eval = f1_of((tp_.astype(np.float16) >= args.threshold).astype(np.int64), tl_)
            f1_pred = f1_of((tp_ >= args.threshold).astype(np.int64), tl_)
            print(f"  {t:6s} | {len(ds)} docs, {len(tl_)} boundaries | own F1 @ {args.threshold:.2f}: "
                  f"{f1_eval:.4f} (eval-log style), {f1_pred:.4f} (predictions-file style)")

        out = out_dir / f"{args.name}_{split}.npz"
        np.savez_compressed(
            out,
            h=np.concatenate(H).astype(np.float16),
            style=np.concatenate(STYLE),
            surf=np.concatenate(SURF),
            overlap=np.concatenate(OVL),
            probs=np.concatenate(PROBS),
            labels=np.concatenate(LABELS),
            doc_lengths=np.array(doc_lengths, dtype=np.int64),
            doc_tier=np.array(doc_tier, dtype=np.int64),
            doc_ids=np.array(doc_ids),
            threshold=np.float32(args.threshold),
        )
        ovl = np.concatenate(OVL)
        print(f"  overlap: mean {ovl.mean():.4f}, median {np.median(ovl):.4f}, share > 0.075: {(ovl > 0.075).mean():.2f}")
        print(f"  saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")

    print("\nCheck: on test, the 'eval-log style' F1 should match this checkpoint's official eval log,")
    print("and the 'predictions-file style' F1 should match what save_test_predictions.py printed.")


if __name__ == "__main__":
    main()
