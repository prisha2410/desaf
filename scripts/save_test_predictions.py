#!/usr/bin/env python3
"""
Loads a trained checkpoint and runs it on the REAL official PAN 2025 test
set (data/raw/pan2025_full/{tier}/test/), same as eval_official_test_2025.py,
but instead of just printing F1/Precision/Recall, saves the raw per-boundary
predictions/probabilities/labels/doc-lengths to a .npz file.

That .npz is what compute_significance.py (bootstrap + McNemar significance
testing) and ensemble_seeds.py (multi-seed probability averaging) actually
need as input -- eval_official_test_2025.py alone never produces this file.

Run this ONCE per checkpoint you intend to report, same discipline as
eval_official_test_2025.py -- this touches the real held-out test set.

Usage:
    python scripts/save_test_predictions.py \\
        --checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt \\
        --threshold 0.52 \\
        --attention-fusion --bilstm --no-char-ngrams --contrastive \\
        --out checkpoints/seed43/test_predictions.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full
from src.features.stylometric import StylometricExtractor
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset
from train_deberta_fusion import encode_document, build_style_diffs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, required=True,
                   help="best_threshold printed at the end of training for this checkpoint")
    p.add_argument("--out", required=True,
                   help="Where to save the predictions .npz, e.g. checkpoints/seed43/test_predictions.npz")
    p.add_argument("--attention-fusion", action="store_true")
    p.add_argument("--film-gate", action="store_true",
                   help="Match a checkpoint trained with --film-gate (FiLM fusion instead of gate fusion)")
    p.add_argument("--bilstm", action="store_true")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--no-style", action="store_true")
    p.add_argument("--no-char-ngrams", action="store_true",
                   help="Match a checkpoint trained with --no-char-ngrams (193-dim instead of 258-dim)")
    p.add_argument("--contrastive", action="store_true",
                   help="Match a checkpoint trained with --contrastive (adds contrastive_proj weights)")
    return p.parse_args()


class Args:
    """Minimal shim so encode_document (which reads attrs off an `args`
    object) works without needing the full argparse Namespace from
    train_deberta_fusion.py."""
    def __init__(self, max_length, sub_batch, no_style):
        self.max_length = max_length
        self.sub_batch = sub_batch
        self.no_style = no_style


@torch.no_grad()
def predict_tier(model, dataset, tokenizer, device, threshold, eval_args):
    """Runs the model over every document in `dataset` and returns flat
    per-boundary preds/probs/labels arrays plus a per-document length array
    (so callers can reconstruct document boundaries for bootstrap resampling
    or WindowDiff)."""
    model.eval()
    all_probs, all_labels, lengths = [], [], []

    for idx in range(len(dataset)):
        doc = dataset[idx]
        labels = doc["labels"]
        n = len(labels)
        lengths.append(n)

        fused = encode_document(
            model, doc, tokenizer, device,
            not eval_args.no_style, eval_args.max_length, eval_args.sub_batch,
        )
        from torch.amp import autocast
        with autocast("cuda", dtype=torch.float16):
            logits = model.forward_document(fused.unsqueeze(0))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels)

    all_probs = np.array(all_probs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_preds = (all_probs >= threshold).astype(np.int64)
    lengths = np.array(lengths, dtype=np.int64)
    return all_preds, all_probs, all_labels, lengths


def main():
    args = parse_args()
    use_style = not args.no_style

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.out}")
    print("Loading OFFICIAL PAN 2025 TEST set (data/raw/pan2025_full/{tier}/test/)...")
    print("This is the real held-out set -- run this only once per checkpoint you report.\n")

    extractor = StylometricExtractor(use_char_ngrams=not args.no_char_ngrams) if use_style else None
    style_dim = extractor.feature_dim if extractor else 0
    print(f"Extractor feature_dim = {style_dim}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    model = DeBERTaFusionE2E(
        style_dim=style_dim,
        unfreeze_layers=args.unfreeze_layers,
        use_style=use_style,
        use_attention_fusion=args.attention_fusion,
        use_film=args.film_gate,
        use_bilstm=args.bilstm,
        bilstm_hidden=args.bilstm_hidden,
        dropout=args.dropout,
        use_adversarial=False,
        use_contrastive=args.contrastive,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    eval_args = Args(max_length=args.max_length, sub_batch=args.sub_batch, no_style=args.no_style)

    out_dict = {}
    print("\n" + "=" * 60)
    print("SAVING PER-BOUNDARY PREDICTIONS (2025, per tier)")
    print("=" * 60)
    for t in ["easy", "medium", "hard"]:
        test_docs = load_pan2025_full("data/raw", tier=t, split="test")
        sents_list = [d.sentences for d in test_docs]
        labels_list = [d.boundary_labels for d in test_docs]
        doc_ids = [d.doc_id for d in test_docs]
        style_diffs = build_style_diffs(test_docs, extractor) if use_style else None
        test_ds = DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

        preds, probs, labels, lengths = predict_tier(model, test_ds, tokenizer, DEVICE, args.threshold, eval_args)

        # compute_significance.py's key convention:
        out_dict[f"{t}__preds"] = preds
        out_dict[f"{t}__probs"] = probs
        out_dict[f"{t}__labels"] = labels
        out_dict[f"{t}__lengths"] = lengths
        # ensemble_seeds.py's key convention (year-prefixed):
        out_dict[f"2025_{t}__probs"] = probs
        out_dict[f"2025_{t}__labels"] = labels

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {t:6s} | F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  (n={len(test_docs)} docs, {len(preds)} boundaries)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    print(f"\nSaved predictions to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("\nUsable with:")
    print(f"  python scripts/compute_significance.py --a <other>.npz --b {out_path} --tier hard --a-name X --b-name Y")
    print(f"  python scripts/ensemble_seeds.py --predictions {out_path} <other seeds>.npz")


if __name__ == "__main__":
    main()
