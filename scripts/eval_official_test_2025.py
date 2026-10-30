#!/usr/bin/env python3
"""
The ONE honest evaluation: loads a trained checkpoint and evaluatesit on
the REAL official PAN 2025 test set (data/raw/pan2025_full/{tier}/test/),
which has never been touched during training or model selection.

Run this ONCE per checkpoint you actually intend to report. Do not use
test-set results to pick between checkpoints/hyperparameters -- that would
turn this held-out set into a validation set in disguise.

Usage:
    python scripts/eval_official_test_2025.py \\
        --checkpoint checkpoints/curriculum_228_2025only_real.pt \\
        --threshold 0.56 \\
        --attention-fusion --bilstm
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

# Reuse the exact same encode_document / evaluate_bilstm logic usedduring
# training, so the eval math is guaranteed identical to what val_F1meant.
from train_deberta_fusion import encode_document, evaluate_bilstm, build_style_diffs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, required=True,
                   help="best_threshold printed at the end of training for this checkpoint")
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
    """Minimal shim so evaluate_bilstm/encode_document (which read attrs
    off an `args` object) work without needing the full argparse Namespace
    from train_deberta_fusion.py."""
    def __init__(self, max_length, sub_batch, no_style):
        self.max_length = max_length
        self.sub_batch = sub_batch
        self.no_style = no_style


def main():
    args = parse_args()
    use_style = not args.no_style

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
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
    model.load_state_dict(state, strict=True)  # strict=True on purpose:
    # this checkpoint should exactly match the architecture you just built,
    # since it's the very checkpoint that was saved with this style_dim.
    model.to(DEVICE)
    model.eval()

    eval_args = Args(max_length=args.max_length, sub_batch=args.sub_batch, no_style=args.no_style)

    per_tier_results = {}
    all_f1s = []
    print("\n" + "=" * 60)
    print("OFFICIAL TEST SET RESULTS (2025, per tier)")
    print("=" * 60)
    for t in ["easy", "medium", "hard"]:
        test_docs = load_pan2025_full("data/raw", tier=t, split="test")
        sents_list = [d.sentences for d in test_docs]
        labels_list = [d.boundary_labels for d in test_docs]
        doc_ids = [d.doc_id for d in test_docs]
        style_diffs = build_style_diffs(test_docs, extractor) if use_style else None
        test_ds = DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

        metrics = evaluate_bilstm(model, test_ds, tokenizer, DEVICE, args.threshold, eval_args)
        per_tier_results[t] = metrics
        all_f1s.append(metrics["f1"])
        print(f"  {t:6s} | F1={metrics['f1']:.4f}  WD={metrics['window_diff']:.4f}  "
             f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  (n={len(test_docs)})")

    macro_f1 = float(np.mean(all_f1s))
    print(f"\n  Macro F1 (mean of Easy/Medium/Hard): {macro_f1:.4f}")

    print(f"\nFor comparison, your paper's existing official-test numbers (2024+2025 combined, 193-dim):")
    print(f"    Easy F1:   0.9287")
    print(f"    Medium F1: 0.7268")
    print(f"    Hard F1:   0.7276")
    print(f"    Macro F1:  0.7944")


if __name__ == "__main__":
    main()