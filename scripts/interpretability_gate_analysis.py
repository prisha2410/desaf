#!/usr/bin/env python3
"""
Interpretability analysis of the StyleAttentionFusion gate: which
stylometric features does the model learn to trust, and does that trust
differ between true style-change boundaries and non-change boundaries,
and across difficulty tiers.

Runs on CPU by default so it is safe to run ALONGSIDE a GPU training run
(e.g. run_remaining_v3.ps1) without competing for VRAM. It only needs one
forward pass over the VALIDATION set per tier (never touches test).

Usage:
    python scripts/interpretability_gate_analysis.py \
        --checkpoint checkpoints/curriculum_228_2025only_contrastive.pt \
        --contrastive --device cpu
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

FUNCTION_WORD_NAMES = None  # filled in from the extractor's own feature names


def load_model(checkpoint_path, use_contrastive, device):
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
    model.to(device)
    model.eval()
    return model, extractor


def collect_gate_values(model, extractor, tokenizer, docs, device, max_length=256, sub_batch=8):
    """Runs a forward pass over every boundary in `docs`, capturing the
    StyleAttentionFusion gate's raw output (g_i, one value per stylometric
    feature) via a forward hook on the gate submodule itself. Returns
    (gate_values: [N, 193], labels: [N])."""
    captured = {}

    def hook(module, inputs, output):
        captured["last"] = output.detach().cpu().numpy()

    handle = model.style_fusion.gate.register_forward_hook(hook)

    all_gates, all_labels = [], []
    try:
        with torch.no_grad():
            for d in docs:
                sents = d.sentences
                labels = d.boundary_labels
                n = min(len(sents) - 1, len(labels))
                if n <= 0:
                    continue

                vecs = [extractor.extract_vector(s) for s in sents]
                diffs = np.stack([
                    np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
                    for i in range(n)
                ])

                for start in range(0, n, sub_batch):
                    end = min(start + sub_batch, n)
                    batch_a = [sents[i] for i in range(start, end)]
                    batch_b = [sents[i + 1] for i in range(start, end)]
                    encoded = tokenizer(
                        batch_a, batch_b, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt",
                    ).to(device)
                    style = torch.tensor(diffs[start:end], dtype=torch.float32).to(device)

                    _ = model.encode_pairs(
                        encoded["input_ids"], encoded["attention_mask"], style_diff=style,
                    )
                    # captured["last"] now holds g_i for this sub-batch, shape (B, 193)
                    all_gates.append(captured["last"])
                    all_labels.extend(labels[start:end])
    finally:
        handle.remove()

    gates = np.concatenate(all_gates, axis=0)
    labels = np.array(all_labels, dtype=int)
    return gates, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/curriculum_228_2025only_contrastive.pt")
    ap.add_argument("--contrastive", action="store_true", default=True)
    ap.add_argument("--device", default="cpu", help="cpu (default, safe to run alongside GPU training) or cuda")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-prefix", default="results/interpretability")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} (CPU is recommended if a training run is using the GPU)")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, extractor = load_model(args.checkpoint, args.contrastive, device)
    feature_names = extractor.feature_names  # assumes StylometricExtractor exposes this

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    per_tier_gates = {}
    per_tier_labels = {}
    for tier in ["easy", "medium", "hard"]:
        print(f"\nCollecting gate values for {tier} tier (validation set)...")
        docs = load_pan2025_full("data/raw", tier=tier, split="validation")
        gates, labels = collect_gate_values(model, extractor, tokenizer, docs, device)
        per_tier_gates[tier] = gates
        per_tier_labels[tier] = labels
        print(f"  {tier}: {gates.shape[0]} boundaries, gate shape {gates.shape}")

    all_gates = np.concatenate([per_tier_gates[t] for t in per_tier_gates], axis=0)
    all_labels = np.concatenate([per_tier_labels[t] for t in per_tier_labels], axis=0)

    change_mask = all_labels == 1
    nochange_mask = all_labels == 0

    mean_gate_change = all_gates[change_mask].mean(axis=0)
    mean_gate_nochange = all_gates[nochange_mask].mean(axis=0)
    delta = mean_gate_change - mean_gate_nochange  # positive = trusted more at TRUE changes

    order = np.argsort(np.abs(delta))[::-1][:args.top_n]

    print(f"\nTop {args.top_n} features by |gate delta| (change - no-change):")
    for idx in order:
        print(f"  {feature_names[idx]:25s} delta={delta[idx]:+.4f}  "
              f"change={mean_gate_change[idx]:.4f}  nochange={mean_gate_nochange[idx]:.4f}")

    # --- Plot 1: top-N features by discriminability ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    ax = axes[0]
    names_sorted = [feature_names[i] for i in order]
    deltas_sorted = [delta[i] for i in order]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in deltas_sorted]
    ax.barh(range(len(order)), deltas_sorted, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gate delta (change - no-change)")
    ax.set_title(f"Top {args.top_n} Stylometric Features\nby Attention Gate Discriminability")
    ax.axvline(0, color="black", linewidth=0.8)

    # --- Plot 2: raw gate values, top 10, change vs no-change ---
    ax = axes[1]
    top10 = order[:10]
    names10 = [feature_names[i] for i in top10]
    x = np.arange(len(top10))
    width = 0.35
    ax.bar(x - width / 2, mean_gate_change[top10], width, label="Change boundary", color="#d62728")
    ax.bar(x + width / 2, mean_gate_nochange[top10], width, label="No-change boundary", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names10, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean attention gate value")
    ax.set_title("Top 10 Features:\nGate Values at Change vs No-Change")
    ax.legend()

    # --- Plot 3: overall gate reliance per tier, with std as error bar ---
    ax = axes[2]
    tier_means = [per_tier_gates[t].mean() for t in ["easy", "medium", "hard"]]
    tier_stds = [per_tier_gates[t].std() for t in ["easy", "medium", "hard"]]
    ax.bar(["easy", "medium", "hard"], tier_means, yerr=tier_stds,
           color=["#2ca02c", "#ff7f0e", "#d62728"], capsize=5)
    ax.set_ylabel("Mean attention gate value")
    ax.set_title("Stylometric Gate Reliance\nby Difficulty Tier")

    plt.tight_layout()
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    out_path = f"{args.out_prefix}_gate_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")

    # --- Correctness-conditioned analysis (the piece flagged as missing) ---
    print("\nComputing correctness-conditioned gate comparison "
          "(requires a forward pass through forward_document, per-document)...")
    # NOTE: this requires document-level predictions, not just per-boundary
    # gate values, so it is a second pass using forward_document per doc.
    correct_gates, incorrect_gates = [], []
    idx_cursor = 0
    for tier in ["easy", "medium", "hard"]:
        docs = load_pan2025_full("data/raw", tier=tier, split="validation")
        for d in docs:
            sents = d.sentences
            labels_d = d.boundary_labels
            n = min(len(sents) - 1, len(labels_d))
            if n <= 0:
                continue
            style_diffs = build_style_diffs([d], extractor)
            ds = DocLevelDataset([sents], [labels_d], style_diffs, doc_ids=[d.doc_id])
            doc = ds[0]

            class EvalArgs:
                max_length = 256
                sub_batch = 8
            fused = encode_document(model, doc, tokenizer, device, True, 256, 8)
            with torch.no_grad():
                logits = model.forward_document(fused.unsqueeze(0))
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
            true = np.array(doc["labels"][:len(preds)])

            doc_gates = all_gates[idx_cursor: idx_cursor + len(preds)]
            idx_cursor += len(preds)

            correct_mask = preds == true
            correct_gates.append(doc_gates[correct_mask])
            incorrect_gates.append(doc_gates[~correct_mask])

    correct_gates = np.concatenate([g for g in correct_gates if len(g)], axis=0)
    incorrect_gates = np.concatenate([g for g in incorrect_gates if len(g)], axis=0)

    print(f"\nCorrectly classified boundaries: {len(correct_gates)}, "
          f"mean gate value = {correct_gates.mean():.4f}")
    print(f"Incorrectly classified boundaries: {len(incorrect_gates)}, "
          f"mean gate value = {incorrect_gates.mean():.4f}")
    diff = correct_gates.mean() - incorrect_gates.mean()
    print(f"Difference (correct - incorrect): {diff:+.4f}")
    print("A large positive difference suggests the model is more confidently using "
          "stylometric evidence when it gets the boundary right, and defaulting to "
          "weaker/more uniform gate values when it gets confused.")


if __name__ == "__main__":
    main()
