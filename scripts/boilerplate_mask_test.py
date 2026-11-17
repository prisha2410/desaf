#!/usr/bin/env python3
"""
Does DeSAF need the specific boilerplate strings it saw in training?  GPU, about 5-10 minutes per encoder.

All 42 lines that occur >= 10 times across the PAN 2025 test documents also occur in the train text, so a model could
have memorised them. Here every such line in the test documents is replaced by ONE neutral sentence the model never saw
("This sentence has been replaced."), keeping every boundary and label. The block structure stays (a run of identical
placeholders, then normal text), but the strings are gone. Only documents that contain a repeated line are re-scored;
all other boundaries keep the saved predictions (a sample of unchanged documents is re-scored as a reproduction check).

Reported per tier and encoder, official pooled two-class macro-F1: boundaries next to a repeated line (masked vs the
saved unmasked predictions), elsewhere, and all; plus the edge rule (change iff exactly one side is a repeated line).

Boundaries next to a repeated line are split into EDGE (exactly one side is a repeated line: masking removes only the
string, the boundary between placeholder and normal text stays visible) and INSIDE (both sides repeated: two different
repeated lines become two identical placeholders, so a change there becomes undetectable for any model; these numbers
are bounded by the masking design and are reported only for completeness).

Pre-registered (2026-11-16, amended before any run after code review): on EDGE boundaries, if Easy and Medium macro-F1
stays >= 0.95 after masking on all 3 encoders, write "DeSAF reads the block structure of repeated lines, not the
memorised strings". If it falls to the edge rule's level or below on those boundaries, write "DeSAF's near-perfect score
next to repeated lines depends on the strings it saw in training". In between: report the numbers only.

Run from D:\\sass-tad:
    python scripts/boilerplate_mask_test.py
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.utils.data_loader import load_pan2025_full  # noqa: E402
from src.features.stylometric import StylometricExtractor  # noqa: E402
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset  # noqa: E402
from train_deberta_fusion import build_style_diffs  # noqa: E402
from save_test_predictions import predict_tier, Args, DEVICE  # noqa: E402

TIERS = ["easy", "medium", "hard"]
PLACEHOLDER = "This sentence has been replaced."
DEFAULT_CKPTS = ("seed43:checkpoints/curriculum_228_2025only_contrastive_seed43.pt:0.50,"
                 "seed44:checkpoints/curriculum_228_2025only_contrastive_seed44.pt:0.52,"
                 "seed45:checkpoints/curriculum_228_2025only_contrastive_seed45.pt:0.64")


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def load_model(ckpt):
    model = DeBERTaFusionE2E(style_dim=StylometricExtractor(use_char_ngrams=False).feature_dim, unfreeze_layers=2,
                             use_style=True, use_attention_fusion=True, use_film=False, use_bilstm=True,
                             bilstm_hidden=128, dropout=0.3, use_adversarial=False, use_contrastive=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    return model.to(DEVICE).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--checkpoints", default=DEFAULT_CKPTS, help="name:path:threshold, comma separated")
    ap.add_argument("--min-copies", type=int, default=10)
    ap.add_argument("--repro-docs", type=int, default=30, help="unchanged documents per tier re-scored as a check")
    ap.add_argument("--out", default="results/boilerplate_mask_test.json")
    a = ap.parse_args()
    t0 = time.time()
    docs = {t: load_pan2025_full(a.data_root, tier=t, split="test") for t in TIERS}
    for t in TIERS:                                   # same documents as the saved predictions (DocLevelDataset skips n <= 0)
        docs[t] = [d for d in docs[t] if min(len(d.sentences) - 1, len(d.boundary_labels)) > 0]
    counts = Counter(s for t in TIERS for d in docs[t] for s in d.sentences)
    rep_lines = {s for s, c in counts.items() if c >= a.min_copies}
    print(f"{len(rep_lines)} distinct test lines occur >= {a.min_copies} times (all tiers pooled); placeholder: '{PLACEHOLDER}'")

    extractor = StylometricExtractor(use_char_ngrams=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)
    eval_args = Args(max_length=256, sub_batch=8, no_style=False)
    rng = np.random.RandomState(0)

    plan = {}
    for t in TIERS:
        has = [i for i, d in enumerate(docs[t]) if any(s in rep_lines for s in d.sentences)]
        others = [i for i, d in enumerate(docs[t]) if i not in set(has)]
        repro = sorted(rng.choice(others, min(a.repro_docs, len(others)), replace=False).tolist()) if others else []
        has_set = set(has)
        idx = has + repro
        masked_sents = [[PLACEHOLDER if (i in has_set and s in rep_lines) else s for s in docs[t][i].sentences] for i in idx]
        labs = [docs[t][i].boundary_labels for i in idx]
        objs = [SimpleNamespace(sentences=s) for s in masked_sents]
        ds = DocLevelDataset(masked_sents, labs, build_style_diffs(objs, extractor), doc_ids=[str(i) for i in idx])
        plan[t] = (has, repro, has_set, idx, ds)
        print(f"  {t}: {len(has)} of {len(docs[t])} documents contain a repeated line; {len(repro)} unchanged documents "
              f"re-scored; masked inputs built ({time.time() - t0:.0f}s)", flush=True)

    report = {}
    for spec in [x for x in a.checkpoints.split(",") if x.strip()]:
        name, rest = spec.split(":", 1)
        ckpt, thr = rest.rsplit(":", 1)
        thr = float(thr)
        saved = np.load(Path("checkpoints") / name / "test_predictions.npz")
        model = load_model(ckpt)
        report[name] = {}
        print(f"\n{name} ({ckpt}, threshold {thr})")
        for t in TIERS:
            D = docs[t]
            lengths = saved[f"{t}__lengths"].astype(int); off = np.concatenate([[0], np.cumsum(lengths)])
            y = saved[f"{t}__labels"].astype(int); p_saved = saved[f"{t}__preds"].astype(int)
            pr_saved = saved[f"{t}__probs"].astype(float)
            assert len(lengths) == len(D) and all(lengths[i] == min(len(D[i].sentences) - 1, len(D[i].boundary_labels))
                                                    for i in range(len(D))), f"{t}: saved predictions do not line up"
            near = np.zeros(len(y), bool); edge = np.zeros(len(y), int); inside = np.zeros(len(y), bool)
            inside_diff = np.zeros(len(y), bool)
            for i, d in enumerate(D):
                r = np.array([s in rep_lines for s in d.sentences])
                for j in range(lengths[i]):
                    near[off[i] + j] = r[j] or r[j + 1]
                    edge[off[i] + j] = int(r[j] != r[j + 1])
                    inside[off[i] + j] = r[j] and r[j + 1]
                    inside_diff[off[i] + j] = r[j] and r[j + 1] and d.sentences[j] != d.sentences[j + 1]
            one = edge.astype(bool)
            has, repro, has_set, idx, ds = plan[t]
            preds, probs, _, lens = predict_tier(model, ds, tokenizer, DEVICE, thr, eval_args)
            p_mask = p_saved.copy(); k = 0; agree, maxdiff = [], []
            for i, n in zip(idx, lens):
                seg = slice(off[i], off[i] + n)
                if i in has_set:
                    p_mask[seg] = preds[k:k + n]
                else:
                    agree.append(preds[k:k + n] == p_saved[seg])
                    maxdiff.append(float(np.max(np.abs(probs[k:k + n] - pr_saved[seg]))))
                k += n
            chk = (float(np.mean(np.concatenate(agree))), max(maxdiff)) if agree else None
            m1 = lambda mask, p: macro(y[mask], p[mask]) if mask.any() else float("nan")
            r = {"near_share": float(near.mean()), "edge_share": float(one.mean()), "inside_share": float(inside.mean()),
                 "edge_unmasked": m1(one, p_saved), "edge_masked": m1(one, p_mask),
                 "inside_unmasked": m1(inside, p_saved), "inside_masked": m1(inside, p_mask),
                 "inside_changes_between_different_lines": int(np.sum(y[inside_diff] == 1)),
                 "near_unmasked": macro(y[near], p_saved[near]), "near_masked": macro(y[near], p_mask[near]),
                 "elsewhere_unmasked": macro(y[~near], p_saved[~near]), "elsewhere_masked": macro(y[~near], p_mask[~near]),
                 "all_unmasked": macro(y, p_saved), "all_masked": macro(y, p_mask),
                 "edge_rule_near": macro(y[near], edge[near]) if near.any() else float("nan"),
                 "repro_pred_agreement": chk[0] if chk else None, "repro_max_prob_diff": chk[1] if chk else None}
            report[name][t] = r
            print(f"  {t:6s} near repeated ({r['near_share']:.1%}): unmasked {r['near_unmasked']:.4f} -> masked {r['near_masked']:.4f} "
                  f"(edge rule {r['edge_rule_near']:.4f}) | elsewhere {r['elsewhere_unmasked']:.4f} -> {r['elsewhere_masked']:.4f} "
                  f"| all {r['all_unmasked']:.4f} -> {r['all_masked']:.4f}"
                  + (f" | unchanged docs re-scored: prediction agreement {chk[0]:.4f}, max prob diff {chk[1]:.4f}" if chk else ""))
            print(f"         EDGE ({r['edge_share']:.1%} of boundaries; pre-registered subset): {r['edge_unmasked']:.4f} -> "
                  f"{r['edge_masked']:.4f} | INSIDE ({r['inside_share']:.1%}; bounded by the masking design, "
                  f"{r['inside_changes_between_different_lines']} changes between two different repeated lines): "
                  f"{r['inside_unmasked']:.4f} -> {r['inside_masked']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {a.out} ({time.time() - t0:.0f}s). 'elsewhere' changes only in documents that contain a repeated line.")


if __name__ == "__main__":
    main()
