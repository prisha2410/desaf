#!/usr/bin/env python3
"""
Bridge between the construction experiment and the detector: run the three trained DeSAF encoders (no
retraining) on the documents built by construction_experiment.py, and score them with the official metric.
Question: does a PAN-trained detector do worse when documents are assembled the PAN-2025-Hard way
(topic_style / style_splice) than when they are assembled at random, on the same text?

Use C50 constructions (text DeSAF never saw) and Reddit constructions built from the PAN 2025 TEST split
(build them first with --source-split test --name reddit2025test; the train-split Reddit documents contain
text DeSAF was trained on and must not be used here).

For each encoder and construction: ROC-AUC, pooled two-class macro-F1 at the encoder's PAN threshold, macro-F1
with one threshold chosen on that set's 'random' construction (applied unchanged to the others; on 'random'
itself this is in-sample), and the predict-0 baseline. Reuses c50_multiauthor_test.run_checkpoint (GPU).

Run from D:\\sass-tad:
    python scripts/desaf_on_constructions.py --sets c50,reddit2025test
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

KINDS = ["random", "same_topic", "content_splice", "length_splice", "style_splice", "topic_style"]
DEFAULT_CKPTS = ("checkpoints/curriculum_228_2025only_contrastive_seed43.pt:0.50:seed43_gate,"
                 "checkpoints/curriculum_228_2025only_contrastive_seed44.pt:0.52:seed44_gate,"
                 "checkpoints/curriculum_228_2025only_contrastive_seed45.pt:0.64:seed45_gate")
GRID = np.unique(np.concatenate([np.arange(0.0005, 0.02, 0.0005), np.arange(0.02, 0.905, 0.01)]))


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def roc_auc(probs, y):
    from scipy.stats import rankdata
    r = rankdata(probs)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructions-dir", default="results/construction")
    ap.add_argument("--sets", default="c50,reddit2025test")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--checkpoints", default=DEFAULT_CKPTS, help="path:threshold:name, comma separated")
    ap.add_argument("--out-dir", default="results/construction_desaf")
    a = ap.parse_args()

    ckpts = []
    for spec in [s for s in a.checkpoints.split(",") if s.strip()]:
        path, thr, name = spec.split(":") if spec.count(":") == 2 else (spec.rsplit(":", 2))
        if not Path(path).exists():
            hits = sorted(Path("checkpoints").glob(f"*{name.replace('_gate', '')}*.pt"))
            raise SystemExit(f"Checkpoint not found: {path}\n  candidates: {[str(h) for h in hits]}\n"
                             "  pass --checkpoints path:threshold:name,... with the gated encoders' real paths")
        ckpts.append((path, float(thr), name))

    docs_by_regime = {}
    for s in [x.strip() for x in a.sets.split(",") if x.strip()]:
        for k in [x.strip() for x in a.kinds.split(",") if x.strip()]:
            f = Path(a.constructions_dir) / f"{s}_{k}.jsonl"
            if not f.exists():
                print(f"(missing {f}; skipped)")
                continue
            docs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            docs_by_regime[f"{s}__{k}"] = [{"doc_id": f"{s}-{k}-{i}", "sentences": d["sentences"], "labels": d["labels"]}
                                           for i, d in enumerate(docs)]
    if not docs_by_regime:
        raise SystemExit("No construction files found.")
    print(f"{len(docs_by_regime)} constructions: {', '.join(docs_by_regime)}")

    from c50_multiauthor_test import run_checkpoint
    for path, thr, name in ckpts:
        done = all((Path(a.out_dir) / f"{name}_{r}.npz").exists() for r in docs_by_regime)
        if done:
            print(f"{name}: all constructions already scored (delete {a.out_dir} files to redo)")
            continue
        args = SimpleNamespace(checkpoint=path, threshold=thr, name=name, out_dir=a.out_dir, attention_fusion=True,
                               film_gate=False, bilstm=True, bilstm_hidden=128, unfreeze_layers=2, dropout=0.3,
                               max_length=256, sub_batch=8, no_char_ngrams=True, contrastive=True)
        todo = {r: d for r, d in docs_by_regime.items() if not (Path(a.out_dir) / f"{name}_{r}.npz").exists()}
        print(f"\n{name}: scoring {len(todo)} constructions with {path} (threshold {thr})")
        run_checkpoint(args, todo)

    report = {}
    print("\n" + "=" * 118)
    print("DeSAF on constructed documents (no retraining). macro = pooled two-class macro-F1 (official metric)")
    print("=" * 118)
    for s in sorted({r.split("__")[0] for r in docs_by_regime}):
        print(f"\n{s}:  {'construction':16s}{'chg':>6s}{'pred0':>7s}" + "".join(
            f"{n + ' AUC':>16s}{'@PAN':>8s}{'@rand':>8s}" for _, _, n in ckpts) + f"{'mean AUC':>10s}{'mean @PAN':>11s}{'mean @rand':>11s}")
        rand_thr = {}
        for _, _, n in ckpts:
            f = Path(a.out_dir) / f"{n}_{s}__random.npz"
            if f.exists():
                z = np.load(f)
                pr, y = z["probs"].astype(np.float64), z["labels"].astype(int)
                rand_thr[n] = float(GRID[int(np.argmax([macro(y, (pr >= t).astype(int)) for t in GRID]))])
        report[s] = {"random_thresholds": rand_thr, "constructions": {}}
        for k in KINDS:
            r = f"{s}__{k}"
            if r not in docs_by_regime:
                continue
            row, aucs, pans, rands = {}, [], [], []
            for path, thr, n in ckpts:
                z = np.load(Path(a.out_dir) / f"{n}_{r}.npz")
                pr, y = z["probs"].astype(np.float64), z["labels"].astype(int)
                au, mp = roc_auc(pr, y), macro(y, (pr >= thr).astype(int))
                mr = macro(y, (pr >= rand_thr[n]).astype(int)) if n in rand_thr else float("nan")
                row[n] = {"auc": au, "macro_pan": mp, "macro_random_thr": mr}
                aucs.append(au); pans.append(mp); rands.append(mr)
            p = y.mean()
            print(f"{'':{len(s) + 3}s}{k:16s}{p:6.3f}{(1 - p) / (2 - p):7.3f}" + "".join(
                f"{row[n]['auc']:16.4f}{row[n]['macro_pan']:8.4f}{row[n]['macro_random_thr']:8.4f}" for _, _, n in ckpts)
                  + f"{np.mean(aucs):10.4f}{np.mean(pans):11.4f}{np.nanmean(rands):11.4f}")
            report[s]["constructions"][k] = {"change_rate": p, "predict0": (1 - p) / (2 - p), "per_encoder": row,
                                             "mean_auc": float(np.mean(aucs)), "mean_macro_pan": float(np.mean(pans)),
                                             "mean_macro_random_thr": float(np.nanmean(rands))}
    print("\n(@rand on 'random' itself is in-sample; on the other constructions it is a fixed, transferred threshold)")
    out = Path(a.out_dir) / "summary_official.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
