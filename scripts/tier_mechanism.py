#!/usr/bin/env python3
"""
Two questions about the oracle tier flag, answered on the saved stage-2 caches (minutes on the GPU):

  1. WHERE does the flag's gain come from?  (validation 5-fold CV only; the test set is not used)
     Flag gain = (head with flag) - (same head without flag), for three inputs:
       DeBERTa [CLS] only      cls_only        vs cls_tier
       stylometric diff only   style_only      vs style_only_tier   ([CLS] zeroed)
       both (the paper's head) concat          vs concat_tier
     If the gain is large with stylometric input only and small with [CLS] only, the flag works by switching
     how the style cues count (their direction), as the feature-direction analysis suggests. If it is as large
     with [CLS] only, it does not need the style cues.

  2. HOW ACCURATE must a tier guess be? (analysis of the already-reported concat_tier test heads)
     The flag is per document and the head reads one document at a time, so each test document's predictions
     are computed once under each of the three flags; any pattern of wrong flags is then exact and free.
       - forced-flag matrix: every document of tier X scored as if it were tier Y
       - accuracy curve: each document keeps its true flag with probability a, otherwise gets one of the
         other two at random (100 draws); break-even accuracy vs the no-flag head and the original model
     Heads are retrained exactly as gate_lab.py --final does (same split, seeds 7777+s); the script checks that
     the true-flag predictions reproduce the saved final_concat_tier_s*_test.npz files.

Official metric throughout: pooled two-class macro-F1 per tier, averaged over tiers ("avg").
Run from D:\\sass-tad:
    python scripts/tier_mechanism.py --names seed43_gate,seed44_gate,seed45_gate
Forced-flag ROC-AUC only (Part 2, GPU minutes; keeps the earlier JSON):
    python scripts/tier_mechanism.py --skip-part1 --out results/tier_mechanism_forced_auc.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_lab as gl  # noqa: E402

gl.VARIANTS.setdefault("cls_tier", dict(style=False, norm=None, gate=None, act=None, overlap=False, tier=True))


def macro_pooled(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def roc_auc(prob, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(prob)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tier_macro(y, p, pair_tier):
    s = [macro_pooled(y[pair_tier == i], p[pair_tier == i]) for i in range(3)]
    return s + [float(np.mean(s))]


def gl_args(name, extra):
    argv = sys.argv
    sys.argv = ["gate_lab.py", "--name", name] + extra
    try:
        a = gl.parse_args()
    finally:
        sys.argv = argv
    return a


def cv_macro(out_dir, v, s, data):
    z = np.load(out_dir / f"cv_{v}_s{s}.npz")
    preds = np.zeros(len(data.labels), np.int64)
    for i, t in enumerate(gl.TIERS):
        preds[data.pair_tier == i] = z[f"{t}__preds"]
    return tier_macro(data.labels, preds, data.pair_tier)


def part1(name, args, device, seeds, report):
    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    val = gl.SplitData(cache / f"{name}_validation.npz", device)
    print(f"\n[{name}] Part 1: CV on validation ({val.n_docs} docs)")
    gl.run_cv(val, ["cls_only", "cls_tier", "concat", "concat_tier"], seeds, args, out_dir)
    noh = gl.SplitData(cache / f"{name}_validation.npz", device)
    noh.feats["h"].zero_()
    noh_dir = out_dir / "style_only"
    noh_dir.mkdir(parents=True, exist_ok=True)
    gl.run_cv(noh, ["concat", "concat_tier"], seeds, args, noh_dir)
    res = {}
    for label, d, data, a, b in (("[CLS] only", out_dir, val, "cls_only", "cls_tier"),
                                 ("stylometric only", noh_dir, noh, "concat", "concat_tier"),
                                 ("both (paper head)", out_dir, val, "concat", "concat_tier")):
        A = np.array([cv_macro(d, a, s, data) for s in seeds])
        B = np.array([cv_macro(d, b, s, data) for s in seeds])
        res[label] = {"no_flag": A.mean(0).tolist(), "flag": B.mean(0).tolist(), "gain": (B - A).mean(0).tolist(),
                      "gain_per_seed_avg": (B - A)[:, 3].tolist()}
    report[name]["part1"] = res
    print(f"\n[{name}] flag gain in CV, official metric (easy / medium / hard / avg; mean over head seeds {seeds}):")
    for label, r in res.items():
        print(f"  {label:20s} without flag {fmt4(r['no_flag'])} | with flag {fmt4(r['flag'])} | gain {fmt4(r['gain'], True)}")


def fmt4(v, signed=False):
    return " / ".join((f"{x:+.4f}" if signed else f"{x:.4f}") for x in v)


def part2(name, args, device, seeds, report, draws, rng):
    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir) / name
    val = gl.SplitData(cache / f"{name}_validation.npz", device)
    test = gl.SplitData(cache / f"{name}_test.npz", device)
    split_rng = np.random.RandomState(args.split_seed)       # exactly as gate_lab.run_final
    dev_mask = np.zeros(val.n_docs, dtype=bool)
    for t in range(3):
        idx = np.where(val.doc_tier == t)[0]
        idx = idx[split_rng.permutation(len(idx))]
        dev_mask[idx[: int(round(0.15 * len(idx)))]] = True
    train, dev = np.where(~dev_mask)[0], np.where(dev_mask)[0]
    all_test = np.arange(test.n_docs)
    y, pt = test.labels, test.pair_tier
    doc_of_pair = np.repeat(np.arange(test.n_docs), test.lengths)
    true_t = test.feats["t"]
    refs = {}
    for v in ("original", "concat"):
        files = [out_dir / f"final_{v}_s{s}_test.npz" for s in ([0] if v == "original" else seeds)]
        vals = []
        for f in files:
            if f.exists():
                z = np.load(f)
                p = np.zeros(len(y), np.int64)
                for i, t in enumerate(gl.TIERS):
                    p[pt == i] = z[f"{t}__preds"]
                vals.append(tier_macro(y, p, pt))
        if vals:
            refs[v] = np.mean(vals, axis=0).tolist()
    print(f"\n[{name}] Part 2: test analysis of the concat_tier heads | reference (official avg): "
          + ", ".join(f"{k} {v[3]:.4f}" for k, v in refs.items()))
    out = {"refs": refs, "heads": []}
    acc_grid = [1.0, 0.98, 0.95, 0.92, 0.9, 0.86, 0.82, 0.8, 0.76, 0.7, 0.6, 0.5, 1 / 3]
    for s in seeds:
        model, thr = gl.train_head("concat_tier", val, train, dev, 7777 + s, args)
        P = []
        for f in range(3):
            test.feats["t"] = test.tier_input(np.eye(3, dtype=np.float32)[np.full(test.n_docs, f)])
            P.append(gl.predict(model, test, all_test, args.batch_docs))
        test.feats["t"] = true_t
        P = np.stack(P)                                       # (3 flags, n_pairs)
        true_pred = (P[pt, np.arange(len(y))] >= thr).astype(np.int64)
        saved = out_dir / f"final_concat_tier_s{s}_test.npz"
        repro = None
        if saved.exists():
            z = np.load(saved)
            sp = np.zeros(len(y), np.int64)
            for i, t in enumerate(gl.TIERS):
                sp[pt == i] = z[f"{t}__preds"]
            repro = float((sp == true_pred).mean())
        true_sc = tier_macro(y, true_pred, pt)
        forced = np.zeros((3, 3))
        forced_auc = np.zeros((3, 3))    # threshold-free: ranking within the true tier under each flag
        for f in range(3):
            pf = (P[f] >= thr).astype(np.int64)
            for i in range(3):
                forced[i, f] = macro_pooled(y[pt == i], pf[pt == i])
                forced_auc[i, f] = roc_auc(P[f][pt == i], y[pt == i])
        curve = []
        for a in acc_grid:
            vals = []
            for _ in range(draws):
                keep = rng.rand(test.n_docs) < a
                other = (test.doc_tier + rng.randint(1, 3, test.n_docs)) % 3
                flag = np.where(keep, test.doc_tier, other)
                pred = (P[flag[doc_of_pair], np.arange(len(y))] >= thr).astype(np.int64)
                vals.append(tier_macro(y, pred, pt)[3])
            curve.append((a, float(np.mean(vals)), float(np.std(vals))))
        out["heads"].append({"seed": s, "thr": thr, "reproduced_share": repro, "true": true_sc,
                             "forced": forced.tolist(), "forced_auc": forced_auc.tolist(), "curve": curve})
        print(f"  head seed {s}: thr {thr:.2f}; true-flag official {fmt4(true_sc)}"
              + (f"; matches saved test predictions on {repro:.2%} of boundaries" if repro is not None else ""))
    F = np.mean([h["forced"] for h in out["heads"]], axis=0)
    print(f"\n[{name}] forced-flag matrix (rows = true tier, columns = flag given; official macro-F1, mean over heads):")
    print("            " + "".join(f"{'flag ' + t:>14s}" for t in gl.TIERS))
    for i, t in enumerate(gl.TIERS):
        print(f"  {t:9s} " + "".join(f"{F[i, f]:14.4f}" for f in range(3)))
    C = np.array([[c[1] for c in h["curve"]] for h in out["heads"]]).mean(0)
    print(f"\n[{name}] accuracy curve (avg official macro-F1, mean over heads, {draws} draws each):")
    for a, c in zip(acc_grid, C):
        print(f"  tier accuracy {a:5.3f}: {c:.4f}")
    for k, ref in refs.items():
        above = [a for a, c in zip(acc_grid, C) if c >= ref[3]]
        print(f"  lowest accuracy still at or above {k} ({ref[3]:.4f}): {min(above):.3f}" if above
              else f"  never reaches {k} ({ref[3]:.4f})")
    FA = np.mean([h["forced_auc"] for h in out["heads"]], axis=0)
    print(f"\n[{name}] forced-flag ROC-AUC within the true tier (threshold-free; mean over heads):")
    print("            " + "".join(f"{'flag ' + t:>14s}" for t in gl.TIERS))
    for i, t in enumerate(gl.TIERS):
        print(f"  {t:9s} " + "".join(f"{FA[i, f]:14.4f}" for f in range(3)))
    print("  (a wrong flag that lowers this AUC changes the ranking; one that lowers only macro-F1 mis-sets the threshold)")
    out["forced_auc_mean"] = FA.tolist()
    out["forced_mean"] = F.tolist()
    out["curve_mean"] = list(zip(acc_grid, C.tolist()))
    report[name]["part2"] = out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default="seed43_gate,seed44_gate,seed45_gate")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--skip-part1", action="store_true")
    p.add_argument("--skip-part2", action="store_true")
    p.add_argument("--out", default="results/tier_mechanism.json")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(0)
    report = {}
    for name in [n.strip() for n in a.names.split(",") if n.strip()]:
        args = gl_args(name, ["--seeds", a.seeds])
        report[name] = {}
        if not a.skip_part1:
            part1(name, args, device, seeds, report)
        if not a.skip_part2:
            part2(name, args, device, seeds, report, a.draws, rng)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if len(report) > 1 and not a.skip_part1:
        print("\nACROSS ENCODERS (part 1, avg official macro-F1 flag gain, mean of head seeds):")
        for label in ("[CLS] only", "stylometric only", "both (paper head)"):
            g = [report[n]["part1"][label]["gain"][3] for n in report]
            print(f"  {label:20s} " + "  ".join(f"{x:+.4f}" for x in g) + f"   mean {np.mean(g):+.4f}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()

