#!/usr/bin/env python3
"""
Significance tests under the OFFICIAL PAN 2025 metric (pooled two-class macro-F1), with positive-class F1
reported alongside. Uses only saved test predictions: no model, no GPU, no new test evaluation.

Per comparison and tier: paired bootstrap over documents (both models resampled with the same documents),
95% CI of the difference, two-sided p; McNemar on per-boundary correctness. Holm correction is applied
within each family of comparisons (to the macro-F1 bootstrap p-values).

Run from D:\\sass-tad:
    python scripts/significance_official.py                    (all pre-defined paper comparisons)
    python scripts/significance_official.py --a X.npz --b Y.npz --a-name A --b-name B   (one pair)
Writes results\\significance_official.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2

TIERS = ["easy", "medium", "hard"]
SEEDS = [43, 44, 45]
HEADS = [1, 2, 3]
GL = "results/gate_lab"


def families():
    f = {}
    f["gate vs concat (end-to-end, per seed)"] = [
        (f"checkpoints/seed{s}/test_predictions.npz", f"checkpoints/concat_seed{s}/test_predictions.npz",
         f"gate{s}", f"concat{s}") for s in SEEDS]
    f["tier head vs original (oracle tier)"] = [
        (f"{GL}/seed{s}_gate/final_concat_tier_s{h}_test.npz", f"{GL}/seed{s}_gate/final_original_s0_test.npz",
         f"tier{s}h{h}", f"orig{s}") for s in SEEDS for h in HEADS]
    f["tier head vs same head without flag"] = [
        (f"{GL}/seed{s}_gate/final_concat_tier_s{h}_test.npz", f"{GL}/seed{s}_gate/final_concat_s{h}_test.npz",
         f"tier{s}h{h}", f"noflag{s}h{h}") for s in SEEDS for h in HEADS]
    f["predicted tier (soft) vs original"] = [
        (f"{GL}/seed{s}_gate/predicted_tier_soft/final_concat_tier_s{h}_test_predtier.npz",
         f"{GL}/seed{s}_gate/final_original_s0_test.npz", f"soft{s}h{h}", f"orig{s}") for s in SEEDS for h in HEADS]
    return f


def doc_counts(preds, labels, lengths):
    ends = np.cumsum(lengths)
    starts = ends - lengths
    tp = np.add.reduceat(((preds == 1) & (labels == 1)).astype(np.int64), starts)
    fp = np.add.reduceat(((preds == 1) & (labels == 0)).astype(np.int64), starts)
    fn = np.add.reduceat(((preds == 0) & (labels == 1)).astype(np.int64), starts)
    tn = lengths - tp - fp - fn
    return np.stack([tp, fp, fn, tn])          # (4, n_docs)


def f1s(c):
    """c: (4, ...) summed counts -> (positive-class F1, two-class macro F1); zero_division = 0 like sklearn."""
    tp, fp, fn, tn = c
    d1 = 2 * tp + fp + fn
    d0 = 2 * tn + fn + fp
    f1_pos = np.where(d1 > 0, 2 * tp / np.maximum(d1, 1), 0.0)
    f1_neg = np.where(d0 > 0, 2 * tn / np.maximum(d0, 1), 0.0)
    return f1_pos, (f1_pos + f1_neg) / 2


def bootstrap(ca, cb, n_boot, seed, chunk=500):
    rng = np.random.default_rng(seed)
    n = ca.shape[1]
    d_pos, d_mac = [], []
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        sa = ca[:, idx].sum(axis=2)            # (4, k)
        sb = cb[:, idx].sum(axis=2)
        pa, ma = f1s(sa)
        pb, mb = f1s(sb)
        d_pos.append(pa - pb)
        d_mac.append(ma - mb)
    out = {}
    for name, d in (("pos", np.concatenate(d_pos)), ("macro", np.concatenate(d_mac))):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[name] = (float(lo), float(hi), float(p))
    return out


def mcnemar(pa, pb, y):
    ca, cb = pa == y, pb == y
    n01, n10 = int(np.sum(~ca & cb)), int(np.sum(ca & ~cb))
    if n01 + n10 == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1 - chi2.cdf(stat, df=1))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(fa, fb, n_boot, seed):
    za, zb = np.load(fa), np.load(fb)
    rows = []
    for t in TIERS:
        ya, yb = za[f"{t}__labels"].astype(int), zb[f"{t}__labels"].astype(int)
        la, lb = za[f"{t}__lengths"].astype(int), zb[f"{t}__lengths"].astype(int)
        if len(ya) != len(yb) or not np.array_equal(ya, yb) or not np.array_equal(la, lb):
            raise SystemExit(f"{fa} and {fb} do not share the same {t} test labels/documents")
        pa, pb = za[f"{t}__preds"].astype(int), zb[f"{t}__preds"].astype(int)
        ca, cb = doc_counts(pa, ya, la), doc_counts(pb, yb, lb)
        fa_pos, fa_mac = f1s(ca.sum(axis=1))
        fb_pos, fb_mac = f1s(cb.sum(axis=1))
        bs = bootstrap(ca, cb, n_boot, seed)
        rows.append({"tier": t, "a_macro": float(fa_mac), "b_macro": float(fb_mac),
                     "d_macro": float(fa_mac - fb_mac), "ci_macro": bs["macro"][:2], "p_macro": bs["macro"][2],
                     "a_pos": float(fa_pos), "b_pos": float(fb_pos), "d_pos": float(fa_pos - fb_pos),
                     "p_pos": bs["pos"][2], "p_mcnemar": mcnemar(pa, pb, ya)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fams = {"single comparison": [(args.a, args.b, args.a_name, args.b_name)]} if args.a and args.b else families()
    out_rows = []
    for fam, pairs in fams.items():
        avail = [p for p in pairs if Path(p[0]).exists() and Path(p[1]).exists()]
        missing = [p for p in pairs if p not in avail]
        print("=" * 118)
        print(f"{fam}: {len(avail)} of {len(pairs)} comparisons found"
              + (f"; missing e.g. {missing[0][0] if not Path(missing[0][0]).exists() else missing[0][1]}" if missing else ""))
        print("=" * 118)
        if not avail:
            continue
        res = []
        for fa, fb, na, nb in avail:
            for r in compare(fa, fb, args.n_boot, args.seed):
                r.update({"family": fam, "a": na, "b": nb})
                res.append(r)
        adj = holm([r["p_macro"] for r in res])
        for r, a in zip(res, adj):
            r["p_macro_holm"] = float(a)
        print(f"{'A vs B':26s} {'tier':6s} {'macro A':>8s} {'macro B':>8s} {'delta':>8s} {'95% CI':>18s} {'p':>7s} "
              f"{'p Holm':>7s} | {'pos delta':>9s} {'p':>7s} | {'McNemar p':>9s}")
        for r in res:
            print(f"{r['a'] + ' vs ' + r['b']:26s} {r['tier']:6s} {r['a_macro']:8.4f} {r['b_macro']:8.4f} "
                  f"{r['d_macro']:+8.4f} [{r['ci_macro'][0]:+.4f}, {r['ci_macro'][1]:+.4f}] {r['p_macro']:7.4f} "
                  f"{r['p_macro_holm']:7.4f} | {r['d_pos']:+9.4f} {r['p_pos']:7.4f} | {r['p_mcnemar']:9.4f}")
        sig = [r for r in res if r["p_macro_holm"] < 0.05]
        pos = sum(r["d_macro"] > 0 for r in res)
        print(f"\nSUMMARY {fam}: A better in {pos}/{len(res)} tier-comparisons (official metric); "
              f"significant after Holm: {len(sig)}/{len(res)} "
              f"({sum(r['d_macro'] > 0 for r in sig)} favour A, {sum(r['d_macro'] < 0 for r in sig)} favour B)\n")
        out_rows += res
    out = Path("results") / "significance_official.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["family", "a", "b", "tier", "a_macro", "b_macro", "d_macro", "ci_macro", "p_macro", "p_macro_holm",
            "a_pos", "b_pos", "d_pos", "p_pos", "p_mcnemar"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in out_rows:
            w.writerow([r[k] if k != "ci_macro" else f"[{r[k][0]:.4f}, {r[k][1]:.4f}]" for k in keys])
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
