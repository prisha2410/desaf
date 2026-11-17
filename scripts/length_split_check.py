#!/usr/bin/env python3
"""
Which part of sentence length carries the PAN 2025 Hard reversal: WHERE changes sit (absolute length: the shorter
sentence's word count) or HOW ALIKE the two sentences are (the word-count difference)?  CPU only.

For each boundary: wa, wb = word counts (str.split), wmin = min(wa, wb), dw = |wa - wb|,
loglen = |log(wa / wb)| (word counts floored at 1). Groupings (each capped at 40):
  G_min   = wmin only          (compares changes and no-changes that sit next to equally long sentences)
  G_diff  = dw only            (compares pairs with the same length difference)
  G_exact = (wmin, dw)         (the earlier "exact length" cell; reference)
AUC inside groups, weighted by n_changes * n_no-changes per group (ties = 1/2); 95% CIs from document resampling.
Cues: |log length ratio| (> 0.5 = changes join sentences of MORE different length), the shorter sentence's word count
("longer at changes" > 0.5), fw_dist, trigram_dissim, punct_dist (the repo's surface_distances), and style_L2 when the
saved per-boundary features of pan_feature_direction.py line up with the text (checked, else skipped).
Also: change rate by the shorter sentence's word count (1-3, 4-7, 8-11, 12-19, 20-29, 30+).

Pre-registered (completeness critic, 2026-11-09), s = (AUC of |log length ratio| under G_min - unmatched AUC) /
(0.5 - unmatched AUC):  s >= 0.6 -> "Hard changes avoid short sentences; that placement, not length matching, makes
them look alike in length";  s <= 0.3 -> "changes join sentences of similar length beyond placement";  else "both".
A cue's driver is named only if one grouping alone brings it within 0.02 of its exact-length value and the other
does not, and only if the test split gives the same CI class.

Run from D:\\sass-tad:
    python scripts/length_split_check.py --pan2026-root "D:\\mawsa26-pan-zenodo"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pan_feature_direction import load_split, surface_distances  # noqa: E402

TIERS = ["easy", "medium", "hard"]
BINS = [(1, 3), (4, 7), (8, 11), (12, 19), (20, 29), (30, 10 ** 9)]


def grouped_auc(x, y, g):
    """P(change > no-change) for pairs in the same group, ties 1/2, weighted by n1*n0 per group."""
    x = np.asarray(x, float); y = np.asarray(y, int); g = np.asarray(g)
    o = np.lexsort((x, g)); xs, ys, gs = x[o], y[o], g[o]
    n = len(xs)
    if n == 0:
        return float("nan")
    newg = np.r_[True, gs[1:] != gs[:-1]]
    gstart = np.maximum.accumulate(np.where(newg, np.arange(n), 0))
    pos = np.arange(n) - gstart + 1.0                                  # 1-based rank inside the group
    newrun = newg | np.r_[True, xs[1:] != xs[:-1]]
    rid = np.cumsum(newrun) - 1
    rsum = np.bincount(rid, weights=pos); rcnt = np.bincount(rid)
    rank = (rsum / rcnt)[rid]                                          # average rank for ties
    gid = np.cumsum(newg) - 1
    n1 = np.bincount(gid, weights=ys); nt = np.bincount(gid).astype(float); n0 = nt - n1
    r1 = np.bincount(gid, weights=rank * ys)
    w = n1 * n0
    if w.sum() == 0:
        return float("nan")
    u = r1 - n1 * (n1 + 1) / 2
    return float(u[w > 0].sum() / w[w > 0].sum())


def boot(fn, doc, n_boot, rng):
    docs = np.unique(doc)
    idx_by_doc = {d: np.flatnonzero(doc == d) for d in docs}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(docs, len(docs), replace=True)
        out.append(fn(np.concatenate([idx_by_doc[d] for d in pick])))
    out = np.array(out, float)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def boundaries(docs):
    rows = {"a": [], "b": [], "y": [], "doc": []}
    for di, (s, lab) in enumerate(docs):
        for i, l in enumerate(lab):
            rows["a"].append(s[i]); rows["b"].append(s[i + 1]); rows["y"].append(int(l)); rows["doc"].append(di)
    return rows


def analyse(label, docs, n_boot, rng, saved=None, surf=True):
    R = boundaries(docs)
    y, doc = np.array(R["y"]), np.array(R["doc"])
    if len(y) == 0 or y.sum() in (0, len(y)):
        print(f"  {label}: not enough boundaries"); return None
    wa = np.array([max(len(t.split()), 1) for t in R["a"]], float)
    wb = np.array([max(len(t.split()), 1) for t in R["b"]], float)
    wmin, dw = np.minimum(wa, wb), np.abs(wa - wb)
    loglen = np.abs(np.log(wa / wb))
    cues = {"|log len ratio|": loglen, "shorter sent. longer": wmin}
    if surf:
        S = np.array([surface_distances(a, b) for a, b in zip(R["a"], R["b"])])
        cues.update({"fw_dist": S[:, 0], "trigram": S[:, 1], "punct": S[:, 2]})
        if saved is not None:
            z = np.load(saved)
            F = z["features"]; names = [str(n) for n in z["feature_names"]]
            ok = F.shape[0] == len(y) and np.array_equal(z["labels"].astype(int), y) and \
                np.allclose(F[:, names.index("fw_dist")], S[:, 0], atol=1e-5) if "fw_dist" in names else False
            sl = [n for n in names if n.startswith("style_L2")]
            if ok and sl:
                cues["style_L2"] = F[:, names.index(sl[0])]
            else:
                print(f"  ({label}: saved features {saved.name} do not line up with this text; style_L2 skipped)")
    G = {"unmatched": np.zeros(len(y), int), "G_min": np.minimum(wmin, 40).astype(int),
         "G_diff": np.minimum(dw, 40).astype(int)}
    G["G_exact"] = G["G_min"] * 100 + G["G_diff"]
    names = list(cues)
    X = np.column_stack([cues[n] for n in names])
    res = {"n": int(len(y)), "changes": int(y.sum()), "cues": {}}
    print(f"\n  {label}  n={len(y)}  changes={int(y.sum())}  (rate {y.mean():.3f})")
    print(f"    {'cue':22s}" + "".join(f"{k:>26s}" for k in G))
    for j, n in enumerate(names):
        row = {}
        cells = []
        for k, g in G.items():
            if (n == "shorter sent. longer" and k in ("G_min", "G_exact")) or (n == "|log len ratio|" and k == "G_exact"):
                row[k] = None; cells.append(f"{'(fixed by grouping)':>26s}"); continue
            a = grouped_auc(X[:, j], y, g)
            lo, hi = boot(lambda i: grouped_auc(X[i, j], y[i], g[i]), doc, n_boot, rng)
            row[k] = {"auc": a, "ci": [float(lo), float(hi)]}
            tagc = "N" if lo > 0.5 else ("R" if hi < 0.5 else "~")
            cells.append(f"{a:7.3f} {tagc} [{lo:.3f}, {hi:.3f}]".rjust(26))
        res["cues"][n] = row
        print(f"    {n:22s}" + "".join(cells))
    rate = []
    for lo_, hi_ in BINS:
        m = (wmin >= lo_) & (wmin <= hi_)
        rate.append({"bin": f"{lo_}-{hi_ if hi_ < 10 ** 9 else ''}", "n": int(m.sum()),
                     "change_rate": float(y[m].mean()) if m.any() else None})
    res["change_rate_by_shorter_len"] = rate
    print("    change rate by shorter sentence's words: " + "  ".join(
        f"{r['bin']}: {r['change_rate']:.3f} (n {r['n']})" if r["change_rate"] is not None else f"{r['bin']}: -"
        for r in rate))
    short = wmin < 8
    if short.any() and (~short).any():
        res["short_vs_long_rate_ratio"] = float(y[short].mean() / max(y[~short].mean(), 1e-9))
        print(f"    change rate, pair with a sentence < 8 words vs both >= 8: {y[short].mean():.3f} vs {y[~short].mean():.3f}"
              f" (ratio {res['short_vs_long_rate_ratio']:.2f})")
    u = res["cues"]["|log len ratio|"]["unmatched"]["auc"]; gm = res["cues"]["|log len ratio|"]["G_min"]["auc"]
    if u < 0.45:  # verdict only where there is a reversal to explain (|AUC - 0.5| >= 0.05)
        s = (gm - u) / (0.5 - u)
        res["s_placement_share"] = s
        verdict = "placement (absolute length)" if s >= 0.6 else ("similar length beyond placement" if s <= 0.3 else "both")
        print(f"    |log len ratio|: unmatched {u:.3f} -> G_min {gm:.3f}; s = {s:.2f} -> {verdict}")
    return res


def read_jsonl(path):
    return [(d["sentences"], d["labels"]) for d in
            (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan2025-root", default="data/raw/pan2025_full")
    ap.add_argument("--pan2026-root", default="")
    ap.add_argument("--pan2026-max-docs", type=int, default=900, help="Hard documents sampled from PAN 2026 validation")
    ap.add_argument("--features-dir", default="results/feature_direction")
    ap.add_argument("--constructions-dir", default="results/construction")
    ap.add_argument("--constructions", default="reddit2025_random,reddit2025_content_splice,reddit2025_topic_style,"
                    "reddit2025_hardonly_random,reddit2025_hardonly_topic_style,c50_random,c50_topic_style")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/feature_direction/length_split.json")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    t0 = time.time()
    report = {}
    print("=" * 140)
    print("LENGTH SPLIT: AUC unmatched / within shorter-sentence length (G_min) / within length difference (G_diff) /"
          " within both (G_exact). N > 0.5, R < 0.5, ~ CI spans 0.5")
    print("=" * 140)
    for split, saved_name in (("validation", "pan2025"), ("test", "pan2025test")):
        for t in TIERS:
            docs, mism = load_split(a.pan2025_root, t, split, 0, rng)
            if not docs:
                print(f"  PAN 2025 {t} {split}: no documents found under {a.pan2025_root}"); continue
            saved = Path(a.features_dir) / f"{saved_name}_{t}.npz"
            report[f"pan2025_{split}_{t}"] = analyse(f"PAN 2025 {t.upper()} {split}", docs, a.boot, rng,
                                                     saved if saved.exists() else None)
        print(f"  ({split} done, {time.time() - t0:.0f}s)")
    if a.pan2026_root:
        docs, _ = load_split(a.pan2026_root, "hard", "validation", a.pan2026_max_docs, np.random.RandomState(a.seed))
        if docs:
            report["pan2026_validation_hard"] = analyse(f"PAN 2026 HARD validation ({len(docs)} sampled documents)",
                                                        docs, a.boot, rng, None)
        print(f"  (PAN 2026 done, {time.time() - t0:.0f}s)")
    for c in [x.strip() for x in a.constructions.split(",") if x.strip()]:
        f = Path(a.constructions_dir) / f"{c}.jsonl"
        if not f.exists():
            print(f"  (missing {f}; skipped)"); continue
        report[f"construction_{c}"] = analyse(f"construction {c}", read_jsonl(f), a.boot, rng, None)
    print(f"\n(all done, {time.time() - t0:.0f}s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
