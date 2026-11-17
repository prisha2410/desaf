#!/usr/bin/env python3
"""
Closing checks for the feature-direction analysis (after pan_feature_direction.py and pan_direction_followup.py).
No spaCy and no GPU; uses the saved per-boundary files plus the text.

  A. Length and topic matched TOGETHER (the follow-up matched them one at a time): every cue AUC within exact
     length cells, within length x content-overlap cells, and within document x length cells; plus the
     zero-content-overlap bin and the positive-overlap bins reported separately.
  B. Readability split into its length-free parts (syllables per word, complex-word ratio) and its
     length-containing parts (Flesch, Gunning Fog, Coleman-Liau), unmatched and at exact matched length.
  C. Repeated lines: what they are (most frequent lines), how they overlap with short lines, cue AUCs in the
     clean cell (no repeated line, both sentences >= 8 words) incl. the z-scored families when available,
     and sensitivity to the repetition cut-off (3 / 10 / 30 copies).
  D. (PAN 2025 only, optional) How saved DeSAF TEST predictions score on boundaries next to repeated lines vs
     the rest, with the official metric (pooled two-class macro-F1) and each subset's own predict-0 baseline.

Run from D:\\sass-tad with the same --name/--splits as the earlier runs:
    python scripts/pan_direction_checks.py --data-root data/raw/pan2025_full --name pan2025 --desaf-test
    python scripts/pan_direction_checks.py --data-root "D:\\mawsa26-pan-zenodo" --name pan2026
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from pan_direction_followup import (load_split, auc_fast, grouped_auc, doc_index, boot_ci,  # noqa: E402
                                    quantile_bins, tag, ci_txt, clean)

TIERS_DEFAULT = "easy,medium,hard"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--splits", default="validation")
    p.add_argument("--tiers", default=TIERS_DEFAULT)
    p.add_argument("--tfidf-split", default="train")
    p.add_argument("--npz-dir", default="results/feature_direction")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--desaf-test", action="store_true",
                   help="part D: score saved DeSAF test predictions near repeated lines (PAN 2025 layout)")
    p.add_argument("--pred-files", default="checkpoints/seed43/test_predictions.npz,checkpoints/seed44/test_predictions.npz,"
                   "checkpoints/seed45/test_predictions.npz,checkpoints/concat_seed43/test_predictions.npz,"
                   "checkpoints/concat_seed44/test_predictions.npz,checkpoints/concat_seed45/test_predictions.npz")
    return p.parse_args()


def length_cells(minw, dw, cap=40):
    """Exact cells of (shorter sentence's word count, word-count difference), capped at 40 words."""
    return np.minimum(minw, cap).astype(np.int64) * 100 + np.minimum(dw, cap).astype(np.int64)


def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    npz_dir = Path(args.npz_dir)
    t0 = time.time()
    summary = {"name": args.name, "splits": splits, "tiers": {t: {} for t in tiers}}
    out_json = npz_dir / f"{args.name}_checks.json"

    def save():
        out_json.write_text(json.dumps(clean(summary), indent=1), encoding="utf-8")

    # ---------------- load saved features and text (must line up, as in the follow-up)
    saved, texts = {}, {}
    for tier in tiers:
        f = npz_dir / f"{args.name}_{tier}.npz"
        if not f.exists():
            raise SystemExit(f"Missing {f}: run pan_feature_direction.py with --name {args.name} --splits {args.splits} first.")
        z = np.load(f, allow_pickle=False)
        saved[tier] = {k: z[k] for k in z.files}
        docs = [d for split in splits for d in load_split(args.data_root, tier, split)]
        y_txt = np.array([c for _, ch in docs for c in ch], dtype=int)
        doc_txt = np.repeat(np.arange(len(docs)), [len(ch) for _, ch in docs])
        if not (np.array_equal(y_txt, saved[tier]["labels"]) and np.array_equal(doc_txt, saved[tier]["doc"])):
            raise SystemExit(f"{tier}: text does not line up with {f}; use the same --data-root/--splits/--name.")
        texts[tier] = docs
    feat_names = [str(s) for s in saved[tiers[0]]["feature_names"]]
    nf = len(feat_names)

    from sklearn.feature_extraction.text import TfidfVectorizer
    fit = [s for tier in tiers for sents, _ in load_split(args.data_root, tier, args.tfidf_split) for s in sents]
    if not fit:
        fit = [s for tier in tiers for sents, _ in texts[tier] for s in sents]
    vec_c = TfidfVectorizer(max_features=20000, stop_words="english").fit(fit)
    del fit
    freq = Counter(s for tier in tiers for sents, _ in texts[tier] for s in sents)
    per_tier_freq = {tier: Counter(s for sents, _ in texts[tier] for s in sents) for tier in tiers}

    try:
        from src.features.readability import extract_readability_features
        read_names = ["avg_syllables_per_word", "complex_word_ratio", "flesch_reading_ease", "gunning_fog", "coleman_liau"]
    except Exception as ex:
        print(f"(readability module not importable: {ex}; part B skipped)")
        extract_readability_features, read_names = None, []

    extra = {}
    for tier in tiers:
        oc, minw, dw, rep_n, R = [], [], [], [], []
        for sents, ch in texts[tier]:
            T = vec_c.transform(sents)
            oc.append(np.asarray(T[:-1].multiply(T[1:]).sum(axis=1)).ravel())
            w = np.array([len(s.split()) for s in sents], dtype=float)
            minw.append(np.minimum(w[1:], w[:-1]))
            dw.append(np.abs(w[1:] - w[:-1]))
            cnt = np.array([freq[s] for s in sents])
            rep_n.append(np.maximum(cnt[1:], cnt[:-1]))
            if extract_readability_features is not None:
                rv = np.array([[extract_readability_features(s)[n] for n in read_names] for s in sents], dtype=float)
                R.append(np.abs(rv[1:] - rv[:-1]))
        extra[tier] = {"ovl_c": np.concatenate(oc), "min_words": np.concatenate(minw), "d_words": np.concatenate(dw),
                       "rep_count": np.concatenate(rep_n)}
        if R:
            extra[tier]["read"] = np.concatenate(R)
    print(f"Loaded and prepared ({time.time() - t0:.0f}s)\n")

    # ============================================================ A. joint matching
    print("=" * 110)
    print(f"{args.name}  A. Cues with length and topic matched TOGETHER")
    print("exact length = same (shorter sentence's word count, word-count difference) cell, each capped at 40;")
    print("length x topic = exact length cell x content-overlap bin (zero + 5 quantile bins);")
    print("doc x length = same document AND same exact length cell. CIs resample documents.")
    print("=" * 110)
    for tier in tiers:
        v, e = saved[tier], extra[tier]
        X, y, doc = v["features"].astype(np.float64), v["labels"].astype(int), v["doc"]
        groups = doc_index(doc)
        L = length_cells(e["min_words"], e["d_words"])
        B = quantile_bins(e["ovl_c"], 5, True)
        LB = L * 10 + B
        zero = B == 0

        def stat(i, cid):
            out = []
            for j in range(nf):
                x, yy = X[i, j], y[i]
                out += [grouped_auc(x, yy, L[i]), grouped_auc(x, yy, LB[i]),
                        grouped_auc(x, yy, cid * 100000 + L[i])]
            return np.array(out)

        lo, hi = boot_ci(stat, groups, args.boot, rng)
        rows = {"exact length": [], "length x topic": [], "doc x length": []}
        for j in range(nf):
            rows["exact length"].append(grouped_auc(X[:, j], y, L))
            rows["length x topic"].append(grouped_auc(X[:, j], y, LB))
            rows["doc x length"].append(grouped_auc(X[:, j], y, doc.astype(np.int64) * 100000 + L))
        z1 = [auc_fast(X[zero, j], y[zero]) for j in range(nf)]
        z2 = [grouped_auc(X[~zero, j], y[~zero], B[~zero]) for j in range(nf)]
        print(f"\n{tier.upper()}  n={len(y)}; zero content overlap: {zero.mean():.1%} of boundaries "
              f"({zero[y == 1].mean():.1%} of changes)")
        print(f"  {'':24s}" + "".join(f"{n:>22s}" for n in feat_names))
        for k, (label, vals) in enumerate(rows.items()):
            print(f"  {label:24s}" + "".join(f"{tag(a):>22s}" for a in vals))
            print(f"  {'  95% CI':24s}" + "".join(f"{ci_txt(lo[3 * j + k], hi[3 * j + k]):>22s}" for j in range(nf)))
        print(f"  {'zero-overlap bin only':24s}" + "".join(f"{tag(a):>22s}" for a in z1))
        print(f"  {'positive-overlap bins':24s}" + "".join(f"{tag(a):>22s}" for a in z2))
        summary["tiers"][tier]["A"] = {
            lab: {n: {"auc": vals[j], "ci": [lo[3 * j + k], hi[3 * j + k]]} for j, n in enumerate(feat_names)}
            for k, (lab, vals) in enumerate(rows.items())}
        summary["tiers"][tier]["A"]["zero_overlap_only"] = dict(zip(feat_names, z1))
        summary["tiers"][tier]["A"]["positive_overlap_bins"] = dict(zip(feat_names, z2))
        save()
    print(f"\n(part A done, {time.time() - t0:.0f}s)")

    # ============================================================ B. readability split
    if read_names:
        print("\n" + "=" * 110)
        print(f"{args.name}  B. Readability: length-free parts vs parts that contain sentence length")
        print("(syllables per word and complex-word ratio contain no sentence length; Flesch, Fog and Coleman-Liau do)")
        print("=" * 110)
        for tier in tiers:
            e = extra[tier]
            y, doc = saved[tier]["labels"].astype(int), saved[tier]["doc"]
            Rm, L = e["read"], length_cells(e["min_words"], e["d_words"])
            un = [auc_fast(Rm[:, j], y) for j in range(len(read_names))]
            ml = [grouped_auc(Rm[:, j], y, L) for j in range(len(read_names))]
            lo, hi = boot_ci(lambda i, cid: np.array([grouped_auc(Rm[i, j], y[i], L[i]) for j in range(len(read_names))]),
                             doc_index(doc), args.boot, rng)
            print(f"\n{tier.upper()}  {'':18s}" + "".join(f"{n[:21]:>23s}" for n in read_names))
            print(f"  {'unmatched':24s}" + "".join(f"{tag(a):>23s}" for a in un))
            print(f"  {'exact matched length':24s}" + "".join(f"{tag(a):>23s}" for a in ml))
            print(f"  {'  95% CI':24s}" + "".join(f"{ci_txt(l, h):>23s}" for l, h in zip(lo, hi)))
            summary["tiers"][tier]["B"] = {n: {"unmatched": a, "matched_length": b, "ci": [l, h]}
                                           for n, a, b, l, h in zip(read_names, un, ml, lo, hi)}
            save()
        print(f"\n(part B done, {time.time() - t0:.0f}s)")

    # ============================================================ C. repeated lines
    print("\n" + "=" * 110)
    print(f"{args.name}  C. Repeated lines: what they are, overlap with short lines, and the clean cell")
    print("=" * 110)
    top = [(s, c) for s, c in freq.most_common(40) if c >= 3][:25]
    if top:
        print(f"Most frequent lines in the analysed documents (count over all tiers; per tier E/M/H):")
        for s, c in top:
            pt = "/".join(str(per_tier_freq[t][s]) for t in tiers)
            shown = s if len(s) <= 90 else s[:87] + "..."
            print(f"  {c:6d}  ({pt:>14s})  {len(s.split()):3d} words  {shown}")
    else:
        print("No line occurs 3 or more times.")
    for tier in tiers:
        v, e = saved[tier], extra[tier]
        X, y, doc = v["features"].astype(np.float64), v["labels"].astype(int), v["doc"]
        short = e["min_words"] < 8
        print(f"\n{tier.upper()}  cut-off sensitivity (share of changes / no-changes next to a line with >= k copies):")
        sens = {}
        for k in (3, 10, 30):
            rep = e["rep_count"] >= k
            sens[k] = (rep[y == 1].mean(), rep[y == 0].mean())
            print(f"    k={k:2d}: {rep[y == 1].mean():.1%} / {rep[y == 0].mean():.1%}")
        rep = e["rep_count"] >= 10
        print("  cross-table (boundaries, change rate):")
        cells = {}
        for r_lab, r in (("no repeated line", ~rep), ("repeated line", rep)):
            for s_lab, s in (("both >= 8 words", ~short), ("a sentence < 8 words", short)):
                m = r & s
                cells[(r_lab, s_lab)] = m
                print(f"    {r_lab:17s} & {s_lab:21s}: n={int(m.sum()):7d}  change rate "
                      f"{(y[m].mean() if m.any() else float('nan')):.3f}")
        cleanm = cells[("no repeated line", "both >= 8 words")]
        idx_map = np.flatnonzero(cleanm)
        Xc, yc, dc = X[cleanm], y[cleanm], doc[cleanm]
        cl = [auc_fast(Xc[:, j], yc) for j in range(nf)]
        lo, hi = boot_ci(lambda i, cid: np.array([auc_fast(Xc[i, j], yc[i]) for j in range(nf)]), doc_index(dc),
                         args.boot, rng)
        print(f"  clean cell (no repeated line, both >= 8 words), n={int(cleanm.sum())}:")
        print(f"  {'':24s}" + "".join(f"{n:>22s}" for n in feat_names))
        print(f"  {'unmatched':24s}" + "".join(f"{tag(a):>22s}" for a in cl))
        print(f"  {'  95% CI':24s}" + "".join(f"{ci_txt(l, h):>22s}" for l, h in zip(lo, hi)))
        res = {"sensitivity": {str(k): {"changes": a, "no_changes": b} for k, (a, b) in sens.items()},
               "clean_cell_n": int(cleanm.sum()),
               "clean_cell": {n: {"auc": a, "ci": [l, h]} for n, a, l, h in zip(feat_names, cl, lo, hi)}}
        gfile = npz_dir / f"{args.name}_{tier}_groups.npz"
        if gfile.exists():
            g = np.load(gfile, allow_pickle=False)
            gdoc = g["doc"]
            # rows of the groups file are boundaries of the sampled documents, in document order
            starts = np.r_[0, np.cumsum([len(ch) for _, ch in texts[tier]])]
            gidx = np.concatenate([np.arange(starts[d], starts[d + 1]) for d in np.unique(gdoc)]) \
                if len(np.unique(gdoc)) < len(texts[tier]) else np.arange(len(y))
            if len(gidx) == len(g["labels"]) and np.array_equal(y[gidx], g["labels"]):
                keep = cleanm[gidx]
                D, cols = g["dist"].astype(np.float64)[keep], [str(c) for c in g["columns"]]
                gy, gd = g["labels"][keep].astype(int), gdoc[keep]
                ga = [auc_fast(D[:, j], gy) for j in range(D.shape[1])]
                glo, ghi = boot_ci(lambda i, cid: np.array([auc_fast(D[i, j], gy[i]) for j in range(D.shape[1])]),
                                   doc_index(gd), args.boot, rng)
                print(f"  z-scored families in the clean cell (n={int(keep.sum())}, part C documents):")
                print(f"  {'':24s}" + "".join(f"{c[:19]:>20s}" for c in cols))
                print(f"  {'unmatched':24s}" + "".join(f"{tag(a):>20s}" for a in ga))
                print(f"  {'  95% CI':24s}" + "".join(f"{ci_txt(l, h):>20s}" for l, h in zip(glo, ghi)))
                res["clean_cell_groups"] = {c: {"auc": a, "ci": [l, h]} for c, a, l, h in zip(cols, ga, glo, ghi)}
            else:
                print("  (groups file does not line up with the text; z-scored families skipped)")
        summary["tiers"][tier]["C"] = res
        save()
    print(f"\n(part C done, {time.time() - t0:.0f}s)")

    # ============================================================ D. DeSAF test predictions near repeated lines
    if args.desaf_test:
        print("\n" + "=" * 110)
        print("D. Saved DeSAF TEST predictions: official metric next to repeated lines vs elsewhere")
        print("(loads the test text exactly as the training code does: files sorted by name, labels padded/cut)")
        print("=" * 110)
        root = Path(args.data_root)
        test = {}
        for tier in tiers:
            docs = []
            for txt in sorted((root / tier / "test").glob("problem-*.txt")):
                truth = txt.with_name(f"truth-{txt.stem}.json")
                if not truth.exists():
                    continue
                sents = [s.strip() for s in txt.read_text(encoding="utf-8").strip().split("\n") if s.strip()]
                ch = json.loads(truth.read_text(encoding="utf-8")).get("changes")
                if ch is None:
                    continue
                n = max(0, len(sents) - 1)
                docs.append((sents, (list(ch) + [0] * n)[:n]))
            test[tier] = docs
        tfreq = Counter(s for tier in tiers for sents, _ in test[tier] for s in sents)
        files = [Path(f) for f in args.pred_files.split(",") if f.strip()]
        summary["D"] = {}
        for tier in tiers:
            docs = test[tier]
            y_t = np.array([c for _, ch in docs for c in ch], dtype=int)
            lens = np.array([len(ch) for _, ch in docs])
            rep = np.concatenate([np.array([max(tfreq[a], tfreq[b]) >= 10 for a, b in zip(s[:-1], s[1:])], dtype=bool)
                                  if len(s) > 1 else np.zeros(0, dtype=bool) for s, _ in docs])
            p_rep, p_oth = y_t[rep].mean() if rep.any() else float("nan"), y_t[~rep].mean()
            base = lambda p: (1 - p) / (2 - p)
            print(f"\n{tier.upper()}  test boundaries next to a line with >= 10 copies: {rep.mean():.1%} "
                  f"({rep[y_t == 1].mean():.1%} of changes); change rate there {p_rep:.3f} vs {p_oth:.3f} elsewhere")
            print(f"  predict-0 baseline: near repeated {base(p_rep):.3f}, elsewhere {base(p_oth):.3f}; "
                  f"one-bit rule 'repeated => change' macro-F1 {macro_f1(y_t, rep.astype(int)):.3f}")
            summary["D"][tier] = {"rep_share": rep.mean(), "change_rate_rep": p_rep, "change_rate_other": p_oth,
                                  "one_bit_macro": macro_f1(y_t, rep.astype(int)), "files": {}}
            for f in files:
                if not f.exists():
                    continue
                z = np.load(f)
                yp, pp, lp = z[f"{tier}__labels"].astype(int), z[f"{tier}__preds"].astype(int), z[f"{tier}__lengths"]
                if not (np.array_equal(yp, y_t) and np.array_equal(lp, lens)):
                    print(f"  {f}: does not line up with the test text (skipped)")
                    continue
                a = macro_f1(yp[rep], pp[rep]) if rep.any() else float("nan")
                b, all_ = macro_f1(yp[~rep], pp[~rep]), macro_f1(yp, pp)
                print(f"  {str(f):55s} all {all_:.4f} | near repeated {a:.4f} | elsewhere {b:.4f}")
                summary["D"][tier]["files"][str(f)] = {"all": all_, "near_repeated": a, "elsewhere": b}
            save()
    print(f"\nSaved {out_json}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
