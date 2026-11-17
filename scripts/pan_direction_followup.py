#!/usr/bin/env python3
"""
Follow-up to scripts/pan_feature_direction.py (v2). Tests whether the cue directions in the first tables
survive the obvious confounds:

  A. Topic overlap. Compare a change only with no-change boundaries at the SAME overlap (stratified AUC).
     Matching variable 1 = the first script's all-word TF-IDF cosine. It shares words with fw_dist and
     nearly determines trigram_dissim, so for those two cues the matched value is partly mechanical.
     Matching variable 2 = CONTENT-word TF-IDF cosine (English stop words removed, fitted on the train
     split), as in the C50 experiment: fw_dist and punct_dist are then held out from the matching variable.
     Also: WITHIN-DOCUMENT AUC (each change compared only with no-change boundaries of its own document).
  B. Sentence length, repeated boilerplate and dialogue. Length-difference cues; every cue at matched length
     (deciles of the shorter sentence's word count x quintiles of the word-count difference); an 8-word floor;
     every cue without boundaries that touch a sentence repeated >= 10 times (moderator boilerplate etc.);
     how much of the unscaled style_L2 is length; cue AUCs split by quotation marks (dialogue).
  C. Which kinds of stylometric feature reverse? The 193 features z-scored (rare features dropped, |z|
     clipped at 5), one distance per feature group, unmatched and at matched length, on a document sample.
     Slow (spaCy): --group-docs sets documents per tier (0 = all, -1 = skip).

AUC = P(change boundary more dissimilar than a no-change boundary): > 0.5 normal (N), < 0.5 reversed (R),
"~" = within 0.0005 of 0.5. 95% CIs resample whole documents. Treat |AUC - 0.5| < 0.05 as near chance.

Run from D:\\sass-tad AFTER pan_feature_direction.py has written results/feature_direction/*.npz,
with the same --name and --splits as that run:
    python scripts/pan_direction_followup.py --data-root data/raw/pan2025_full --name pan2025 --group-docs 0
    python scripts/pan_direction_followup.py --data-root "D:\\mawsa26-pan-zenodo" --name pan2026 --group-docs 400 --boot 100
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
QUOTES = ('"', "\u201c", "\u201d")
MIN_PER_CLASS = 20
DUP_MIN = 10   # a sentence occurring >= 10 times in the analysed documents counts as repeated boilerplate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--name", required=True, help="same --name as the pan_feature_direction.py run")
    p.add_argument("--splits", default="validation", help="must match the pan_feature_direction.py run")
    p.add_argument("--tiers", default="easy,medium,hard")
    p.add_argument("--tfidf-split", default="train", help="split that fits the content-word TF-IDF")
    p.add_argument("--npz-dir", default="results/feature_direction")
    p.add_argument("--bins", type=int, default=10, help="quantile bins of positive overlap, per tier")
    p.add_argument("--group-docs", type=int, default=400,
                   help="documents per tier for part C; 0 = all, -1 = skip part C")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- loading (same order as the first script)
def problem_number(path):
    digits = re.sub(r"\D", "", path.stem)
    return int(digits) if digits else 0


def load_split(root, tier, split):
    d = Path(root) / tier / split
    files = sorted(d.glob("problem-*.txt"), key=problem_number)
    docs = []
    for f in files:
        truth = d / f"truth-{f.stem}.json"
        if not truth.exists():
            continue
        changes = json.loads(truth.read_text(encoding="utf-8")).get("changes")
        if changes is None:
            continue
        sents = [s.strip() for s in f.read_text(encoding="utf-8", errors="replace").split("\n") if s.strip()]
        n = min(len(sents) - 1, len(changes))
        if n > 0:
            docs.append((sents[:n + 1], [int(c) for c in changes[:n]]))
    return docs


# ---------------------------------------------------------------- statistics
def auc_fast(x, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def grouped_auc(x, y, g):
    """AUC over (change, no-change) pairs that share a group (ties = 1/2); groups weighted by n1*n0.
    g = overlap bin -> overlap-matched AUC; g = document copy -> within-document AUC."""
    order = np.lexsort((x, g))
    xs, ys, gs = x[order], y[order].astype(np.float64), g[order]
    n = len(xs)
    if n == 0:
        return float("nan")
    gnew = np.r_[True, gs[1:] != gs[:-1]]
    gid = np.cumsum(gnew) - 1
    pos = np.arange(n) - np.flatnonzero(gnew)[gid] + 1.0
    tnew = gnew | np.r_[True, xs[1:] != xs[:-1]]
    tid = np.cumsum(tnew) - 1
    rank = (np.bincount(tid, weights=pos) / np.bincount(tid))[tid]
    n1 = np.bincount(gid, weights=ys)
    n0 = np.bincount(gid) - n1
    u = np.bincount(gid, weights=rank * ys) - n1 * (n1 + 1) / 2
    w = n1 * n0
    return float(u[w > 0].sum() / w[w > 0].sum()) if w.sum() > 0 else float("nan")


def doc_index(doc):
    order = np.argsort(doc, kind="stable")
    _, starts = np.unique(doc[order], return_index=True)
    ends = np.append(starts[1:], len(doc))
    return [order[s:e] for s, e in zip(starts, ends)]


def boot_ci(fn, groups, n_boot, rng):
    """fn(idx, copy_id) -> 1-D array; resamples whole documents (copy_id keeps duplicates apart)."""
    out = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        cid = np.repeat(np.arange(len(pick)), [len(groups[i]) for i in pick])
        out.append(fn(idx, cid))
    out = np.asarray(out, dtype=float)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def tag(a):
    if a is None or not np.isfinite(a):
        return "  n/a  "
    return f"{a:.3f} {'~' if abs(a - 0.5) < 5e-4 else ('N' if a > 0.5 else 'R')}"


def ci_txt(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def quantile_bins(v, n_bins, zero_bin=False):
    """Quantile bins; with zero_bin, exact zeros get bin 0 and positive values bins 1..k."""
    bins = np.zeros(len(v), dtype=np.int64)
    m = v > 1e-12 if zero_bin else np.ones(len(v), dtype=bool)
    if m.sum() > 0:
        edges = np.unique(np.quantile(v[m], np.linspace(0, 1, n_bins + 1)[1:-1]))
        bins[m] = (1 if zero_bin else 0) + np.searchsorted(edges, v[m], side="right")
    return bins


def length_strata(min_words, d_words):
    return quantile_bins(min_words.astype(float), 10) * 100 + quantile_bins(d_words.astype(float), 5)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------------------------------------------------------- part C feature groups
def group_of(name):
    # punct_period is count('.') / words: for a one-period sentence it is exactly 1/length, so it goes
    # with the length features.
    if name in ("char_count", "word_count", "avg_sentence_length", "punct_period"):
        return "length"
    if name in ("flesch_reading_ease", "gunning_fog", "coleman_liau", "avg_syllables_per_word",
                "complex_word_ratio"):
        return "readability"
    if name.startswith("punct_"):
        return "punctuation"
    if name.startswith("fw_"):
        return "function_words"
    if name.startswith("pos_"):
        return "pos"
    return "other_surface"


GROUP_ORDER = ["length", "readability", "punctuation", "function_words", "pos", "other_surface"]


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    rng_c = np.random.RandomState(args.seed + 1)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    npz_dir = Path(args.npz_dir)
    out_json = npz_dir / f"{args.name}_followup.json"
    t0 = time.time()
    summary = {"name": args.name, "splits": splits, "tiers": {t: {} for t in tiers}}

    def save():
        out_json.write_text(json.dumps(clean(summary), indent=1), encoding="utf-8")

    # ---------------- load saved per-boundary features and the text, check they line up
    saved, texts = {}, {}
    for tier in tiers:
        f = npz_dir / f"{args.name}_{tier}.npz"
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run pan_feature_direction.py with --name {args.name} first.")
        z = np.load(f, allow_pickle=False)
        saved[tier] = {k: z[k] for k in z.files}
        docs = []
        for split in splits:
            docs.extend(load_split(args.data_root, tier, split))
        y_txt = np.array([c for _, ch in docs for c in ch], dtype=int)
        doc_txt = np.repeat(np.arange(len(docs)), [len(ch) for _, ch in docs])
        if not (len(y_txt) == len(saved[tier]["labels"]) and np.array_equal(y_txt, saved[tier]["labels"])
                and np.array_equal(doc_txt, saved[tier]["doc"])):
            raise SystemExit(f"{tier}: the text under --data-root/--splits does not line up with {f} "
                             f"({len(y_txt)} vs {len(saved[tier]['labels'])} boundaries). Use the same "
                             "--data-root, --splits and --name as the pan_feature_direction.py run (no --max-docs).")
        texts[tier] = docs
    feat_names = [str(s) for s in saved[tiers[0]]["feature_names"]]
    n_feat = len(feat_names)
    counts = ", ".join(f"{t} {len(saved[t]['labels'])}" for t in tiers)
    print(f"Loaded {counts} boundaries; text lines up with the saved features ({time.time() - t0:.0f}s)")

    # ---------------- text-derived variables per boundary
    from sklearn.feature_extraction.text import TfidfVectorizer
    fit = [s for tier in tiers for sents, _ in load_split(args.data_root, tier, args.tfidf_split) for s in sents]
    src = args.tfidf_split
    if not fit:
        fit = [s for tier in tiers for sents, _ in texts[tier] for s in sents]
        src = "+".join(splits)
    vec_c = TfidfVectorizer(max_features=20000, stop_words="english").fit(fit)
    print(f"Content-word TF-IDF (stop words removed) fitted on {len(fit)} '{src}' sentences ({time.time() - t0:.0f}s)")
    del fit
    from collections import Counter
    freq = Counter(s for tier in tiers for sents, _ in texts[tier] for s in sents)
    extra = {}
    for tier in tiers:
        oc, minw, dw, dc, q, dup = [], [], [], [], [], []
        for sents, ch in texts[tier]:
            T = vec_c.transform(sents)
            oc.append(np.asarray(T[:-1].multiply(T[1:]).sum(axis=1)).ravel())
            w = np.array([len(s.split()) for s in sents], dtype=float)
            c = np.array([len(s) for s in sents], dtype=float)
            hq = np.array([any(ch_ in s for ch_ in QUOTES) for s in sents], dtype=int)
            minw.append(np.minimum(w[1:], w[:-1]))
            dw.append(np.abs(w[1:] - w[:-1]))
            dc.append(np.abs(c[1:] - c[:-1]))
            q.append(hq[1:] + hq[:-1])
            rep = np.array([freq[s] >= DUP_MIN for s in sents], dtype=bool)
            dup.append(rep[1:] | rep[:-1])
        extra[tier] = {k: np.concatenate(v) for k, v in
                       dict(ovl_c=oc, min_words=minw, d_words=dw, d_chars=dc, quotes=q, dup=dup).items()}
    print(f"Text variables done ({time.time() - t0:.0f}s)")

    # ============================================================ A. topic overlap
    print("\n" + "=" * 110)
    print(f"{args.name}  A. Do the cues keep their direction at MATCHED topic overlap, and within documents?")
    print("all-word overlap = the first script's TF-IDF cosine (tokens of 2+ characters, top 20k train terms, "
          "function words INCLUDED: matching on it is partly mechanical for fw_dist and trigram_dissim).")
    print("content overlap  = TF-IDF cosine without English stop words: fw_dist, punct_dist (and style_L2) are "
          "held out from it.")
    print(f"matched = AUC over change/no-change pairs in the same overlap bin (zero bin + {args.bins} "
          "within-tier quantile bins). within-doc = pairs from the same document only.")
    print("overlap AUC > 0.5 = changes share LESS vocabulary than no-change pairs (the natural direction).")
    print("=" * 110)
    for tier in tiers:
        v, e = saved[tier], extra[tier]
        X, y, doc = v["features"].astype(np.float64), v["labels"].astype(int), v["doc"]
        ovl_a, ovl_c = v["overlap"].astype(np.float64), e["ovl_c"]
        groups = doc_index(doc)
        b_a, b_c = quantile_bins(ovl_a, args.bins, True), quantile_bins(ovl_c, args.bins, True)
        raw = [auc_fast(X[:, j], y) for j in range(n_feat)]
        m_a = [grouped_auc(X[:, j], y, b_a) for j in range(n_feat)]
        m_c = [grouped_auc(X[:, j], y, b_c) for j in range(n_feat)]
        wdoc = [grouped_auc(X[:, j], y, doc) for j in range(n_feat)]

        def stat(i, cid):
            out = [grouped_auc(X[i, j], y[i], b_c[i]) for j in range(n_feat)]
            out += [grouped_auc(X[i, j], y[i], cid) for j in range(n_feat)]
            out += [auc_fast(-ovl_a[i], y[i]), auc_fast(-ovl_c[i], y[i])]
            return np.array(out)

        lo, hi = boot_ci(stat, groups, args.boot, rng)
        k = n_feat
        oa, oc_ = auc_fast(-ovl_a, y), auc_fast(-ovl_c, y)
        ra = ovl_a[y == 1].mean() / max(ovl_a[y == 0].mean(), 1e-12)
        rc = ovl_c[y == 1].mean() / max(ovl_c[y == 0].mean(), 1e-12)
        print(f"\n{tier.upper()}  n={len(y)}  change rate {y.mean():.3f}")
        print(f"  all-word overlap change/within {ovl_a[y == 1].mean():.4f}/{ovl_a[y == 0].mean():.4f} "
              f"(ratio {ra:.2f}), overlap AUC {tag(oa)} {ci_txt(lo[2 * k], hi[2 * k])}")
        print(f"  content overlap  change/within {ovl_c[y == 1].mean():.4f}/{ovl_c[y == 0].mean():.4f} "
              f"(ratio {rc:.2f}), overlap AUC {tag(oc_)} {ci_txt(lo[2 * k + 1], hi[2 * k + 1])}; "
              f"zero content overlap: {(b_c == 0)[y == 1].mean():.1%} of changes, "
              f"{(b_c == 0)[y == 0].mean():.1%} of no-changes")
        print(f"  {'':26s}" + "".join(f"{n:>22s}" for n in feat_names))
        print(f"  {'unmatched (as before)':26s}" + "".join(f"{tag(a):>22s}" for a in raw))
        print(f"  {'matched all-word overlap':26s}" + "".join(f"{tag(a):>22s}" for a in m_a))
        print(f"  {'matched content overlap':26s}" + "".join(f"{tag(a):>22s}" for a in m_c))
        print(f"  {'  95% CI':26s}" + "".join(f"{ci_txt(lo[j], hi[j]):>22s}" for j in range(k)))
        rob = {}
        for nb in (5, 20):
            bb = quantile_bins(ovl_c, nb, True)
            rob[nb] = [grouped_auc(X[:, j], y, bb) for j in range(n_feat)]
            print(f"  {'  content, ' + str(nb) + ' bins':26s}" + "".join(f"{tag(a):>22s}" for a in rob[nb]))
        print(f"  {'within-document':26s}" + "".join(f"{tag(a):>22s}" for a in wdoc))
        print(f"  {'  95% CI':26s}" + "".join(f"{ci_txt(lo[k + j], hi[k + j]):>22s}" for j in range(k)))
        print(f"  per content-overlap bin (AUC shown only with >= {MIN_PER_CLASS} changes and no-changes):")
        print(f"    {'':6s} {'overlap range':>17s} {'n':>7s} {'changes':>8s}" + "".join(f"{n[:12]:>14s}" for n in feat_names))
        per_bin = []
        for b in np.unique(b_c):
            m = b_c == b
            n1 = int(y[m].sum())
            ok = n1 >= MIN_PER_CLASS and int(m.sum()) - n1 >= MIN_PER_CLASS
            aucs = [auc_fast(X[m, j], y[m]) if ok else float("nan") for j in range(n_feat)]
            print(f"    bin {b:2d} {ovl_c[m].min():8.4f}-{ovl_c[m].max():8.4f} {int(m.sum()):7d} {n1:8d}"
                  + "".join(f"{tag(a):>14s}" for a in aucs))
            per_bin.append({"bin": int(b), "lo": ovl_c[m].min(), "hi": ovl_c[m].max(), "n": int(m.sum()),
                            "n_change": n1, "auc": aucs})
        summary["tiers"][tier]["A"] = {
            "n": len(y), "change_rate": y.mean(),
            "overlap_allword": {"change": ovl_a[y == 1].mean(), "within": ovl_a[y == 0].mean(), "ratio": ra,
                                "auc": oa, "ci": [lo[2 * k], hi[2 * k]]},
            "overlap_content": {"change": ovl_c[y == 1].mean(), "within": ovl_c[y == 0].mean(), "ratio": rc,
                                "auc": oc_, "ci": [lo[2 * k + 1], hi[2 * k + 1]]},
            "auc_unmatched": dict(zip(feat_names, raw)),
            "auc_matched_allword": dict(zip(feat_names, m_a)),
            "auc_matched_content": dict(zip(feat_names, m_c)),
            "auc_matched_content_ci": {n: [lo[j], hi[j]] for j, n in enumerate(feat_names)},
            "auc_matched_content_5bins": dict(zip(feat_names, rob[5])),
            "auc_matched_content_20bins": dict(zip(feat_names, rob[20])),
            "auc_within_doc": dict(zip(feat_names, wdoc)),
            "auc_within_doc_ci": {n: [lo[k + j], hi[k + j]] for j, n in enumerate(feat_names)},
            "per_content_bin": per_bin,
        }
        save()
    print(f"\n(part A done, {time.time() - t0:.0f}s)")

    # ============================================================ B. length and dialogue
    print("\n" + "=" * 110)
    print(f"{args.name}  B. Sentence length and dialogue")
    print("matched length = pairs in the same stratum of (shorter sentence's word count, 10 bins) x "
          "(word-count difference, 5 bins).")
    print("=" * 110)
    for tier in tiers:
        v, e = saved[tier], extra[tier]
        X, y, doc = v["features"].astype(np.float64), v["labels"].astype(int), v["doc"]
        groups = doc_index(doc)
        L = np.column_stack([e["d_words"], e["d_chars"],
                             np.abs(np.log((e["d_words"] + e["min_words"] + 1) / (e["min_words"] + 1)))])
        names_b = ["|d words|", "|d chars|", "|log length ratio|"]
        strata = length_strata(e["min_words"], e["d_words"])
        len_auc = [auc_fast(L[:, j], y) for j in range(3)]
        lm = [grouped_auc(X[:, j], y, strata) for j in range(n_feat)]
        floor8 = e["min_words"] >= 8
        f8 = [auc_fast(X[floor8, j], y[floor8]) for j in range(n_feat)]

        def stat(i, cid):
            return np.array([auc_fast(L[i, j], y[i]) for j in range(3)]
                            + [grouped_auc(X[i, j], y[i], strata[i]) for j in range(n_feat)])

        lo, hi = boot_ci(stat, groups, args.boot, rng)
        print(f"\n{tier.upper()}  median words per sentence pair (shorter sentence) {np.median(e['min_words']):.0f}; "
              f"pairs with a sentence < 4 words {(e['min_words'] < 4).mean():.1%}, < 8 words {(~floor8).mean():.1%}")
        print("  length cues: " + "  ".join(f"{n} {tag(a)} {ci_txt(lo[j], hi[j])}"
                                            for j, (n, a) in enumerate(zip(names_b, len_auc))))
        print(f"  {'':26s}" + "".join(f"{n:>22s}" for n in feat_names))
        print(f"  {'matched length':26s}" + "".join(f"{tag(a):>22s}" for a in lm))
        print(f"  {'  95% CI':26s}" + "".join(f"{ci_txt(lo[3 + j], hi[3 + j]):>22s}" for j in range(n_feat)))
        print(f"  {'both sentences >= 8 words':26s}" + "".join(f"{tag(a):>22s}" for a in f8)
              + f"   (n={int(floor8.sum())})")
        nd = ~e["dup"]
        b_c = quantile_bins(e["ovl_c"], args.bins, True)
        nodup = [auc_fast(X[nd, j], y[nd]) for j in range(n_feat)]
        nodup_m = [grouped_auc(X[nd, j], y[nd], b_c[nd]) for j in range(n_feat)]
        print(f"  repeated sentences (>= {DUP_MIN} copies) touch {e['dup'].mean():.1%} of boundaries "
              f"({e['dup'][y == 1].mean():.1%} of changes, {e['dup'][y == 0].mean():.1%} of no-changes)")
        print(f"  {'without repeated sentences':26s}" + "".join(f"{tag(a):>22s}" for a in nodup)
              + f"   (n={int(nd.sum())})")
        print(f"  {'  + matched content overlap':26s}" + "".join(f"{tag(a):>22s}" for a in nodup_m))
        res = {"length_auc": dict(zip(names_b, len_auc)),
               "length_auc_ci": {n: [lo[j], hi[j]] for j, n in enumerate(names_b)},
               "auc_matched_length": dict(zip(feat_names, lm)),
               "auc_matched_length_ci": {n: [lo[3 + j], hi[3 + j]] for j, n in enumerate(feat_names)},
               "auc_floor8": dict(zip(feat_names, f8)), "n_floor8": int(floor8.sum()),
               "dup_share": e["dup"].mean(), "dup_share_changes": e["dup"][y == 1].mean(),
               "dup_share_nochanges": e["dup"][y == 0].mean(),
               "auc_no_dup": dict(zip(feat_names, nodup)), "auc_no_dup_matched_content": dict(zip(feat_names, nodup_m))}
        if "style_L2 (193)" in feat_names:
            s = X[:, feat_names.index("style_L2 (193)")]
            rho = stats.spearmanr(s, e["d_chars"]).correlation
            coef = np.polyfit(e["d_chars"], s, 2)
            resid = s - np.polyval(coef, e["d_chars"])
            ra = auc_fast(resid, y)
            print(f"  style_L2 vs |d chars|: Spearman rho {rho:.3f}; AUC of style_L2 after removing a quadratic "
                  f"fit on |d chars| {tag(ra)}")
            res.update({"styleL2_dchars_spearman": rho, "styleL2_resid_auc": ra})
        print("  by quotation marks (dialogue):")
        res["by_quotes"] = {}
        for qv, lab in [(0, "neither sentence quotes"), (1, "one sentence quotes"), (2, "both sentences quote")]:
            m = e["quotes"] == qv
            n1 = int(y[m].sum())
            ok = n1 >= MIN_PER_CLASS and int(m.sum()) - n1 >= MIN_PER_CLASS
            aucs = [auc_fast(X[m, j], y[m]) if ok else float("nan") for j in range(n_feat)]
            print(f"    {lab:24s} n={int(m.sum()):7d} changes={n1:6d}" + "".join(f"{tag(a):>14s}" for a in aucs))
            res["by_quotes"][lab] = {"n": int(m.sum()), "n_change": n1, "auc": dict(zip(feat_names, aucs))}
        summary["tiers"][tier]["B"] = res
        save()
    print(f"\n(part B done, {time.time() - t0:.0f}s)")

    # ============================================================ C. z-scored feature groups
    if args.group_docs < 0:
        print(f"\nSaved {out_json}  ({time.time() - t0:.0f}s total)")
        return
    print("\n" + "=" * 110)
    print(f"{args.name}  C. Which kinds of stylometric feature reverse? (z-scored, one distance per group)")
    print("=" * 110)
    try:
        sys.path.insert(0, str(ROOT))
        from src.features.stylometric import StylometricExtractor
        extractor = StylometricExtractor(use_char_ngrams=False)
        names = extractor.feature_names
    except Exception as ex:
        print(f"!!! Could not load the stylometric extractor ({type(ex).__name__}: {ex}); skipping part C.")
        print(f"\nSaved {out_json}  ({time.time() - t0:.0f}s total)")
        return
    try:
        sample = {}
        for tier in tiers:
            docs = texts[tier]
            k = len(docs) if args.group_docs == 0 or args.group_docs >= len(docs) else args.group_docs
            pick = sorted(rng_c.choice(len(docs), k, replace=False)) if k < len(docs) else list(range(len(docs)))
            vecs, ys, dids, minw, dwd = [], [], [], [], []
            for di in pick:
                sents, ch = docs[di]
                vecs.append(np.stack([np.asarray(extractor.extract_vector(s), dtype=np.float64) for s in sents]))
                w = np.array([len(s.split()) for s in sents], dtype=float)
                ys.append(np.asarray(ch, dtype=int))
                dids.append(np.full(len(ch), di))
                minw.append(np.minimum(w[1:], w[:-1]))
                dwd.append(np.abs(w[1:] - w[:-1]))
            sample[tier] = (vecs, np.concatenate(ys), np.concatenate(dids), np.concatenate(minw), np.concatenate(dwd))
            print(f"  {tier}: {k} documents, {len(sample[tier][1])} boundaries extracted ({time.time() - t0:.0f}s)")
    except Exception as ex:
        print(f"!!! Feature extraction failed ({type(ex).__name__}: {ex}); parts A and B are saved.")
        return
    allv = np.concatenate([V for vecs, *_ in sample.values() for V in vecs])
    sd = allv.std(0)
    present = (np.abs(allv) > 1e-12).mean(0)
    keep = (sd > 1e-9) & (present >= 0.01)
    sd[~keep] = 1.0
    grp = np.array([group_of(n) for n in names])
    gidx = {g: np.flatnonzero((grp == g) & keep) for g in GROUP_ORDER}
    gidx = {g: ix for g, ix in gidx.items() if len(ix)}
    print(f"  z-scored with label-free SDs of all {len(allv)} sampled sentences; {int((~keep).sum())} of "
          f"{len(names)} features dropped (constant or non-zero in < 1% of sentences); |z| clipped at 5")
    print("  features kept per group: " + ", ".join(f"{g} {len(ix)}" for g, ix in gidx.items())
          + "   (length includes punct_period = periods per word)")
    no_lr = keep & ~np.isin(grp, ["length", "readability"])
    no_fw = keep & (grp != "function_words")
    cols = ["raw_L2 (=style_L2)", "all_z"] + list(gidx) + ["all_z_no_len_read", "all_z_no_fw"]
    print(f"\n  {'':16s}" + "".join(f"{c[:19]:>20s}" for c in cols))
    for tier, (vecs, yv, dv, mw, dwd) in sample.items():
        D_raw = np.concatenate([np.abs(V[1:] - V[:-1]) for V in vecs])
        Dz = np.minimum(np.abs(D_raw) / sd, 5.0)
        Dz[:, ~keep] = 0.0
        dist = [np.linalg.norm(D_raw, axis=1), np.linalg.norm(Dz, axis=1)]
        dist += [np.linalg.norm(Dz[:, ix], axis=1) for ix in gidx.values()]
        dist += [np.linalg.norm(Dz[:, no_lr], axis=1), np.linalg.norm(Dz[:, no_fw], axis=1)]
        dist = np.column_stack(dist)
        strata = length_strata(mw, dwd)
        nc = dist.shape[1]
        un = [auc_fast(dist[:, j], yv) for j in range(nc)]
        lm = [grouped_auc(dist[:, j], yv, strata) for j in range(nc)]

        def stat(i, cid):
            return np.array([auc_fast(dist[i, j], yv[i]) for j in range(nc)]
                            + [grouped_auc(dist[i, j], yv[i], strata[i]) for j in range(nc)])

        lo, hi = boot_ci(stat, doc_index(dv), args.boot, rng_c)
        print(f"  {tier.upper() + ' unmatched':16s}" + "".join(f"{tag(a):>20s}" for a in un))
        print(f"  {'  95% CI':16s}" + "".join(f"{ci_txt(lo[j], hi[j]):>20s}" for j in range(nc)))
        print(f"  {'  matched length':16s}" + "".join(f"{tag(a):>20s}" for a in lm))
        print(f"  {'  95% CI':16s}" + "".join(f"{ci_txt(lo[nc + j], hi[nc + j]):>20s}" for j in range(nc)))
        summary["tiers"][tier]["C"] = {
            "n_boundaries": int(len(yv)), "n_docs": int(len(np.unique(dv))),
            "features_kept": {g: int(len(ix)) for g, ix in gidx.items()},
            "auc": {c: {"unmatched": a, "ci": [lo[j], hi[j]], "matched_length": b, "ci_matched": [lo[nc + j], hi[nc + j]]}
                    for j, (c, a, b) in enumerate(zip(cols, un, lm))}}
        np.savez_compressed(npz_dir / f"{args.name}_{tier}_groups.npz", dist=dist.astype(np.float32),
                            labels=yv, doc=dv, columns=np.array(cols))
        save()
    print(f"\n(part C done, {time.time() - t0:.0f}s)")
    print(f"\nSaved {out_json}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
