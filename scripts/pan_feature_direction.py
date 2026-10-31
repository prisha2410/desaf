#!/usr/bin/env python3
"""
Which way does each stylometric cue point at author changes, tier by tier, in a PAN
multi-author dataset? Works on any edition in the standard layout:
    {data-root}/{tier}/{split}/problem-N.txt        one sentence per line
    {data-root}/{tier}/{split}/truth-problem-N.json  {"changes": [0, 1, ...]}

Run it on PAN 2026 (fanfiction) and PAN 2025 (Reddit) with the same code. If Hard reverses
in both, the reversal follows the construction recipe (both editions select consecutive
sentences by topical AND stylistic similarity), not the source corpus.

AUC = P(a change boundary is MORE dissimilar than a no-change boundary):
above 0.5 = normal (classical stylometry), below 0.5 = reversed.
Reported per tier with a 95% CI from resampling documents, split into low/high topic
overlap inside each tier, and rerun without very short sentences (PAN has many 1-3 word
lines that make sentence distances unstable).

Run from D:\\sass-tad so the 193-feature stylometric extractor can be imported:
    python scripts/pan_feature_direction.py --data-root "D:\\mawsa26-pan-zenodo" --name pan2026
    python scripts/pan_feature_direction.py --data-root data/raw/pan2025_full --name pan2025
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FUNCTION_WORDS = sorted(set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in
into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your
yours yourself yourselves also however though although yet still even quite rather
""".split()))
FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}
PUNCT = ".,;:!?'\"-()"
WORD_RE = re.compile(r"[a-z']+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--name", required=True, help="label for the output files, e.g. pan2026")
    p.add_argument("--splits", default="validation", help="comma list, e.g. validation or train,validation")
    p.add_argument("--tiers", default="easy,medium,hard")
    p.add_argument("--tfidf-split", default="train",
                   help="split whose sentences fit the topic-overlap TF-IDF (falls back to the analysed splits)")
    p.add_argument("--max-docs", type=int, default=0, help="per tier and split; 0 = all")
    p.add_argument("--min-words", type=int, default=4, help="robustness rerun drops pairs with a shorter sentence")
    p.add_argument("--no-style193", action="store_true", help="skip the 193-feature distance (much faster)")
    p.add_argument("--no-char-ngrams", action="store_true", default=True)
    p.add_argument("--boot", type=int, default=200, help="document-level bootstrap resamples for the CIs")
    p.add_argument("--out-dir", default="results/feature_direction")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def problem_number(path):
    digits = re.sub(r"\D", "", path.stem)
    return int(digits) if digits else 0


def load_split(root, tier, split, max_docs, rng):
    d = Path(root) / tier / split
    files = sorted(d.glob("problem-*.txt"), key=problem_number)
    if not files:
        return None, 0
    if max_docs and len(files) > max_docs:
        files = [files[i] for i in sorted(rng.choice(len(files), max_docs, replace=False))]
    docs, mismatched = [], 0
    for f in files:
        truth = d / f"truth-{f.stem}.json"
        if not truth.exists():
            continue
        changes = json.loads(truth.read_text(encoding="utf-8")).get("changes")
        if changes is None:
            continue
        sents = [s.strip() for s in f.read_text(encoding="utf-8", errors="replace").split("\n") if s.strip()]
        if len(changes) != len(sents) - 1:
            mismatched += 1
        n = min(len(sents) - 1, len(changes))
        if n > 0:
            docs.append((sents[:n + 1], [int(c) for c in changes[:n]]))
    return docs, mismatched


def fw_vector(text):
    words = WORD_RE.findall(text.lower())
    v = np.zeros(len(FUNCTION_WORDS), dtype=np.float32)
    for w in words:
        j = FW_INDEX.get(w)
        if j is not None:
            v[j] += 1
    return v / max(len(words), 1)


def punct_vector(text):
    v = np.array([text.count(c) for c in PUNCT], dtype=np.float32)
    return v / max(len(text), 1)


def trigrams(text):
    t = text.lower()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def surface_distances(a, b):
    fw = float(np.abs(fw_vector(a) - fw_vector(b)).sum())
    ta, tb = trigrams(a), trigrams(b)
    union = len(ta | tb)
    tri = 1.0 - (len(ta & tb) / union) if union else 0.0
    pu = float(np.abs(punct_vector(a) - punct_vector(b)).sum())
    return fw, tri, pu


def auc_fast(x, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cell(x, y):
    if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
        return "n/a"
    res = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    auc = res.statistic / (int(y.sum()) * int((1 - y).sum()))
    stars = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else ""
    return f"{auc:.3f}{stars:3s} {'N' if auc > 0.5 else 'R'}"


def boot_ci(x, y, doc, n_boot, rng):
    """95% CI of the AUC, resampling whole documents."""
    order = np.argsort(doc, kind="stable")
    x, y, doc = x[order], y[order], doc[order]
    uniq, starts = np.unique(doc, return_index=True)
    ends = np.append(starts[1:], len(doc))
    aucs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
        aucs.append(auc_fast(x[idx], y[idx]))
    return np.nanpercentile(aucs, [2.5, 97.5])


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    extractor = None
    if not args.no_style193:
        try:
            sys.path.insert(0, str(ROOT))
            from src.features.stylometric import StylometricExtractor
            extractor = StylometricExtractor(use_char_ngrams=not args.no_char_ngrams)
            print(f"193-feature extractor loaded (dim {extractor.feature_dim})")
        except Exception as e:  # keep going with the three surface cues
            print(f"!!! Could not load the stylometric extractor ({type(e).__name__}: {e}); "
                  "skipping style_L2. Run from the sass-tad folder to include it.")

    t0 = time.time()
    data = {}
    for tier in tiers:
        docs = []
        for split in splits:
            loaded, mismatched = load_split(args.data_root, tier, split, args.max_docs, rng)
            if loaded is None:
                raise SystemExit(f"No problem-*.txt files in {Path(args.data_root) / tier / split}")
            if mismatched:
                print(f"  note: {tier}/{split}: {mismatched} documents where the number of labels != "
                      "sentences - 1 (truncated to the shorter)")
            docs.extend(loaded)
        data[tier] = docs
        print(f"  {tier}: {len(docs)} documents, {sum(len(c) for _, c in docs)} boundaries")

    from sklearn.feature_extraction.text import TfidfVectorizer
    fit_sents = []
    for tier in tiers:
        loaded, _ = load_split(args.data_root, tier, args.tfidf_split, 0, rng)
        if loaded:
            fit_sents.extend(s for sents, _ in loaded for s in sents)
    source = args.tfidf_split
    if not fit_sents:
        fit_sents = [s for docs in data.values() for sents, _ in docs for s in sents]
        source = "+".join(splits)
    vectorizer = TfidfVectorizer(max_features=20000).fit(fit_sents)
    print(f"Topic-overlap TF-IDF fitted on {len(fit_sents)} '{source}' sentences")
    del fit_sents

    feat_names = ["fw_dist", "trigram_dissim", "punct_dist"] + (["style_L2 (193)"] if extractor else [])
    per_tier = {}
    for tier, docs in data.items():
        rows, labels, overlap, doc_idx, short = [], [], [], [], []
        for di, (sents, changes) in enumerate(docs):
            vecs = [np.asarray(extractor.extract_vector(s), dtype=np.float32) for s in sents] if extractor else None
            a, b = sents[:-1], sents[1:]
            ovl = np.asarray(vectorizer.transform(a).multiply(vectorizer.transform(b)).sum(axis=1)).ravel()
            for i in range(len(changes)):
                r = list(surface_distances(a[i], b[i]))
                if extractor:
                    r.append(float(np.linalg.norm(np.abs(vecs[i + 1] - vecs[i]))))
                rows.append(r)
                labels.append(changes[i])
                overlap.append(ovl[i])
                doc_idx.append(di)
                short.append(min(len(a[i].split()), len(b[i].split())) < args.min_words)
        per_tier[tier] = {k: np.asarray(v) for k, v in dict(
            X=rows, y=labels, overlap=overlap, doc=doc_idx, short=short).items()}
        per_tier[tier]["X"] = per_tier[tier]["X"].astype(np.float32)
        print(f"  features done for {tier} ({time.time() - t0:.0f}s)")

    all_overlap = np.concatenate([v["overlap"] for v in per_tier.values()])
    split_at = float(np.median(all_overlap))

    print("\n" + "=" * 100)
    print(f"{args.name}: direction of stylometric cues at author changes ({'+'.join(splits)}, "
          f"{len(all_overlap)} boundaries)")
    print("AUC = P(change boundary more dissimilar than a no-change one); N = normal (>0.5), R = reversed (<0.5)")
    print("* p<.05 ** p<.01 *** p<.001 (Mann-Whitney); [95% CI] resamples documents")
    print("=" * 100)
    for tier, v in per_tier.items():
        X, y, doc = v["X"], v["y"], v["doc"]
        print(f"\n{tier.upper()}  (change rate {y.mean():.3f}; mean topic overlap at a change "
              f"{v['overlap'][y == 1].mean():.4f} vs within author {v['overlap'][y == 0].mean():.4f})")
        print(f"  {'subset':34s}" + "".join(f"{n:>20s}" for n in feat_names))
        print(f"  {'all (n=' + str(len(y)) + ')':34s}" + "".join(f"{cell(X[:, j], y):>20s}" for j in range(X.shape[1])))
        cis = [boot_ci(X[:, j], y, doc, args.boot, rng) for j in range(X.shape[1])]
        print(f"  {'  95% CI':34s}" + "".join(f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>20s}"
                                             for lo, hi in cis))
        for label, m in [(f"overlap <= {split_at:.4f} (median)", v["overlap"] <= split_at),
                         (f"overlap >  {split_at:.4f}", v["overlap"] > split_at),
                         (f"both sentences >= {args.min_words} words", ~v["short"])]:
            row = f"  {label + ' (n=' + str(int(m.sum())) + ')':34s}"
            row += "".join(f"{cell(X[m, j], y[m]):>20s}" for j in range(X.shape[1]))
            print(row)

    hi = {t: (v["overlap"] > split_at).mean() for t, v in per_tier.items()}
    print("\nShare of boundaries above the median overlap, by tier: "
          + ", ".join(f"{t} {s:.2f}" for t, s in hi.items()))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tier, v in per_tier.items():
        np.savez_compressed(out / f"{args.name}_{tier}.npz", features=v["X"], labels=v["y"],
                            overlap=v["overlap"], doc=v["doc"], short=v["short"],
                            feature_names=np.array(feat_names))
    print(f"Saved per-boundary features to {out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
