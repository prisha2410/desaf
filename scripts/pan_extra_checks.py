#!/usr/bin/env python3
"""
Four model-free checks that decide how the construction story may be worded. CPU only, no spaCy.

  1. LAG TEST (splice filter vs natural reply-echoing). A splice filter acts on ONE sentence pair, so the extra
     similarity at an author change should sit at lag 1 (new author's first sentence vs the last sentence of the
     previous author) and fade at lags 2-3 (vs the previous author's earlier sentences). Natural accommodation in a
     thread (replies echoing the comment they answer) is a property of the whole previous run and should persist.
     Only boundaries whose previous three sentences belong to one author are used (same rule for both classes).
     Calibration: the constructed documents (topic_style = known splice filter; same_topic = no splice filter).
     Also: share of changes whose new first sentence nearly duplicates a sentence of the previous author's run.
  2. MECHANISM PROFILE of the constructions: the same length / length x topic matching that was applied to PAN 2025
     Hard, on the constructed documents (is the match with PAN Hard length-led like PAN Hard, or not?).
  3. ONE SHARED content TF-IDF for PAN 2025 and PAN 2026 (fitted on an equal sample of both editions' train
     sentences): content overlap at Hard changes vs within an author, on one scale.
  4. BOILERPLATE on the PAN 2025 test set: an 'edge rule' (change iff exactly one of the two sentences is a
     repeated line) on boundaries next to repeated lines, vs saved DeSAF predictions; and how many repeated test
     lines also occur in the train text (memorisation is only possible if they do).

Run from D:\\sass-tad after construction_experiment.py has written results/construction/*.jsonl:
    python scripts/pan_extra_checks.py --pan2026-root "D:\\mawsa26-pan-zenodo"
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pan_direction_followup import load_split, auc_fast, grouped_auc, doc_index, boot_ci, quantile_bins, tag, ci_txt, clean  # noqa: E402
from pan_feature_direction import surface_distances  # noqa: E402

TIERS = ["easy", "medium", "hard"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pan2025-root", default="data/raw/pan2025_full")
    p.add_argument("--pan2026-root", default="")
    p.add_argument("--constructions-dir", default="results/construction")
    p.add_argument("--construction-names", default="reddit2025,c50")
    p.add_argument("--fit-sample", type=int, default=150000, help="train sentences per edition for the shared TF-IDF")
    p.add_argument("--pred-files", default="checkpoints/seed43/test_predictions.npz,checkpoints/seed44/test_predictions.npz,"
                   "checkpoints/seed45/test_predictions.npz,checkpoints/concat_seed43/test_predictions.npz,"
                   "checkpoints/concat_seed44/test_predictions.npz,checkpoints/concat_seed45/test_predictions.npz")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/feature_direction/extra_checks.json")
    return p.parse_args()


def content_vectorizer(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=20000, stop_words="english").fit(sentences)


def cos_rows(V, i, j):
    return np.asarray(V[i].multiply(V[j]).sum(axis=1)).ravel()


def trigram_dis(a, b):
    return np.array([surface_distances(x, y)[1] for x, y in zip(a, b)])


def lag_table(docs, vec, rng, boot, label):
    """docs: list of (sentences, labels). Boundaries i with labels[i-1] == labels[i-2] == 0 (s_{i-2}..s_i one author)."""
    rows = {"y": [], "doc": [], "c1": [], "c2": [], "c3": [], "t1": [], "t2": [], "t3": [], "dup": []}
    for di, (sents, lab) in enumerate(docs):
        if len(sents) < 4:
            continue
        V = vec.transform(sents)
        for i in range(2, len(lab)):
            if lab[i - 1] != 0 or lab[i - 2] != 0:
                continue
            new = i + 1
            rows["y"].append(lab[i]); rows["doc"].append(di)
            for k, prev in ((1, i), (2, i - 1), (3, i - 2)):
                rows[f"c{k}"].append(float(V[new].multiply(V[prev]).sum()))
                rows[f"t{k}"].append(surface_distances(sents[new], sents[prev])[1])
            run = [i - 2, i - 1, i]
            rows["dup"].append(max(float(V[new].multiply(V[r]).sum()) for r in run) >= 0.8)
    R = {k: np.asarray(v) for k, v in rows.items()}
    y = R["y"].astype(int)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        print(f"  {label:28s} not enough boundaries")
        return None
    cols = [("content", "c"), ("trigram", "t")]
    X = np.column_stack([-R["c1"], -R["c2"], -R["c3"], R["t1"], R["t2"], R["t3"]])
    aucs = [auc_fast(X[:, j], y) for j in range(6)]
    lo, hi = boot_ci(lambda i, cid: np.array([auc_fast(X[i, j], y[i]) for j in range(6)]), doc_index(R["doc"]), boot, rng)
    dup_c, dup_n = R["dup"][y == 1].mean(), R["dup"][y == 0].mean()
    print(f"  {label:28s} n={len(y):6d} changes={int(y.sum()):5d} | content lag1/2/3 "
          + " ".join(f"{tag(a)} {ci_txt(l, h)}" for a, l, h in zip(aucs[:3], lo[:3], hi[:3])))
    print(f"  {'':28s} {'':22s} | trigram lag1/2/3 "
          + " ".join(f"{tag(a)} {ci_txt(l, h)}" for a, l, h in zip(aucs[3:], lo[3:], hi[3:])))
    print(f"  {'':28s} near-duplicate (content cosine >= 0.8) of a previous-run sentence: changes {dup_c:.1%}, "
          f"no-changes {dup_n:.1%}")
    return {"n": int(len(y)), "changes": int(y.sum()),
            "content_lag": [{"auc": a, "ci": [l, h]} for a, l, h in zip(aucs[:3], lo[:3], hi[:3])],
            "trigram_lag": [{"auc": a, "ci": [l, h]} for a, l, h in zip(aucs[3:], lo[3:], hi[3:])],
            "near_dup_changes": dup_c, "near_dup_nochanges": dup_n}


def read_jsonl(path):
    return [(d["sentences"], d["labels"]) for d in
            (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())]


def length_cells(minw, dw, cap=40):
    return np.minimum(minw, cap).astype(np.int64) * 100 + np.minimum(dw, cap).astype(np.int64)


def profile(docs, vec, rng, boot, label):
    a, b, y, doc = [], [], [], []
    for di, (s, lab) in enumerate(docs):
        for i, l in enumerate(lab):
            a.append(s[i]); b.append(s[i + 1]); y.append(l); doc.append(di)
    y, doc = np.array(y), np.array(doc)
    surf = np.array([surface_distances(x, z) for x, z in zip(a, b)])
    wa, wb = np.array([len(x.split()) for x in a], float), np.array([len(x.split()) for x in b], float)
    Va, Vb = vec.transform(a), vec.transform(b)
    ovl = np.asarray(Va.multiply(Vb).sum(axis=1)).ravel()
    L = length_cells(np.minimum(wa, wb), np.abs(wa - wb))
    LB = L * 10 + quantile_bins(ovl, 5, True)
    out = {}
    for name, g in (("unmatched", np.zeros(len(y), np.int64)), ("exact length", L), ("length x topic", LB)):
        aucs = [grouped_auc(surf[:, j], y, g) for j in range(3)]
        lo, hi = boot_ci(lambda i, cid: np.array([grouped_auc(surf[i, j], y[i], g[i]) for j in range(3)]),
                         doc_index(doc), boot, rng)
        both = np.zeros(len(y), bool)
        for c in np.unique(g):
            m = g == c
            if 0 < y[m].sum() < m.sum():
                both |= m
        print(f"  {label:28s} {name:15s} fw / trigram / punct "
              + "  ".join(f"{tag(x)} {ci_txt(l, h)}" for x, l, h in zip(aucs, lo, hi))
              + f"   (changes in informative cells: {both[y == 1].mean():.0%})")
        out[name] = {"auc": aucs, "ci": [[l, h] for l, h in zip(lo, hi)], "informative_change_share": both[y == 1].mean()}
    return out


def main():
    a = parse_args()
    rng = np.random.RandomState(a.seed)
    t0 = time.time()
    report = {}

    # ------------------------------------------------ 1 + 2: lag test and profiles
    print("=" * 110)
    print("1. LAG TEST: AUC of (low) content overlap and trigram dissimilarity between the new author's first sentence and")
    print("   the previous author's last (lag 1), second-to-last (lag 2) and third-to-last (lag 3) sentence.")
    print("   < 0.5 = changes MORE similar than no-changes. Splice filter: reversal at lag 1 that fades; echoing: persists.")
    print("=" * 110)
    val = {t: load_split(a.pan2025_root, t, "validation") for t in TIERS}
    train_s = [s for t in TIERS for sents, _ in load_split(a.pan2025_root, t, "train") for s in sents]
    vec25 = content_vectorizer(train_s)
    report["lag"] = {}
    for t in TIERS:
        report["lag"][f"pan2025_{t}"] = lag_table(val[t], vec25, rng, a.boot, f"PAN 2025 {t} (validation)")
    cons = {}
    for name in [n.strip() for n in a.construction_names.split(",") if n.strip()]:
        for kind in ("random", "same_topic", "content_splice", "style_splice", "topic_style"):
            f = Path(a.constructions_dir) / f"{name}_{kind}.jsonl"
            if f.exists():
                cons[(name, kind)] = read_jsonl(f)
    for name in sorted({n for n, _ in cons}):
        sents = [s for (n, _), docs in cons.items() if n == name for d in docs for s in d[0]]
        vecc = content_vectorizer(sents)
        for (n, kind), docs in cons.items():
            if n == name:
                report["lag"][f"{n}_{kind}"] = lag_table(docs, vecc, rng, a.boot, f"{n} {kind}")
    print(f"(lag test done, {time.time() - t0:.0f}s)")

    print("\n" + "=" * 110)
    print("2. MECHANISM PROFILE: surface cues unmatched, at exact matched length, and at length x content overlap")
    print("   (compare PAN 2025 Hard validation: exact length 0.490 / 0.333 / 0.546; length x topic 0.533 / 0.427 / 0.607)")
    print("=" * 110)
    report["profile"] = {}
    report["profile"]["pan2025_hard"] = profile(val["hard"], vec25, rng, a.boot, "PAN 2025 hard (validation)")
    for name in sorted({n for n, _ in cons}):
        sents = [s for (n, _), docs in cons.items() if n == name for d in docs for s in d[0]]
        vecc = content_vectorizer(sents)
        for kind in ("random", "style_splice", "topic_style"):
            if (name, kind) in cons:
                report["profile"][f"{name}_{kind}"] = profile(cons[(name, kind)], vecc, rng, a.boot, f"{name} {kind}")
    print(f"(profiles done, {time.time() - t0:.0f}s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(clean(report), indent=1), encoding="utf-8")

    # ------------------------------------------------ 3: one shared TF-IDF for both editions
    if a.pan2026_root:
        print("\n" + "=" * 110)
        print("3. ONE SHARED content TF-IDF (equal train samples of PAN 2025 and PAN 2026): overlap at changes vs within")
        print("=" * 110)
        tr26 = [s for t in TIERS for sents, _ in load_split(a.pan2026_root, t, "train") for s in sents]
        pick = lambda xs: [xs[i] for i in rng.choice(len(xs), min(a.fit_sample, len(xs)), replace=False)]
        shared = content_vectorizer(pick(train_s) + pick(tr26))
        del tr26
        report["shared_overlap"] = {}
        for ed, root in (("PAN 2025", a.pan2025_root), ("PAN 2026", a.pan2026_root)):
            for t in TIERS:
                docs = val[t] if ed == "PAN 2025" else load_split(root, t, "validation")
                ov, y = [], []
                for sents, lab in docs:
                    V = shared.transform(sents)
                    ov.append(cos_rows(V, np.arange(len(lab)), np.arange(1, len(lab) + 1)))
                    y.append(np.asarray(lab))
                ov, y = np.concatenate(ov), np.concatenate(y)
                c, w = ov[y == 1], ov[y == 0]
                print(f"  {ed} {t:6s} change mean {c.mean():.4f} (median {np.median(c):.4f}) | within mean {w.mean():.4f} "
                      f"(median {np.median(w):.4f}) | ratio {c.mean() / max(w.mean(), 1e-12):.2f} | AUC {tag(auc_fast(-ov, y))}")
                report["shared_overlap"][f"{ed} {t}"] = {"change_mean": c.mean(), "change_median": float(np.median(c)),
                                                       "within_mean": w.mean(), "within_median": float(np.median(w)),
                                                       "auc": auc_fast(-ov, y)}
        Path(a.out).write_text(json.dumps(clean(report), indent=1), encoding="utf-8")
        print(f"(shared TF-IDF done, {time.time() - t0:.0f}s)")

    # ------------------------------------------------ 4: boilerplate edge rule and train overlap (PAN 2025 test)
    print("\n" + "=" * 110)
    print("4. BOILERPLATE on the PAN 2025 test set (lines with >= 10 copies across the test documents)")
    print("=" * 110)
    root = Path(a.pan2025_root)
    test = {}
    for t in TIERS:
        docs = []
        for txt in sorted((root / t / "test").glob("problem-*.txt")):
            truth = txt.with_name(f"truth-{txt.stem}.json")
            if not truth.exists():
                continue
            s = [x.strip() for x in txt.read_text(encoding="utf-8").strip().split("\n") if x.strip()]
            ch = json.loads(truth.read_text(encoding="utf-8")).get("changes")
            if ch is None:
                continue
            n = max(0, len(s) - 1)
            docs.append((s, (list(ch) + [0] * n)[:n]))
        test[t] = docs
    tfreq = Counter(line for t in TIERS for sents, _ in test[t] for line in sents)
    rep_lines = {s for s, c in tfreq.items() if c >= 10}
    train_set = set(train_s)
    val_set = {s for t in TIERS for sents, _ in val[t] for s in sents}
    in_train = sum(1 for s in rep_lines if s in train_set)
    in_val = sum(1 for s in rep_lines if s in val_set)
    print(f"  {len(rep_lines)} distinct test lines occur >= 10 times; {in_train} of them also occur in the TRAIN text, "
          f"{in_val} in the validation text")
    report["boilerplate"] = {"n_repeated_test_lines": len(rep_lines), "in_train": in_train, "in_val": in_val, "tiers": {}}
    files = [Path(f) for f in a.pred_files.split(",") if f.strip()]

    def macro(y, p):
        out = []
        for c in (0, 1):
            tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
            d = 2 * tp + fp + fn
            out.append(2 * tp / d if d else 0.0)
        return float(np.mean(out))

    for t in TIERS:
        y = np.array([c for _, ch in test[t] for c in ch], int)
        lens = np.array([len(ch) for _, ch in test[t]])
        ra = np.concatenate([np.array([x in rep_lines for x in s[:-1]], bool) for s, _ in test[t] if len(s) > 1])
        rb = np.concatenate([np.array([x in rep_lines for x in s[1:]], bool) for s, _ in test[t] if len(s) > 1])
        near = ra | rb
        edge = (ra ^ rb).astype(int)
        r = {"near_share": near.mean(), "edge_rule_near": macro(y[near], edge[near]) if near.any() else float("nan"),
             "both_repeated_change_rate": float(y[ra & rb].mean()) if (ra & rb).any() else None,
             "one_repeated_change_rate": float(y[ra ^ rb].mean()) if (ra ^ rb).any() else None, "desaf_near": {}}
        print(f"  {t.upper():6s} near repeated {near.mean():.1%}; change rate when exactly one side is repeated "
              f"{r['one_repeated_change_rate'] if r['one_repeated_change_rate'] is not None else float('nan'):.3f}, "
              f"when both are {r['both_repeated_change_rate'] if r['both_repeated_change_rate'] is not None else float('nan'):.3f}; "
              f"edge rule macro-F1 on near-repeated boundaries {r['edge_rule_near']:.4f}")
        for f in files:
            if not f.exists():
                continue
            z = np.load(f)
            if not (np.array_equal(z[f"{t}__labels"].astype(int), y) and np.array_equal(z[f"{t}__lengths"], lens)):
                print(f"    {f}: does not line up with the test text (skipped)")
                continue
            p = z[f"{t}__preds"].astype(int)
            r["desaf_near"][str(f)] = macro(y[near], p[near]) if near.any() else float("nan")
            print(f"    {str(f):55s} DeSAF on near-repeated {r['desaf_near'][str(f)]:.4f}")
        report["boilerplate"]["tiers"][t] = r
    Path(a.out).write_text(json.dumps(clean(report), indent=1), encoding="utf-8")
    print(f"\nSaved {a.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
