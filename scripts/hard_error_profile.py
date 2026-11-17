#!/usr/bin/env python3
"""
Where does DeSAF miss author changes, and where does the oracle tier flag help?  CPU only; saved test predictions.

Links the detector to the cue-direction finding: at PAN 2025 Hard changes the two sentences are MORE alike than
same-author neighbours (shared content words, similar length, shared character trigrams). If that is what makes Hard
hard, DeSAF should miss changes most where the junction is most similar, and the tier flag (which changes the ranking
within a tier) should help most there.

Junction similarity of a boundary = mean of three within-tier percentile ranks: content-word overlap (TF-IDF without
stop words, fitted on the train sentences), minus |log length ratio|, minus character-trigram dissimilarity.
Boundaries are split into tertiles of this score (cut points over all boundaries of the tier, so the same bins serve
changes and no-changes). Per tertile and run: recall of changes, false-positive rate on no-changes, pooled two-class
macro-F1, and ROC-AUC inside the tertile (ranking among equally similar pairs).

Runs: the original gated DeSAF (checkpoints/seed4X/test_predictions.npz) and, when present, the stage-2 heads without
and with the tier flag (results/gate_lab/seed4X_gate/final_concat_s{1,2,3}_test.npz and final_concat_tier_s*_test.npz).
Every file is checked against the text (same labels and document lengths) before use.

Pre-registered (2026-11-16), PAN 2025 Hard test:
  P1  the original DeSAF's recall of changes is at least 0.10 lower in the most-similar tertile than in the least
      similar one, for all 3 encoders (document-bootstrap CI of the mean difference excludes 0).
  P2  the tier head's recall gain over the no-flag head (mean of 3 heads) is larger in the most-similar tertile than in
      the least-similar one, for all 3 encoders.
If P1 holds: "DeSAF misses Hard changes where the junction is most alike". If P2 holds too: "the tier flag helps
where similarity at the junction misleads". Otherwise report the table without those sentences.

Run from D:\\sass-tad:
    python scripts/hard_error_profile.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.utils.data_loader import load_pan2025_full  # noqa: E402
from pan_feature_direction import surface_distances  # noqa: E402

TIERS = ["easy", "medium", "hard"]


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def auc(prob, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(prob)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct_rank(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 0.5) / len(x)


def load_tier(tier, data_root):
    """Pairs in exactly the order the prediction scripts use (load_pan2025_full, DocLevelDataset skips n <= 0)."""
    docs = load_pan2025_full(data_root, tier=tier, split="test")
    a, b, y, doc, lengths, ids = [], [], [], [], [], []
    for di, d in enumerate(docs):
        n = min(len(d.sentences) - 1, len(d.boundary_labels))
        if n <= 0:
            continue
        lengths.append(n); ids.append(d.doc_id)
        for i in range(n):
            a.append(d.sentences[i]); b.append(d.sentences[i + 1]); y.append(int(d.boundary_labels[i])); doc.append(di)
    return a, b, np.array(y), np.array(doc), np.array(lengths), ids


def cache_matches(name, tier, doc_ids, y, cache_dir):
    """The gate_lab files carry no doc_ids; their order is the stage-2 cache's. Check that cache against the text."""
    f = Path(cache_dir) / f"{name}_test.npz"
    if not f.exists():
        return None
    cz = np.load(f)
    k = TIERS.index(tier)
    dt = cz["doc_tier"].astype(int)
    same_ids = [str(x) for x in cz["doc_ids"][dt == k]] == list(doc_ids)
    same_y = np.array_equal(cz["labels"][np.repeat(dt, cz["doc_lengths"].astype(int)) == k].astype(int), y)
    return bool(same_ids and same_y)


def runs_for_tier(tier, y, lengths, names, gate_lab_dir, seeds, doc_ids=None, cache_dir="checkpoints/stage2_cache"):
    runs = {}
    for name in names:
        enc = name.replace("_gate", "")
        ok = cache_matches(name, tier, doc_ids, y, cache_dir) if doc_ids is not None else None
        if ok is False:
            print(f"  ({cache_dir}/{name}_test.npz is not in the text's document order: the gate_lab head files of "
                  f"{name} are skipped)")
        paths = {f"original {enc}": [Path(f"checkpoints/{enc}/test_predictions.npz")],
                 f"no-flag head {enc}": [Path(gate_lab_dir) / name / f"final_concat_s{s}_test.npz" for s in seeds],
                 f"tier head {enc}": [Path(gate_lab_dir) / name / f"final_concat_tier_s{s}_test.npz" for s in seeds]}
        for label, files in paths.items():
            if ok is False and not label.startswith("original"):
                continue
            expected = len(files)
            files = [f for f in files if f.exists()]
            if not files:
                print(f"  ({label}: no files found)")
                continue
            if len(files) < expected:
                print(f"  (WARNING {label}: {len(files)} of {expected} files found)")
            preds, probs = [], []
            for f in files:
                z = np.load(f)
                if not (np.array_equal(z[f"{tier}__labels"].astype(int), y) and
                        np.array_equal(z[f"{tier}__lengths"].astype(int), lengths)):
                    print(f"  ({f} does not line up with the {tier} test text; skipped)")
                    continue
                preds.append(z[f"{tier}__preds"].astype(int)); probs.append(z[f"{tier}__probs"].astype(float))
            if preds:
                runs[label] = (preds, probs)
                if ok is None and not label.startswith("original"):
                    print(f"  ({label}: order checked by labels and lengths only; stage-2 cache not found)")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--names", default="seed43_gate,seed44_gate,seed45_gate")
    ap.add_argument("--gate-lab-dir", default="results/gate_lab")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/hard_error_profile.json")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()

    from sklearn.feature_extraction.text import TfidfVectorizer
    train_sents = [s for t in TIERS for d in load_pan2025_full(a.data_root, tier=t, split="train") for s in d.sentences]
    vec = TfidfVectorizer(max_features=20000, stop_words="english").fit(train_sents)
    print(f"Content TF-IDF fitted on {len(train_sents)} train sentences ({time.time() - t0:.0f}s)")

    report = {}
    for tier in TIERS:
        A, B, y, doc, lengths, ids = load_tier(tier, a.data_root)
        Va, Vb = vec.transform(A), vec.transform(B)
        ovl = np.asarray(Va.multiply(Vb).sum(axis=1)).ravel()
        wa = np.array([max(len(s.split()), 1) for s in A], float); wb = np.array([max(len(s.split()), 1) for s in B], float)
        loglen = np.abs(np.log(wa / wb))
        tri = np.array([surface_distances(p, q)[1] for p, q in zip(A, B)])
        sim = (pct_rank(ovl) + pct_rank(-loglen) + pct_rank(-tri)) / 3
        cuts = np.quantile(sim, [1 / 3, 2 / 3])
        tert = np.digitize(sim, cuts)                                  # 0 = least similar, 2 = most similar
        sim_auc = auc(sim, y)
        print(f"\n{'=' * 118}\n{tier.upper()} test: {len(y)} boundaries, {int(y.sum())} changes; junction-similarity AUC "
              f"(P(change more similar)) = {sim_auc:.3f}; changes per tertile (least -> most similar): "
              + " / ".join(str(int(y[tert == k].sum())) for k in range(3)) + f"\n{'=' * 118}")
        runs = runs_for_tier(tier, y, lengths, names, a.gate_lab_dir, seeds, ids, a.cache_dir)
        rep = {"n": int(len(y)), "changes": int(y.sum()), "changes_per_tertile": [int(y[tert == k].sum()) for k in range(3)],
               "runs": {}}
        print(f"  {'run':20s}" + "".join(f"{'T' + str(k) + ' rec / FPR / macro / AUC':>34s}" for k in range(3)))
        for label, (preds, probs) in runs.items():
            cells, stats = [], []
            for k in range(3):
                m = tert == k
                rec = np.mean([p[m & (y == 1)].mean() for p in preds])
                fpr = np.mean([p[m & (y == 0)].mean() for p in preds])
                mac = np.mean([macro(y[m], p[m]) for p in preds])
                au = np.mean([auc(q[m], y[m]) for q in probs])
                stats.append({"recall": rec, "fpr": fpr, "macro": mac, "auc": au})
                cells.append(f"{rec:7.3f} {fpr:6.3f} {mac:7.3f} {au:7.3f}".rjust(34))
            rep["runs"][label] = stats
            print(f"  {label:20s}" + "".join(cells))
        # P1: recall gap (least - most similar) for the original DeSAF, with a document-bootstrap CI of the mean over encoders
        orig = [r for r in runs if r.startswith("original")]
        if orig:
            docs_u = np.unique(doc); idx_by_doc = {d: np.flatnonzero(doc == d) for d in docs_u}
            def gap(ix, preds_list):
                yy, tt = y[ix], tert[ix]
                return np.mean([p[ix][(tt == 0) & (yy == 1)].mean() - p[ix][(tt == 2) & (yy == 1)].mean() for p in preds_list])
            P = [runs[r][0][0] for r in orig]
            point = gap(np.arange(len(y)), P)
            bs = [gap(np.concatenate([idx_by_doc[d] for d in rng.choice(docs_u, len(docs_u))]), P) for _ in range(a.boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            per = [rep["runs"][r][0]["recall"] - rep["runs"][r][2]["recall"] for r in orig]
            rep["P1_recall_gap_least_minus_most"] = {"mean": float(point), "ci": [float(lo), float(hi)], "per_encoder": per,
                                                     "evaluable": len(orig) == len(names)}
            if len(orig) < len(names):
                print(f"  (P1 NOT fully evaluable: {len(orig)} of {len(names)} encoders)")
            print(f"  P1 (original DeSAF) recall, least - most similar tertile: per encoder "
                  + " / ".join(f"{g:+.3f}" for g in per) + f"; mean {point:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        # P2: tier-head recall gain over the no-flag head, by tertile
        p2 = []
        for name in names:
            enc = name.replace("_gate", "")
            nf, th = rep["runs"].get(f"no-flag head {enc}"), rep["runs"].get(f"tier head {enc}")
            if nf and th:
                g = [th[k]["recall"] - nf[k]["recall"] for k in range(3)]
                ga = [th[k]["auc"] - nf[k]["auc"] for k in range(3)]
                p2.append({"encoder": enc, "recall_gain": g, "auc_gain": ga})
                print(f"  P2 {enc}: tier-head minus no-flag recall gain by tertile " + " / ".join(f"{x:+.3f}" for x in g)
                      + " | within-tertile AUC gain " + " / ".join(f"{x:+.4f}" for x in ga))
        rep["P2"] = p2
        if len(p2) < len(names):
            print(f"  (P2 NOT fully evaluable: {len(p2)} of {len(names)} encoders have both heads)")
        report[tier] = rep
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
