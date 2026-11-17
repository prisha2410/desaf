#!/usr/bin/env python3
"""
Clue-only and position-only baselines for PAN 2025 (version 2): how much do the five simple clues explain, in which
direction do models have to read them, and what does a model lose without the level? CPU only, a few minutes.

Feature sets per boundary (sentences s_i, s_i+1), oriented so that a larger value means "more different":
  clues      content dissimilarity (minus the cosine of content-word TF-IDF vectors; English stop words removed; fitted
             on the TRAIN documents' sentences), |log word-count ratio|, function-word distance, character-trigram
             dissimilarity, punctuation distance (the last three with the same code as the clue AUCs in the paper)
  position   relative position of the boundary in its document only (text-free; does not reveal document length)
  structure  boundary index, index counted from the end, relative position, number of sentences (text-free)
  doclen     number of sentences in the document (text-free; in pooled models it could reveal the level)
Models: fitted on TRAIN, decision threshold chosen on VALIDATION with the official pooled two-class macro-F1, TEST
scored once.
  random    a uniform random score (same threshold search): the empirical chance level of this pipeline
  single    best single clue: each clue oriented by the sign of its own TRAIN AUC, the best one chosen on VALIDATION
  logreg    logistic regression on the standardised clues
  gbt       gradient-boosted trees (HistGradientBoostingClassifier, random_state 0), unconstrained
  gbt_rev   gbt with each clue constrained monotone in the direction of its own TRAIN AUC on that level
            (the sign vectors are printed and saved)
  gbt_cls   gbt with every clue constrained monotone in the classical direction (more different -> change)
  per-level models get the level (one model and threshold per level, like the oracle flag); pooled models get no
  level input and are scored with one global threshold and, separately, with per-level thresholds.
Also printed: cross-level transfer (logreg fitted on one level's TRAIN, ROC-AUC on another level's TEST), logistic
partial coefficients with document-bootstrap CIs next to each clue's single-clue TRAIN AUC, paired gaps, and paired
comparisons with DeSAF's saved Hard test predictions (checkpoints/seed4X/test_predictions.npz, if present and aligned).
CIs: document bootstrap. Macro-F1: 2,000 resamples, the same for every model on a level (gaps are paired). ROC-AUC:
500 resamples, the same draws for both scores in a paired AUC gap. Hard results are also given on boundaries that do
not touch a line repeated >= 10 times across the test documents of all levels.

Chance level. Under two-class macro-F1 a guess at the change rate p scores 0.5 in expectation, while predict-0 scores
(1-p)/(2-p) = 0.439 / 0.440 / 0.453 on Easy / Medium / Hard. Gains are therefore measured from 0.5, not from predict-0.

Revision note. Version 1 of these rules (fixed earlier on 2026-11-17) measured gains from predict-0. A code review on
synthetic data then showed that this gives a model with no signal about +0.05 for free, so that version-1 Q5 fails
even for a useless position model, and that the version-1 position features (index and index from the end) also
reveal document length. Version 2 below was fixed after that review and before any output of this script on the real
data was seen in the analysis.

Fixed before running (version 2, 2026-11-17). Hard level, official test macro-F1, unless stated otherwise:
  Q1  the five-clue gbt beats the best single clue by >= 0.02, paired 95% CI lower bound > 0.
  Q2  gbt is above 0.5, gbt_rev keeps >= 90% of gbt's gain over 0.5, and gbt_cls keeps <= 25% of it.
  Q3  cross-level transfer: logreg fitted on Easy TRAIN ranks Hard TEST boundaries below chance (ROC-AUC CI upper bound
      < 0.5), and logreg fitted on Hard TRAIN ranks Easy TEST boundaries below chance (upper bound < 0.5).
  Q4  per-level gbt minus pooled gbt scored with per-level thresholds >= 0.02, paired CI lower bound > 0.
  Q4r per-level gbt minus pooled gbt, paired ROC-AUC gap, CI lower bound > 0.
  Q5  the per-level relative-position model has test ROC-AUC <= 0.55 on every level.
Reporting rule for the share of the gain over 0.5, with matched information: per-level gbt against the true-level
head (Hard 0.844); pooled gbt with one threshold against DeSAF without level (Hard 0.826).
  share < 0.40 -> "a minority of the gain"; 0.40-0.75 -> "roughly half to two-thirds"; > 0.75 -> "most of the gain".
Sentences allowed only if their condition holds:
  Q1: "combining the five clues adds to the best single clue"
  Q2 and all five Hard gbt_rev signs reversed: "on Hard the clues help only when read in the reversed direction";
      Q2 with k < 5 signs reversed: "on Hard the clues help only when each is read in its own Hard direction
      (k of 5 reversed)"
  Q3: "a clue model fitted on Easy ranks Hard boundaries worse than chance, and the reverse"
  Q4: "without the level as an input (threshold still set per level), the clues lose <gap> on Hard"
  Q4r: "without the level, the clue model ranks Hard boundaries worse"; with Q3 also: "... worse, consistent with
       clues that point opposite ways across levels"
  Q5: "position in the document alone carries little signal"
Otherwise: report the numbers only.

Sanity checks the log must pass before any number is used: predict-0 prints 0.439 / 0.440 / 0.453; random prints
about 0.5 on every level; each DeSAF file's own Hard macro-F1 matches its verified value (0.830 / 0.819 / 0.829 for
seeds 43 / 44 / 45).

Run from D:\\sass-tad:
    python -u scripts/clue_baselines.py 2>&1 | Tee-Object -FilePath logs\\clue_baselines.log
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.utils.data_loader import load_pan2025_full  # noqa: E402
from pan_feature_direction import surface_distances  # noqa: E402

TIERS = ["easy", "medium", "hard"]
CLUES = ["content", "length", "function_words", "trigram", "punctuation"]
CHANCE = 0.5                   # expected two-class macro-F1 of a guess at the change rate
TRUE_LEVEL_HEAD_HARD = 0.844   # REAL_VERIFIED_NUMBERS.md section 11 (9-run mean, official metric)
DESAF_HARD = 0.826             # 3-seed mean, official metric


# ----------------------------------------------------------------------------------------------- metrics
def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def auc(score, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def doc_counts(y, p, doc, n_docs, mask=None):
    """Per-document tp, fp, fn, tn of the change class (columns), optionally on a subset of boundaries."""
    m = np.ones(len(y), bool) if mask is None else mask
    cols = [((p == 1) & (y == 1) & m), ((p == 1) & (y == 0) & m), ((p == 0) & (y == 1) & m), ((p == 0) & (y == 0) & m)]
    return np.column_stack([np.bincount(doc, weights=c.astype(float), minlength=n_docs) for c in cols])


def macro_from_totals(T):
    tp, fp, fn, tn = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f0 = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1 + f0) / 2


class Boot:
    """Document bootstrap with one fixed set of resamples per level, so every comparison on a level is paired."""

    def __init__(self, doc, n_boot, rng, n_auc=500, auc_seed=0):
        self.doc = doc; self.n_docs = int(doc.max()) + 1
        self.W = np.stack([np.bincount(rng.choice(self.n_docs, self.n_docs), minlength=self.n_docs)
                           for _ in range(n_boot)]).astype(float)
        order = np.argsort(doc, kind="stable")
        self.groups = np.split(order, np.cumsum(np.bincount(doc, minlength=self.n_docs))[:-1])
        r = np.random.RandomState(auc_seed)   # one fixed set of index draws for every AUC CI and AUC gap on a level
        self.auc_ix = [np.concatenate([self.groups[d] for d in r.choice(self.n_docs, self.n_docs)]) for _ in range(n_auc)]

    def ci(self, y, p, mask=None):
        C = doc_counts(y, p, self.doc, self.n_docs, mask)
        point = float(macro_from_totals(C.sum(0)))
        lo, hi = np.percentile(macro_from_totals(self.W @ C), [2.5, 97.5])
        return point, [float(lo), float(hi)]

    def gap(self, y, p_a, p_b, mask=None):
        Ca = doc_counts(y, p_a, self.doc, self.n_docs, mask); Cb = doc_counts(y, p_b, self.doc, self.n_docs, mask)
        point = float(macro_from_totals(Ca.sum(0)) - macro_from_totals(Cb.sum(0)))
        d = macro_from_totals(self.W @ Ca) - macro_from_totals(self.W @ Cb)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return point, [float(lo), float(hi)]

    def auc_ci(self, score, y):
        vals = [auc(score[ix], y[ix]) for ix in self.auc_ix]
        return auc(score, y), [float(v) for v in np.percentile(vals, [2.5, 97.5])]

    def auc_gap(self, score_a, score_b, y):
        vals = [auc(score_a[ix], y[ix]) - auc(score_b[ix], y[ix]) for ix in self.auc_ix]
        return auc(score_a, y) - auc(score_b, y), [float(v) for v in np.percentile(vals, [2.5, 97.5])]


# ----------------------------------------------------------------------------------------------- data
def load(tier, split, root):
    """Boundaries in the order the prediction scripts use (DocLevelDataset skips documents with n <= 0)."""
    a, b, y, doc, idx, n_s, sents = [], [], [], [], [], [], []
    k = 0
    for d in load_pan2025_full(root, tier=tier, split=split):
        n = min(len(d.sentences) - 1, len(d.boundary_labels))
        if n <= 0:
            continue
        sents.append(d.sentences[:n + 1])
        for i in range(n):
            a.append(d.sentences[i]); b.append(d.sentences[i + 1]); y.append(int(d.boundary_labels[i]))
            doc.append(k); idx.append(i); n_s.append(n + 1)
        k += 1
    return dict(a=a, b=b, y=np.array(y), doc=np.array(doc), idx=np.array(idx), n_s=np.array(n_s), sents=sents)


def features(D, vec):
    a, b = D["a"], D["b"]
    Va, Vb = vec.transform(a), vec.transform(b)
    content = -np.asarray(Va.multiply(Vb).sum(axis=1)).ravel()
    wa = np.array([max(len(s.split()), 1) for s in a], float); wb = np.array([max(len(s.split()), 1) for s in b], float)
    length = np.abs(np.log(wa / wb))
    surf = np.array([surface_distances(p, q) for p, q in zip(a, b)])          # fw, trigram, punct
    n_b = D["n_s"] - 1; idx = D["idx"]
    rel = (idx / np.maximum(n_b - 1, 1)).astype(float)
    return {"clues": np.column_stack([content, length, surf[:, 0], surf[:, 1], surf[:, 2]]),
            "position": rel.reshape(-1, 1),
            "structure": np.column_stack([idx, n_b - 1 - idx, rel, D["n_s"]]).astype(float),
            "doclen": D["n_s"].reshape(-1, 1).astype(float)}


def best_prob_threshold(prob, y):
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
    scores = [macro(y, (prob >= t).astype(int)) for t in grid]
    k = int(np.argmax(scores))
    return float(grid[k])


def best_raw_threshold(score, y):
    grid = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    scores = [macro(y, (score >= t).astype(int)) for t in grid]
    k = int(np.argmax(scores))
    return float(grid[k]), float(scores[k])


def share_word(s):
    return "a minority of the gain" if s < 0.40 else ("roughly half to two-thirds" if s <= 0.75 else "most of the gain")


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--auc-boot", type=int, default=500)
    ap.add_argument("--coef-boot", type=int, default=200)
    ap.add_argument("--min-copies", type=int, default=10)
    ap.add_argument("--desaf", default="checkpoints/seed43/test_predictions.npz,checkpoints/seed44/test_predictions.npz,"
                                       "checkpoints/seed45/test_predictions.npz")
    ap.add_argument("--out", default="results/clue_baselines.json")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(0)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print("clue_baselines.py version 2 (gains measured from 0.5; rules in the docstring)", flush=True)
    data = {s: {t: load(t, s, a.data_root) for t in TIERS} for s in ("train", "validation", "test")}
    vec = TfidfVectorizer(max_features=20000, stop_words="english").fit(
        [x for t in TIERS for ss in data["train"][t]["sents"] for x in ss])
    F = {s: {t: features(data[s][t], vec) for t in TIERS} for s in data}
    Y = {s: {t: data[s][t]["y"] for t in TIERS} for s in data}
    for s in data:
        print(f"{s}: " + ", ".join(f"{t} {len(Y[s][t])} boundaries ({int(Y[s][t].sum())} changes)" for t in TIERS)
              + f" ({time.time() - t0:.0f}s)", flush=True)
    counts = Counter(x for t in TIERS for ss in data["test"][t]["sents"] for x in ss)
    rep = {x for x, c in counts.items() if c >= a.min_copies}
    clean = {t: np.array([(p not in rep) and (q not in rep) for p, q in zip(data["test"][t]["a"], data["test"][t]["b"])])
             for t in TIERS}
    print(f"{len(rep)} test lines occur >= {a.min_copies} times; boundaries not touching them: "
          + ", ".join(f"{t} {clean[t].mean():.1%}" for t in TIERS))
    boot = {t: Boot(data["test"][t]["doc"], a.boot, rng, n_auc=a.auc_boot) for t in TIERS}
    train_auc = {t: [auc(F["train"][t]["clues"][:, j], Y["train"][t]) for j in range(5)] for t in TIERS}
    signs = {t: [1 if v > 0.5 else -1 for v in train_auc[t]] for t in TIERS}

    R = {"version": 2, "rows": {}, "notes": {}, "train_auc": train_auc, "gbt_rev_signs": signs}
    preds, scores = {}, {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def save():   # after every section, so a late crash keeps what was already computed
        Path(a.out).write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

    def record(name, t, pred, score):
        preds[(name, t)] = pred; scores[(name, t)] = score
        f1, ci = boot[t].ci(Y["test"][t], pred)
        f1c, cic = boot[t].ci(Y["test"][t], pred, clean[t])
        R["rows"].setdefault(name, {})[t] = {"f1": f1, "ci": ci, "f1_clean": f1c, "ci_clean": cic,
                                              "auc": auc(score, Y["test"][t]) if score is not None else None}

    # references: predict-0 and a random score through the same threshold search
    rr = np.random.RandomState(1)
    for t in TIERS:
        record("predict-0", t, np.zeros_like(Y["test"][t]), None)
        sv, st = rr.rand(len(Y["validation"][t])), rr.rand(len(Y["test"][t]))
        record("random", t, (st >= best_prob_threshold(sv, Y["validation"][t])).astype(int), st)
    # best single clue
    for t in TIERS:
        best = None
        for j, c in enumerate(CLUES):
            sgn = 1.0 if train_auc[t][j] > 0.5 else -1.0
            thr, vf1 = best_raw_threshold(sgn * F["validation"][t]["clues"][:, j], Y["validation"][t])
            if best is None or vf1 > best[0]:
                best = (vf1, j, sgn, thr)
        _, j, sgn, thr = best
        sc = sgn * F["test"][t]["clues"][:, j]
        record("single", t, (sc >= thr).astype(int), sc)
        R["notes"].setdefault("single_clue", {})[t] = f"{CLUES[j]} ({'usual' if sgn > 0 else 'reversed'} direction)"

    def per_level(name, make, fs="clues"):
        for t in TIERS:
            m = make(t).fit(F["train"][t][fs], Y["train"][t])
            thr = best_prob_threshold(m.predict_proba(F["validation"][t][fs])[:, 1], Y["validation"][t])
            sc = m.predict_proba(F["test"][t][fs])[:, 1]
            record(name, t, (sc >= thr).astype(int), sc)

    per_level("logreg", lambda t: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
    per_level("gbt", lambda t: HGB(random_state=0))
    per_level("gbt_rev", lambda t: HGB(random_state=0, monotonic_cst=signs[t]))
    per_level("gbt_cls", lambda t: HGB(random_state=0, monotonic_cst=[1] * 5))
    per_level("position", lambda t: HGB(random_state=0), fs="position")
    per_level("structure", lambda t: HGB(random_state=0), fs="structure")
    per_level("doclen", lambda t: HGB(random_state=0), fs="doclen")
    print(f"per-level models done ({time.time() - t0:.0f}s)", flush=True)

    # pooled models: no level input
    for name, make in (("pooled_gbt", lambda: HGB(random_state=0)),
                       ("pooled_logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))):
        m = make().fit(np.vstack([F["train"][t]["clues"] for t in TIERS]), np.concatenate([Y["train"][t] for t in TIERS]))
        pv = {t: m.predict_proba(F["validation"][t]["clues"])[:, 1] for t in TIERS}
        g_thr = best_prob_threshold(np.concatenate([pv[t] for t in TIERS]), np.concatenate([Y["validation"][t] for t in TIERS]))
        for t in TIERS:
            sc = m.predict_proba(F["test"][t]["clues"])[:, 1]
            record(f"{name}_1thr", t, (sc >= g_thr).astype(int), sc)
            record(f"{name}_lvlthr", t, (sc >= best_prob_threshold(pv[t], Y["validation"][t])).astype(int), sc)
    print(f"pooled models done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ print the main table
    order = ["predict-0", "random", "single", "logreg", "gbt", "gbt_rev", "gbt_cls", "pooled_gbt_1thr",
             "pooled_gbt_lvlthr", "pooled_logreg_1thr", "pooled_logreg_lvlthr", "position", "structure", "doclen"]
    print(f"\n{'=' * 122}\nOfficial test macro-F1 [document-bootstrap 95% CI]; Hard also without repeated-line boundaries, "
          f"and Hard ROC-AUC on all Hard boundaries\n{'=' * 122}")
    print(f"  {'model':22s}{'Easy':>24s}{'Medium':>24s}{'Hard':>24s}{'Hard, no repeated':>26s}{'Hard AUC(all)':>14s}")
    for name in order:
        r = R["rows"][name]
        cells = "".join(f"{r[t]['f1']:>8.3f} [{r[t]['ci'][0]:.3f}, {r[t]['ci'][1]:.3f}]" for t in TIERS)
        hh = r["hard"]
        print(f"  {name:22s}{cells}{hh['f1_clean']:>10.3f} [{hh['ci_clean'][0]:.3f}, {hh['ci_clean'][1]:.3f}]"
              f"{(hh['auc'] if hh['auc'] is not None else float('nan')):>14.3f}")
    print("  ROC-AUC per level (test): " + "; ".join(
        f"{n} " + "/".join(f"{R['rows'][n][t]['auc']:.3f}" for t in TIERS)
        for n in ["single", "logreg", "gbt", "gbt_rev", "gbt_cls", "pooled_gbt_1thr", "position", "structure", "doclen"]))
    print("  best single clue: " + "; ".join(f"{t} {R['notes']['single_clue'][t]}" for t in TIERS))
    print("  gbt_rev signs (+1 usual, -1 reversed; order " + ", ".join(CLUES) + "): "
          + "; ".join(f"{t} {signs[t]}" for t in TIERS))
    print("  note: pooled models get no level input; 'doclen' and 'structure' could reveal the level, so they are only "
          "used per level")
    save()

    # ------------------------------------------------------------------ logistic coefficients
    print(f"\nLogistic partial coefficients (standardised clues, TRAIN; 95% CI from {a.coef_boot} document resamples) "
          f"and each clue's own TRAIN AUC. + = usual direction, - = reversed")
    R["logistic"] = {}
    for t in TIERS:
        X, yy, dd = F["train"][t]["clues"], Y["train"][t], data["train"][t]["doc"]
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        coef = LogisticRegression(max_iter=2000).fit(Xs, yy).coef_[0]
        order_d = np.argsort(dd, kind="stable")
        groups = np.split(order_d, np.cumsum(np.bincount(dd))[:-1])
        bs = []
        for _ in range(a.coef_boot):
            ix = np.concatenate([groups[d] for d in rng.choice(len(groups), len(groups))])
            bs.append(LogisticRegression(max_iter=2000).fit(Xs[ix], yy[ix]).coef_[0])
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        R["logistic"][t] = {c: {"coef": float(coef[j]), "ci": [float(lo[j]), float(hi[j])], "train_auc": train_auc[t][j]}
                            for j, c in enumerate(CLUES)}
        print(f"  {t:7s} " + " | ".join(f"{c} {coef[j]:+.2f} [{lo[j]:+.2f}, {hi[j]:+.2f}] (AUC {train_auc[t][j]:.3f})"
                                         for j, c in enumerate(CLUES)))
    save()

    # ------------------------------------------------------------------ cross-level transfer
    print(f"\nCross-level transfer: logreg fitted on the row level's TRAIN, ROC-AUC [95% CI, {a.auc_boot} document "
          f"resamples] on the column level's TEST")
    R["transfer"] = {}
    for s in TIERS:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F["train"][s]["clues"], Y["train"][s])
        row = {}
        for t in TIERS:
            v, ci = boot[t].auc_ci(m.predict_proba(F["test"][t]["clues"])[:, 1], Y["test"][t])
            row[t] = {"auc": v, "ci": ci}
        R["transfer"][s] = row
        print(f"  from {s:7s}" + "".join(f"   {t} {row[t]['auc']:.3f} [{row[t]['ci'][0]:.3f}, {row[t]['ci'][1]:.3f}]" for t in TIERS))
    save()

    # ------------------------------------------------------------------ paired gaps on Hard
    h = "hard"; yh = Y["test"][h]; bh = boot[h]
    G, GA = {}, {}
    G["gbt_minus_single"] = bh.gap(yh, preds[("gbt", h)], preds[("single", h)])
    G["gbt_minus_pooled_gbt_lvlthr"] = bh.gap(yh, preds[("gbt", h)], preds[("pooled_gbt_lvlthr", h)])
    G["gbt_minus_pooled_gbt_1thr"] = bh.gap(yh, preds[("gbt", h)], preds[("pooled_gbt_1thr", h)])
    G["pooled_gbt_lvlthr_minus_1thr"] = bh.gap(yh, preds[("pooled_gbt_lvlthr", h)], preds[("pooled_gbt_1thr", h)])
    G["logreg_minus_pooled_logreg_lvlthr"] = bh.gap(yh, preds[("logreg", h)], preds[("pooled_logreg_lvlthr", h)])
    G["gbt_rev_minus_gbt_cls"] = bh.gap(yh, preds[("gbt_rev", h)], preds[("gbt_cls", h)])
    GA["gbt_minus_pooled_gbt"] = bh.auc_gap(scores[("gbt", h)], scores[("pooled_gbt_1thr", h)], yh)
    GA["logreg_minus_pooled_logreg"] = bh.auc_gap(scores[("logreg", h)], scores[("pooled_logreg_1thr", h)], yh)
    lengths = np.bincount(data["test"][h]["doc"])
    R["desaf_hard"] = {}
    for f in [x.strip() for x in a.desaf.split(",") if x.strip()]:
        p = Path(f)
        if not p.exists():
            print(f"  ({f} not found; DeSAF comparison skipped for it)"); continue
        try:
            z = np.load(p)
            ok = (np.array_equal(z[f"{h}__labels"].astype(int), yh)
                  and np.array_equal(z[f"{h}__lengths"].astype(int), lengths))
            pd_ = z[f"{h}__preds"].astype(int)
        except Exception as e:   # a damaged or differently keyed file must not stop the run
            print(f"  ({f} could not be read: {e}; skipped)"); continue
        if not ok:
            print(f"  ({f} does not line up with the Hard test boundaries; skipped)"); continue
        tag = str(p.parent).replace("\\", "/")
        R["desaf_hard"][tag] = bh.ci(yh, pd_)
        G[f"desaf[{tag}]_minus_pooled_gbt_1thr"] = bh.gap(yh, pd_, preds[("pooled_gbt_1thr", h)])
        G[f"desaf[{tag}]_minus_gbt"] = bh.gap(yh, pd_, preds[("gbt", h)])
    R["gaps_hard"] = G; R["auc_gaps_hard"] = GA
    if R["desaf_hard"]:
        print("\nDeSAF files used (own Hard macro-F1 [95% CI]; sanity check against the verified values): "
              + "; ".join(f"{k} {v[0]:.4f} [{v[1][0]:.3f}, {v[1][1]:.3f}]" for k, v in R["desaf_hard"].items()))
    print("\nPaired gaps on Hard, macro-F1 (same 2,000 document resamples) [95% CI]")
    for k, (v, ci) in G.items():
        print(f"  {k:52s} {v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"Paired gaps on Hard, ROC-AUC (same {a.auc_boot} document resamples) [95% CI]")
    for k, (v, ci) in GA.items():
        print(f"  {k:52s} {v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    save()

    # ------------------------------------------------------------------ fixed-before-running checks (version 2)
    rows = R["rows"]
    gain = lambda name: rows[name][h]["f1"] - CHANCE
    share_lvl = gain("gbt") / (TRUE_LEVEL_HEAD_HARD - CHANCE)
    share_pool = gain("pooled_gbt_1thr") / (DESAF_HARD - CHANCE)
    n_rev = sum(1 for s in signs[h] if s < 0)
    q1 = G["gbt_minus_single"][0] >= 0.02 and G["gbt_minus_single"][1][0] > 0
    q2 = gain("gbt") > 0 and gain("gbt_rev") >= 0.9 * gain("gbt") and gain("gbt_cls") <= 0.25 * gain("gbt")
    q3 = R["transfer"]["easy"]["hard"]["ci"][1] < 0.5 and R["transfer"]["hard"]["easy"]["ci"][1] < 0.5
    q4 = G["gbt_minus_pooled_gbt_lvlthr"][0] >= 0.02 and G["gbt_minus_pooled_gbt_lvlthr"][1][0] > 0
    q4r = GA["gbt_minus_pooled_gbt"][1][0] > 0
    q5 = all(rows["position"][t]["auc"] <= 0.55 for t in TIERS)
    R["checks"] = {"Q1": q1, "Q2": q2, "Q2_hard_signs_reversed": n_rev, "Q3": q3, "Q4": q4, "Q4r": q4r, "Q5": q5,
                   "share_per_level_vs_true_level_head": share_lvl, "share_pooled_vs_desaf": share_pool}
    p0s = "/".join("%.3f" % rows["predict-0"][t]["f1"] for t in TIERS)
    rnd = "/".join("%.3f" % rows["random"][t]["f1"] for t in TIERS)
    print(f"\nSanity: predict-0 {p0s} (expect 0.439/0.440/0.453); random {rnd} (expect about 0.5)")
    print(f"Gains over 0.5 on Hard: gbt {gain('gbt'):+.3f}, gbt_rev {gain('gbt_rev'):+.3f}, gbt_cls {gain('gbt_cls'):+.3f}, "
          f"pooled gbt (1 thr) {gain('pooled_gbt_1thr'):+.3f}; Hard gbt_rev signs reversed: {n_rev} of 5")
    print(f"Share of the gain over 0.5 on Hard: per-level gbt {share_lvl:.3f} of the true-level head's "
          f"({share_word(share_lvl)}); pooled gbt {share_pool:.3f} of DeSAF's ({share_word(share_pool)})")
    print("Fixed-before-running checks (version 2): " + " | ".join(
        f"{k} {'holds' if v else 'FAILS'}" for k, v in [("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", q4), ("Q4r", q4r), ("Q5", q5)]))
    save()
    print(f"Saved {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
