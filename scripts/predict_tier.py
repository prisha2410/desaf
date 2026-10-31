#!/usr/bin/env python3
"""
Can the tier (easy / medium / hard) be predicted from the document itself?

If yes, the tier-aware head no longer needs PAN to say which tier a document belongs
to, and works on any document. Features are document-level summaries of the cached
per-boundary statistics (topic overlap, surface distances, stylometric distance,
number of boundaries). No author-change labels are used.

The classifier is trained on VALIDATION documents (their tier is part of the labelled
data) and chosen by 5-fold CV on validation; the TEST documents' tiers are predicted once.

Usage:
    python scripts/predict_tier.py --name seed43_gate
Writes checkpoints/stage2_cache/{name}_test_tierpred.npz for gate_lab.py --test-tier-file.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TIERS = ["easy", "medium", "hard"]
CROSSOVER = 0.075


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    p.add_argument("--with-encoder", action="store_true",
                   help="also use the encoder's mean [CLS] vector and predicted-change rate")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def doc_features(z, with_encoder):
    lengths = z["doc_lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    ovl, surf = z["overlap"], z["surf"]
    style_l2 = np.linalg.norm(z["style"], axis=1)
    if with_encoder:
        h, probs = z["h"].astype(np.float32), z["probs"]
    rows = []
    for i in range(len(lengths)):
        s, e = offsets[i], offsets[i + 1]
        o = ovl[s:e]
        f = [len(o), o.mean(), o.std(), o.min(), o.max(), np.median(o),
             (o > CROSSOVER).mean(), (o == 0).mean()]
        for col in (surf[s:e, 0], surf[s:e, 1], surf[s:e, 2], style_l2[s:e]):
            f += [col.mean(), col.std(), col.min(), col.max()]
        if with_encoder:
            pr = probs[s:e]
            f += [pr.mean(), pr.std(), (pr >= 0.5).mean()]
            f += list(h[s:e].mean(0))
        rows.append(f)
    return np.array(rows, dtype=np.float64)


def main():
    args = parse_args()
    cache = Path(args.cache_dir)
    val = np.load(cache / f"{args.name}_validation.npz")
    test = np.load(cache / f"{args.name}_test.npz")
    Xv, yv = doc_features(val, args.with_encoder), val["doc_tier"]
    Xt, yt = doc_features(test, args.with_encoder), test["doc_tier"]
    print(f"Documents: validation {len(yv)}, test {len(yt)} | features per document: {Xv.shape[1]}"
          f" ({'text + encoder' if args.with_encoder else 'text only, same for every encoder'})\n")

    print("Per-tier medians on validation (what separates the tiers):")
    print(f"  {'tier':8s}{'boundaries':>12s}{'mean overlap':>14s}{'share > 0.075':>15s}{'mean style L2':>15s}")
    for i, t in enumerate(TIERS):
        m = yv == i
        print(f"  {t:8s}{np.median(Xv[m, 0]):12.1f}{np.median(Xv[m, 1]):14.4f}"
              f"{np.median(Xv[m, 6]):15.3f}{np.median(Xv[m, 20]):15.2f}")

    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            random_state=args.seed),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    print("\nChoosing the classifier on validation only (5-fold CV accuracy):")
    best_name, best_acc = None, -1.0
    for name, model in models.items():
        acc = accuracy_score(yv, cross_val_predict(model, Xv, yv, cv=cv))
        print(f"  {name:20s} {acc:.3f}")
        if acc > best_acc:
            best_name, best_acc = name, acc

    # Out-of-fold tier probabilities for the validation documents, so a head can be trained on
    # predicted (noisy) tiers instead of true ones (gate_lab.py --train-on-predicted-tier).
    val_probs = cross_val_predict(models[best_name], Xv, yv, cv=cv, method="predict_proba")
    model = models[best_name].fit(Xv, yv)
    assert list(model.classes_) == [0, 1, 2]
    probs = model.predict_proba(Xt)
    pred = probs.argmax(1)
    acc = accuracy_score(yt, pred)
    print(f"\nTEST (predicted once with '{best_name}'): accuracy {acc:.3f}")
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
    print("  confusion matrix (rows = true tier, columns = predicted):")
    print(f"  {'':8s}" + "".join(f"{t:>8s}" for t in TIERS) + f"{'recall':>9s}")
    for i, t in enumerate(TIERS):
        print(f"  {t:8s}" + "".join(f"{c:8d}" for c in cm[i]) + f"{cm[i, i] / cm[i].sum():9.3f}")

    out = cache / f"{args.name}_test_tierpred.npz"
    np.savez_compressed(out, probs=probs.astype(np.float32), pred=pred.astype(np.int64),
                        doc_ids=test["doc_ids"], classifier=np.array(best_name),
                        val_probs=val_probs.astype(np.float32), val_doc_ids=val["doc_ids"])
    print(f"\nSaved {out}")
    print(f"Next: python -u scripts/gate_lab.py --name {args.name} --final concat_tier --test-tier-file {out}")


if __name__ == "__main__":
    main()
