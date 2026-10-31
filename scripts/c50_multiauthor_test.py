#!/usr/bin/env python3
"""
Does DeSAF find author changes outside PAN?

Builds multi-author documents from Reuters C50 (50 real journalists, naturally written
news) in three ways, then runs a trained DeSAF checkpoint on them WITHOUT retraining:

  random          each segment comes from a random article of a different author
                  (topics vary, like an easy document)
  same_topic      every segment comes from articles on the same topic as the first one
                  (TF-IDF article similarity), like a single-topic hard document
  similar_splice  same topic, AND each new author's first sentence is chosen to be the most
                  similar to the previous sentence (similarity-filtered splice points,
                  in the spirit of PAN's construction). Similarity uses CONTENT words only
                  (TF-IDF without stop words), so function-word and punctuation distances are
                  held-out features the filter never sees.
  splice_ge_T     optional sweep (--splice-thresholds): like similar_splice, but the first
                  sentence is a random same-topic sentence with content similarity >= T

Only the construction changes; the authors and the text are the same.
Segments are runs of consecutive sentences from one article, so every
same-author boundary is natural writing.

For each construction it reports the checkpoint's F1 at its own threshold, the best
possible F1 (optimistic, threshold chosen on the same data), ROC-AUC (threshold-free),
and which direction each stylometric distance points (AUC > 0.5 = more dissimilar at
an author change = classical direction).

Usage (documents are built once and reused, so every checkpoint sees the same documents):
    python scripts/c50_multiauthor_test.py --c50-dir "C:\\Users\\Admin\\Downloads\\reuter+50+50" \
        --checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt --threshold 0.50 \
        --attention-fusion --bilstm --no-char-ngrams --contrastive --name seed43_gate
"""
import argparse
import html
import json
import re
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ["random", "same_topic", "similar_splice"]
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sen", "rep", "gov", "gen", "col", "lt", "sgt", "st",
          "inc", "corp", "co", "ltd", "plc", "bros", "no", "vs", "jan", "feb", "mar", "apr", "jun",
          "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "u.n", "e.u", "jr", "sr",
          "a.m", "p.m"}
SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|(?<=[.!?][\"'])\s+(?=[A-Z0-9\"'(])")

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
    p.add_argument("--c50-dir", required=True, help="folder containing C50train/ and C50test/")
    p.add_argument("--docs-dir", default="data/c50_multiauthor")
    p.add_argument("--n-docs", type=int, default=900, help="documents per construction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true", help="rebuild the documents even if saved")
    p.add_argument("--splice-thresholds", default="",
                   help="optional sweep, e.g. 0.05,0.15,0.25: extra constructions where each new author's "
                        "first sentence is a random same-topic sentence with similarity >= the threshold")
    p.add_argument("--build-only", action="store_true", help="only build and describe the documents")
    p.add_argument("--checkpoint")
    p.add_argument("--threshold", type=float)
    p.add_argument("--name", default="checkpoint")
    p.add_argument("--out-dir", default="results/c50_multiauthor")
    p.add_argument("--attention-fusion", action="store_true")
    p.add_argument("--film-gate", action="store_true")
    p.add_argument("--bilstm", action="store_true")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--no-char-ngrams", action="store_true")
    p.add_argument("--contrastive", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Building the documents
# --------------------------------------------------------------------------- #

def split_sentences(text):
    out = []
    for para in html.unescape(text).splitlines():
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        pieces = SPLIT_RE.split(para)
        merged = []
        for piece in pieces:
            if merged:
                last_word = merged[-1].rsplit(" ", 1)[-1].lower().rstrip(".")
                if last_word in ABBREV or re.fullmatch(r"[a-z]", last_word):
                    merged[-1] = merged[-1] + " " + piece
                    continue
            merged.append(piece)
        out.extend(s for s in merged if len(s.split()) >= 3)
    return out


def load_c50(c50_dir):
    articles = []  # (author, sentences, full_text)
    for split in ["C50train", "C50test"]:
        split_dir = Path(c50_dir) / split
        if not split_dir.exists():
            continue
        for author_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(author_dir.glob("*.txt")):
                text = f.read_text(encoding="utf-8", errors="replace")
                sents = split_sentences(text)
                if len(sents) >= 6:
                    articles.append((author_dir.name, sents, text))
    if not articles:
        raise SystemExit(f"No C50 articles found under {c50_dir} (expected C50train/ and C50test/).")
    return articles


def build_documents(articles, regime, n_docs, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.RandomState(seed)
    tau = float(regime[len("splice_ge_"):]) if regime.startswith("splice_ge_") else None
    authors = np.array([a for a, _, _ in articles])
    art_vec = TfidfVectorizer(max_features=50000, stop_words="english").fit_transform(
        [t for _, _, t in articles])
    sent_vectorizer = None
    if regime == "similar_splice" or tau is not None:
        # Content words only: the filter never sees function words or punctuation, so
        # fw_dist and punct_dist are held-out features (a reversal there is not mechanical).
        sent_vectorizer = TfidfVectorizer(max_features=50000, stop_words="english").fit(
            [s for _, sents, _ in articles for s in sents])
    n_splices, n_fallbacks = 0, 0

    def segment(art_idx, length, start=None):
        sents = articles[art_idx][1]
        length = min(length, len(sents))
        if start is None:
            start = rng.randint(0, len(sents) - length + 1)
        start = min(start, len(sents) - length)
        return sents[start:start + length]

    docs = []
    for d in range(n_docs):
        n_seg = rng.randint(2, 5)
        lengths = [rng.randint(2, 7) for _ in range(n_seg)]
        first = rng.randint(len(articles))
        used = {authors[first]}
        seg_list = [(authors[first], segment(first, lengths[0]))]
        if regime != "random":
            sims = (art_vec @ art_vec[first].T).toarray().ravel()
        for k in range(1, n_seg):
            allowed = np.array([a not in used for a in authors])
            if regime == "random":
                cands = np.where(allowed)[0]
                pick = cands[rng.randint(len(cands))]
                seg = segment(pick, lengths[k])
            else:
                cands = np.where(allowed)[0]
                top = cands[np.argsort(-sims[cands])[:5 if regime == "same_topic" else 20]]
                if regime == "same_topic":
                    pick = top[rng.randint(len(top))]
                    seg = segment(pick, lengths[k])
                else:
                    # candidate first sentences for the new author: every valid start position in
                    # the 20 most on-topic articles, scored by similarity to the previous sentence
                    prev = sent_vectorizer.transform([seg_list[-1][1][-1]])
                    options = []
                    for art in top:
                        sents = articles[art][1]
                        n_start = len(sents) - min(lengths[k], len(sents)) + 1
                        s = (sent_vectorizer.transform(sents[:n_start]) @ prev.T).toarray().ravel()
                        options.extend((float(v), int(art), j) for j, v in enumerate(s))
                    best = max(options, key=lambda o: o[0])
                    n_splices += 1
                    if tau is None:
                        chosen = best
                    else:
                        ok = [o for o in options if o[0] >= tau]
                        if ok:
                            chosen = ok[rng.randint(len(ok))]
                        else:
                            chosen, n_fallbacks = best, n_fallbacks + 1
                    pick = chosen[1]
                    seg = segment(pick, lengths[k], start=chosen[2])
            used.add(authors[pick])
            seg_list.append((authors[pick], seg))
        sentences, sent_authors = [], []
        for author, seg in seg_list:
            sentences.extend(seg)
            sent_authors.extend([str(author)] * len(seg))
        labels = [int(sent_authors[i] != sent_authors[i + 1]) for i in range(len(sentences) - 1)]
        docs.append({"doc_id": f"c50-{regime}-{d}", "sentences": sentences,
                     "labels": labels, "authors": sent_authors})
    if tau is not None and n_splices:
        print(f"    ({regime}: {n_fallbacks / n_splices:.1%} of splices had no candidate >= {tau} "
              "and used the most similar sentence instead)")
    return docs


def regimes_for(args):
    extra = [f"splice_ge_{float(t):.2f}" for t in args.splice_thresholds.split(",") if t.strip()]
    return REGIMES + extra


def load_or_build(args, articles_cache):
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for regime in regimes_for(args):
        path = docs_dir / f"{regime}.jsonl"
        if path.exists() and not args.rebuild:
            out[regime] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            print(f"  {regime:15s} loaded {len(out[regime])} documents from {path}")
            continue
        if articles_cache.get("articles") is None:
            t0 = time.time()
            articles_cache["articles"] = load_c50(args.c50_dir)
            print(f"  loaded {len(articles_cache['articles'])} C50 articles in {time.time() - t0:.0f}s")
        t0 = time.time()
        regime_seed = args.seed + zlib.crc32(regime.encode()) % 10000
        out[regime] = build_documents(articles_cache["articles"], regime, args.n_docs, regime_seed)
        with path.open("w", encoding="utf-8") as fh:
            for doc in out[regime]:
                fh.write(json.dumps(doc) + "\n")
        print(f"  {regime:15s} built {len(out[regime])} documents in {time.time() - t0:.0f}s -> {path}")
    return out


# --------------------------------------------------------------------------- #
#  Measurements
# --------------------------------------------------------------------------- #

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


def f1_prf(preds, labels):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def auc_cell(values, labels):
    from scipy import stats
    change, same = values[labels == 1], values[labels == 0]
    res = stats.mannwhitneyu(change, same, alternative="two-sided")
    auc = res.statistic / (len(change) * len(same))
    stars = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else ""
    return f"{auc:.3f}{stars:3s} {'N' if auc > 0.5 else 'R'}"


def describe(docs_by_regime):
    """Model-free checks: does each construction do what it should, and which way do the
    surface distances point?"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\nDocuments per construction:")
    print(f"  {'construction':15s}{'docs':>6s}{'sentences/doc':>15s}{'boundaries':>12s}{'change rate':>13s}"
          f"{'all-change F1':>15s}")
    for regime, docs in docs_by_regime.items():
        labels = np.concatenate([d["labels"] for d in docs])
        rate = labels.mean()
        print(f"  {regime:15s}{len(docs):6d}{np.mean([len(d['sentences']) for d in docs]):15.1f}"
              f"{len(labels):12d}{rate:13.3f}{2 * rate / (1 + rate):15.3f}")

    vec = TfidfVectorizer(max_features=20000).fit(
        [s for docs in docs_by_regime.values() for d in docs for s in d["sentences"]])
    print("\nTopic overlap of the two sentences (TF-IDF cosine), mean, and surface-distance direction")
    print("(AUC > 0.5 = more dissimilar at an author change = N, the classical direction):")
    print(f"  {'construction':15s}{'overlap: change':>16s}{'no change':>11s}"
          + "".join(f"{n:>16s}" for n in ["fw_dist", "trigram_dissim", "punct_dist"]))
    for regime, docs in docs_by_regime.items():
        a = [d["sentences"][i] for d in docs for i in range(len(d["labels"]))]
        b = [d["sentences"][i + 1] for d in docs for i in range(len(d["labels"]))]
        y = np.concatenate([d["labels"] for d in docs])
        ovl = np.asarray(vec.transform(a).multiply(vec.transform(b)).sum(axis=1)).ravel()
        surf = np.array([surface_distances(x, z) for x, z in zip(a, b)], dtype=np.float32)
        cells = [auc_cell(surf[:, j], y) for j in range(3)]
        print(f"  {regime:15s}{ovl[y == 1].mean():16.4f}{ovl[y == 0].mean():11.4f}"
              + "".join(f"{c:>16s}" for c in cells))


# --------------------------------------------------------------------------- #
#  Running the checkpoint
# --------------------------------------------------------------------------- #

def run_checkpoint(args, docs_by_regime):
    import torch
    from torch.amp import autocast
    sys.path.insert(0, str(ROOT))
    from src.features.stylometric import StylometricExtractor
    from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset
    from train_deberta_fusion import encode_document, build_style_diffs
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StylometricExtractor(use_char_ngrams=not args.no_char_ngrams)
    style_dim = extractor.feature_dim
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)
    model = DeBERTaFusionE2E(
        style_dim=style_dim, unfreeze_layers=args.unfreeze_layers, use_style=True,
        use_attention_fusion=args.attention_fusion, use_film=args.film_gate, use_bilstm=True,
        bilstm_hidden=args.bilstm_hidden, dropout=args.dropout, use_adversarial=False,
        use_contrastive=args.contrastive,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    results, direction_rows = {}, []
    for regime, docs in docs_by_regime.items():
        t0 = time.time()
        objs = [SimpleNamespace(sentences=d["sentences"], boundary_labels=d["labels"], doc_id=d["doc_id"])
                for d in docs]
        style_diffs = build_style_diffs(objs, extractor)
        ds = DocLevelDataset([o.sentences for o in objs], [o.boundary_labels for o in objs],
                             style_diffs, doc_ids=[o.doc_id for o in objs])
        probs, labels, surf, style_l2, lengths = [], [], [], [], []
        for i in range(len(ds)):
            doc = ds[i]
            with torch.no_grad():
                fused = encode_document(model, doc, tokenizer, device, True, args.max_length, args.sub_batch)
                with autocast("cuda", dtype=torch.float16):
                    logits = model.forward_document(fused.unsqueeze(0))
            probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1).astype(np.float32))
            labels.append(np.asarray(doc["labels"], dtype=np.int64))
            lengths.append(len(doc["labels"]))
            surf.extend(surface_distances(a, b) for a, b in doc["pairs"])
            style_l2.append(np.linalg.norm(np.asarray(doc["style_diffs"], dtype=np.float32), axis=1))
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        surf, style_l2 = np.array(surf, dtype=np.float32), np.concatenate(style_l2)

        f1, prec, rec = f1_prf((probs.astype(np.float16) >= args.threshold).astype(np.int64), labels)
        best = max((f1_prf((probs >= t).astype(np.int64), labels)[0], float(t))
                   for t in np.round(np.arange(0.02, 0.99, 0.02), 2))
        auc = float(roc_auc_score(labels, probs))
        results[regime] = {"f1": f1, "precision": prec, "recall": rec, "best_f1": best[0],
                           "best_threshold": best[1], "roc_auc": auc, "change_rate": float(labels.mean()),
                           "boundaries": int(len(labels))}
        direction_rows.append((regime, [auc_cell(x, labels) for x in
                                        (surf[:, 0], surf[:, 1], surf[:, 2], style_l2)]))
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{args.name}_{regime}.npz", probs=probs, labels=labels,
                            lengths=np.array(lengths), surf=surf, style_l2=style_l2)
        print(f"  {regime:15s} done in {time.time() - t0:.0f}s")

    print("\n" + "=" * 96)
    print(f"DeSAF ({args.name}) on C50 multi-author documents, no retraining")
    print("=" * 96)
    print(f"  {'construction':15s}{'F1 @ ' + format(args.threshold, '.2f'):>11s}{'precision':>11s}{'recall':>9s}"
          f"{'best F1 (thr)':>17s}{'ROC-AUC':>10s}{'all-change F1':>15s}")
    for regime, r in results.items():
        rate = r["change_rate"]
        print(f"  {regime:15s}{r['f1']:11.4f}{r['precision']:11.4f}{r['recall']:9.4f}"
              f"{r['best_f1']:11.4f} ({r['best_threshold']:.2f}){r['roc_auc']:10.4f}{2 * rate / (1 + rate):15.4f}")
    print("  (best F1 picks the threshold on the same documents, so it is optimistic; ROC-AUC needs no threshold)")

    print("\nDirection of each stylometric distance (AUC > 0.5 = more dissimilar at an author change = N):")
    print(f"  {'construction':15s}" + "".join(f"{n:>16s}" for n in
                                             ["fw_dist", "trigram_dissim", "punct_dist", "style_L2 (193)"]))
    for regime, cells in direction_rows:
        print(f"  {regime:15s}" + "".join(f"{c:>16s}" for c in cells))

    summary = Path(args.out_dir) / f"{args.name}_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nSaved per-boundary results and {summary}")


def main():
    args = parse_args()
    print("Multi-author documents from C50:")
    docs = load_or_build(args, {})
    describe(docs)
    if args.build_only:
        return
    if not args.checkpoint or args.threshold is None:
        raise SystemExit("Pass --checkpoint and --threshold to evaluate (or --build-only).")
    if not args.bilstm:
        raise SystemExit("Only BiLSTM checkpoints are supported (pass --bilstm).")
    run_checkpoint(args, docs)


if __name__ == "__main__":
    main()
