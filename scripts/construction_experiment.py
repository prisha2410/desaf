#!/usr/bin/env python3
"""
Same text, different assembly: does the way a multi-author document is put together decide which way the
stylometric cues point at author changes?

Source text (only single-author runs of consecutive sentences are used, so every same-author boundary is
natural writing):
  --source pan   single-author segments cut out of PAN documents at their author changes (e.g. PAN 2025
                 Reddit, train split). Two segments from the same source document are never combined (the
                 same author can appear twice in one source document).
  --source c50   Reuters C50 news articles (needs --c50-dir; uses scripts/c50_multiauthor_test.py's loader).

Constructions (900 documents each, 2-4 authors, 2-6 consecutive sentences per author):
  random          each next segment from a random source (topics vary)
  same_topic      next segment from the 5 sources most similar in content to the first one
  content_splice  among the 20 most on-topic sources, start the next author at the sentence with the highest
                  content-word similarity to the previous sentence (stop words removed; as in the C50 run)
  length_splice   among 20 random sources, start at the sentence closest in length to the previous one
  style_splice    among 20 random sources, start at the sentence with the most similar z-scored style
                  vector (surface, readability, punctuation and function-word features; no POS)
  style_nolen     same, but the style vector leaves out length and readability features (length is held out)
  topic_style     among the 20 most on-topic sources, start at the sentence best on content AND style
                  similarity together (sum of the two ranks), the recipe both PAN overviews describe in words
  topic_length    [added 2026-11-16] as topic_style, but the style rank is replaced by length similarity
                  (-|log((words_new + 1) / (words_prev + 1))|)
  topic_rawstyle  [added 2026-11-16] as topic_style, but the style rank uses the UNSCALED raw feature vector
                  (minus the L2 distance of the raw 182-feature vectors, which sentence length dominates), i.e. a
                  filter on a stylistic vector that nobody standardised
Only the splice choice changes; within-author text is untouched.
The two added constructions test whether a content + length-dominated filter reproduces PAN 2025 Hard's length-led
PROFILE, not just its signs. Pre-registered criteria (critic, 2026-11-09; checked by construction_profile_check.py):
C1 content <= 0.35; C2 |log length ratio| <= 0.35; C3 fw unmatched <= 0.45 and >= 0.47 at exact length; C4 punct
unmatched <= 0.47 and >= 0.50 at exact length; C5 trigram <= 0.46 at length x topic; C6 lag-1 content <= 0.40 and lags
2-3 within 0.05 of chance; C7 change-rate ratio (pairs with a sentence < 8 words / both >= 8) <= 0.5. "Reproduces PAN
Hard's profile" only if one construction meets C1-C5. Run once; report both whatever the outcome.

Measured at every boundary (AUC = P(change more dissimilar than no-change); > 0.5 normal, < 0.5 reversed;
95% CIs resample documents): content overlap, length difference, fw_dist / trigram / punct (same code as the
PAN analysis), and z-scored feature families (same grouping as pan_direction_followup.py, without POS).
Pre-registered expectation: random = normal everywhere; each filter reverses the cue it selects on; cues it
does not select on stay normal or go to chance.

Run from D:\\sass-tad (model-free, CPU, no spaCy):
    python scripts/construction_experiment.py --source pan --data-root data/raw/pan2025_full --name reddit2025
    python scripts/construction_experiment.py --source c50 --c50-dir "C:\\Users\\Admin\\Downloads\\reuter+50+50" --name c50
"""
import argparse
import json
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from pan_direction_followup import load_split, auc_fast, doc_index, boot_ci, tag, ci_txt, clean  # noqa: E402
from pan_feature_direction import surface_distances  # noqa: E402

CONSTRUCTIONS = ["random", "same_topic", "content_splice", "length_splice", "style_splice", "style_nolen",
                 "topic_style", "topic_length", "topic_rawstyle"]
TOPIC_SPLICES = ("content_splice", "topic_style", "topic_length", "topic_rawstyle")
LENGTH_FEATS = {"char_count", "word_count", "avg_sentence_length", "punct_period"}
READ_FEATS = {"flesch_reading_ease", "gunning_fog", "coleman_liau", "avg_syllables_per_word", "complex_word_ratio"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pan", "c50"], required=True)
    p.add_argument("--data-root", help="PAN layout root (for --source pan)")
    p.add_argument("--source-split", default="train")
    p.add_argument("--source-tiers", default="easy,medium,hard")
    p.add_argument("--max-source-docs", type=int, default=0, help="per tier; 0 = all")
    p.add_argument("--c50-dir", help="for --source c50")
    p.add_argument("--name", required=True)
    p.add_argument("--constructions", default=",".join(CONSTRUCTIONS))
    p.add_argument("--n-docs", type=int, default=900)
    p.add_argument("--keep-repeated", action="store_true",
                   help="keep source segments containing a line repeated >= 10 times (dropped by default)")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/construction")
    return p.parse_args()


def group_of(name):
    if name in LENGTH_FEATS:
        return "length"
    if name in READ_FEATS:
        return "readability"
    if name.startswith("punct_"):
        return "punctuation"
    if name.startswith("fw_"):
        return "function_words"
    return "other_surface"


GROUPS = ["length", "readability", "punctuation", "function_words", "other_surface"]


def load_units(args, rng):
    """Returns list of units: (author_key, [global sentence ids]) and the sentence list."""
    sentences, units = [], []
    if args.source == "pan":
        docs = []
        for tier in [t.strip() for t in args.source_tiers.split(",") if t.strip()]:
            d = load_split(args.data_root, tier, args.source_split)
            if args.max_source_docs and len(d) > args.max_source_docs:
                d = [d[i] for i in sorted(rng.choice(len(d), args.max_source_docs, replace=False))]
            docs += [(f"{tier}-{i}", s, c) for i, (s, c) in enumerate(d)]
        freq = Counter(s for _, sents, _ in docs for s in sents)
        dropped = 0
        for key, sents, ch in docs:
            start = 0
            for i in range(len(sents)):
                if i == len(sents) - 1 or ch[i] == 1:
                    seg = sents[start:i + 1]
                    start = i + 1
                    if len(seg) < 2:
                        continue
                    if not args.keep_repeated and any(freq[s] >= 10 for s in seg):
                        dropped += 1
                        continue
                    ids = list(range(len(sentences), len(sentences) + len(seg)))
                    sentences += seg
                    units.append((key, ids))
        print(f"Source: {len(docs)} PAN documents ({args.source_split}) -> {len(units)} single-author segments "
              f"of >= 2 sentences ({dropped} dropped for containing a line repeated >= 10 times)")
    else:
        from c50_multiauthor_test import load_c50
        arts = load_c50(args.c50_dir)
        for author, sents, _ in arts:
            if len(sents) < 2:
                continue
            ids = list(range(len(sentences), len(sentences) + len(sents)))
            sentences += sents
            units.append((str(author), ids))
        print(f"Source: {len(units)} C50 articles")
    return units, sentences


def build(construction, units, unit_vec, sent_vec, words, style_all, style_nolen, n_docs, seed, guard, raw=None):
    """guard = (text_of_sentence_id, natural_pairs): a candidate segment is skipped if it repeats any sentence
    text already in the document, and a splice is skipped if (previous sentence, new first sentence) occurs as a
    natural same-author adjacent pair anywhere in the source (the same text can sit in several source documents)."""
    rng = np.random.RandomState(seed)
    keys = np.array([k for k, _ in units])
    n_units = len(units)
    text, natural = guard
    unit_texts = [frozenset(text[i] for i in ids) for _, ids in units]
    stats = {"splices": 0, "skipped_options": 0}

    def clash(u, doc_texts):
        return not unit_texts[u].isdisjoint(doc_texts)

    def window(u, length, start=None):
        ids = units[u][1]
        length = min(length, len(ids))
        if start is None:
            start = rng.randint(0, len(ids) - length + 1)
        return ids[start:start + length]

    def options(cands, length):
        out = []
        for u in cands:
            ids = units[u][1]
            n_start = len(ids) - min(length, len(ids)) + 1
            out += [(u, j, ids[j]) for j in range(n_start)]
        return out

    docs = []
    for d in range(n_docs):
        n_seg = rng.randint(2, 5)
        lengths = [rng.randint(2, 7) for _ in range(n_seg)]
        first = rng.randint(n_units)
        used = {keys[first]}
        segs = [(keys[first], window(first, lengths[0]))]
        doc_texts = set(unit_texts[first])
        sims = None
        if construction == "same_topic" or construction in TOPIC_SPLICES:
            sims = (unit_vec @ unit_vec[first].T).toarray().ravel()
        for k in range(1, n_seg):
            allowed = np.flatnonzero(~np.isin(keys, list(used)))
            prev = segs[-1][1][-1]
            if construction == "random":
                for _ in range(50):
                    u = allowed[rng.randint(len(allowed))]
                    if not clash(u, doc_texts):
                        break
                seg = window(u, lengths[k])
            elif construction == "same_topic":
                order = allowed[np.argsort(-sims[allowed])]
                top = [u for u in order[:200] if not clash(u, doc_texts)][:5] or list(order[:5])
                u = top[rng.randint(len(top))]
                seg = window(u, lengths[k])
            else:
                if construction in TOPIC_SPLICES:
                    order = allowed[np.argsort(-sims[allowed])]
                    cands = [u for u in order[:200] if not clash(u, doc_texts)][:20] or list(order[:20])
                else:
                    pool = allowed[rng.choice(len(allowed), min(200, len(allowed)), replace=False)]
                    cands = [u for u in pool if not clash(u, doc_texts)][:20] or list(pool[:20])
                opts = options(cands, lengths[k])
                ok = [o for o in opts if (text[prev], text[o[2]]) not in natural]
                stats["splices"] += 1
                stats["skipped_options"] += len(opts) - len(ok)
                opts = ok or opts
                first_ids = np.array([o[2] for o in opts])
                if construction == "content_splice":
                    score = (sent_vec[first_ids] @ sent_vec[prev].T).toarray().ravel()
                elif construction == "length_splice":
                    score = -np.abs(np.log((words[first_ids] + 1) / (words[prev] + 1)))
                elif construction == "style_splice":
                    score = style_all[first_ids] @ style_all[prev]
                elif construction == "style_nolen":
                    score = style_nolen[first_ids] @ style_nolen[prev]
                elif construction in ("topic_length", "topic_rawstyle"):
                    c = (sent_vec[first_ids] @ sent_vec[prev].T).toarray().ravel()
                    if construction == "topic_length":
                        s = -np.abs(np.log((words[first_ids] + 1) / (words[prev] + 1)))
                    else:
                        s = -np.linalg.norm(raw[first_ids] - raw[prev], axis=1)
                    score = np.argsort(np.argsort(c)) + np.argsort(np.argsort(s))
                else:  # topic_style: sum of ranks of content and style similarity
                    c = (sent_vec[first_ids] @ sent_vec[prev].T).toarray().ravel()
                    s = style_all[first_ids] @ style_all[prev]
                    score = np.argsort(np.argsort(c)) + np.argsort(np.argsort(s))
                best = np.flatnonzero(score == score.max())
                u, j, _ = opts[best[rng.randint(len(best))]]
                seg = window(u, lengths[k], start=j)
            used.add(keys[u])
            doc_texts |= unit_texts[u]
            segs.append((keys[u], seg))
        ids, auth = [], []
        for key, seg in segs:
            ids += seg
            auth += [key] * len(seg)
        labels = [int(auth[i] != auth[i + 1]) for i in range(len(ids) - 1)]
        docs.append({"ids": ids, "labels": labels})
    return docs, stats


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units, sentences = load_units(args, rng)
    words = np.array([len(s.split()) for s in sentences], dtype=float)

    from sklearn.feature_extraction.text import TfidfVectorizer
    sent_vectorizer = TfidfVectorizer(max_features=50000, stop_words="english").fit(sentences)
    sent_vec = sent_vectorizer.transform(sentences).tocsr()
    unit_vec = TfidfVectorizer(max_features=50000, stop_words="english").fit_transform(
        [" ".join(sentences[i] for i in ids) for _, ids in units]).tocsr()
    print(f"{len(sentences)} sentences; content TF-IDF ready ({time.time() - t0:.0f}s)")

    from src.features.stylometric import StylometricExtractor
    ex = StylometricExtractor(use_pos=False, use_char_ngrams=False)
    names = ex.feature_names
    V = np.empty((len(sentences), len(names)), dtype=np.float32)
    for i, s in enumerate(sentences):
        V[i] = ex.extract_vector(s)
        if (i + 1) % 50000 == 0:
            print(f"  style vectors {i + 1}/{len(sentences)} ({time.time() - t0:.0f}s)")
    sd = V.std(0)
    keep = (sd > 1e-9) & ((np.abs(V) > 1e-12).mean(0) >= 0.01)
    sd[~keep] = 1.0
    Z = np.clip((V - V.mean(0)) / sd, -5, 5)
    Z[:, ~keep] = 0.0
    grp = np.array([group_of(n) for n in names])
    nolen = keep & ~np.isin(grp, ["length", "readability"])

    def unit_norm(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, 1e-9)

    style_all = unit_norm(Z * keep)
    style_nolen = unit_norm(Z * nolen)
    print(f"Style vectors: {int(keep.sum())} of {len(names)} features kept "
          + ", ".join(f"{g} {int(((grp == g) & keep).sum())}" for g in GROUPS) + f" ({time.time() - t0:.0f}s)")

    cols = ["content overlap", "|log length ratio|", "|d words|", "fw_dist", "trigram_dissim", "punct_dist",
            "all_z"] + GROUPS + ["all_z_no_len_read"]
    summary = {"name": args.name, "source": args.source, "constructions": {}}
    print("\n" + "=" * 120)
    print(f"{args.name}: cue direction at author changes under each construction (AUC; N > 0.5 normal, R < 0.5 reversed)")
    print("content overlap AUC = P(change shares LESS content than no-change); the others = P(change more dissimilar)")
    print("=" * 120)
    natural = {(sentences[ids[i]], sentences[ids[i + 1]]) for _, ids in units for i in range(len(ids) - 1)}
    guard = (sentences, natural)
    for construction in [c.strip() for c in args.constructions.split(",") if c.strip()]:
        seed = args.seed + zlib.crc32(construction.encode()) % 10000
        docs, st = build(construction, units, unit_vec, sent_vec, words, style_all, style_nolen, args.n_docs, seed,
                         guard, raw=V)
        # leak check: change boundaries whose two sentences are a natural same-author pair somewhere in the source
        leak = np.mean([(sentences[d["ids"][i]], sentences[d["ids"][i + 1]]) in natural
                        for d in docs for i in range(len(d["labels"])) if d["labels"][i] == 1])
        with open(out_dir / f"{args.name}_{construction}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps({"sentences": [sentences[i] for i in d["ids"]], "labels": d["labels"]}) + "\n")
        a = np.array([d["ids"][i] for d in docs for i in range(len(d["labels"]))])
        b = np.array([d["ids"][i + 1] for d in docs for i in range(len(d["labels"]))])
        y = np.concatenate([d["labels"] for d in docs]).astype(int)
        doc = np.repeat(np.arange(len(docs)), [len(d["labels"]) for d in docs])
        ovl = np.asarray(sent_vec[a].multiply(sent_vec[b]).sum(axis=1)).ravel()
        lr = np.abs(np.log((words[a] + 1) / (words[b] + 1)))
        dw = np.abs(words[a] - words[b])
        surf = np.array([surface_distances(sentences[i], sentences[j]) for i, j in zip(a, b)])
        D = np.abs(Z[a] - Z[b])
        fam = [np.linalg.norm(D[:, keep], axis=1)]
        fam += [np.linalg.norm(D[:, (grp == g) & keep], axis=1) for g in GROUPS]
        fam += [np.linalg.norm(D[:, nolen], axis=1)]
        M = np.column_stack([-ovl, lr, dw, surf, np.column_stack(fam)])
        aucs = [auc_fast(M[:, j], y) for j in range(M.shape[1])]
        lo, hi = boot_ci(lambda i, cid: np.array([auc_fast(M[i, j], y[i]) for j in range(M.shape[1])]),
                         doc_index(doc), args.boot, rng)
        ratio = ovl[y == 1].mean() / max(ovl[y == 0].mean(), 1e-12)
        print(f"\n{construction.upper()}  docs {len(docs)}, boundaries {len(y)}, change rate {y.mean():.3f}; "
              f"content overlap change/within {ovl[y == 1].mean():.4f}/{ovl[y == 0].mean():.4f} (ratio {ratio:.2f}); "
              f"words change/within |log ratio| {lr[y == 1].mean():.3f}/{lr[y == 0].mean():.3f}")
        print(f"  guard: {leak:.2%} of change boundaries are a natural same-author pair in the source; "
              f"{st['skipped_options']} candidate splice points skipped for that reason over {st['splices']} splices")
        for start in range(0, len(cols), 7):
            sl = slice(start, start + 7)
            print("  " + "".join(f"{c[:19]:>21s}" for c in cols[sl]))
            print("  " + "".join(f"{tag(x):>21s}" for x in aucs[sl]))
            print("  " + "".join(f"{ci_txt(l, h):>21s}" for l, h in zip(lo[sl], hi[sl])))
        summary["constructions"][construction] = {
            "n_docs": len(docs), "n_boundaries": int(len(y)), "change_rate": float(y.mean()),
            "overlap_change": float(ovl[y == 1].mean()), "overlap_within": float(ovl[y == 0].mean()),
            "overlap_ratio": float(ratio),
            "auc": {c: {"auc": x, "ci": [l, h]} for c, x, l, h in zip(cols, aucs, lo, hi)}}
        (out_dir / f"{args.name}_summary.json").write_text(json.dumps(clean(summary), indent=1), encoding="utf-8")
    print(f"\nSaved {out_dir / (args.name + '_summary.json')} and the built documents ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
