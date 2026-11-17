#!/usr/bin/env python3
"""
A learned tier router: can a document classifier tell PAN 2025's tiers apart well enough to replace the oracle flag?

DeBERTa-v3-small sequence classifier (3 classes) on the document text, head + tail tokens (max 512), trained on the
PAN 2025 TRAIN documents of all tiers (tier = folder; no author-change labels used), epoch chosen on a held-out slice of
TRAIN (see below), VALIDATION and TEST predicted once. Training data stays PAN 2025 only.

Output: a tier file in the format gate_lab.py --test-tier-file reads (probs, pred, doc_ids, val_probs, val_doc_ids),
ordered like the stage-2 caches. The epoch is chosen on a stratified 10% slice held out from TRAIN, so the validation
predictions are fully out of sample and --train-on-predicted-tier can use them as they are.

Why: under random errors a tier guess must be right for about 81-85% of documents to match the tier-free DeSAF, and
about 92-95% to keep half of the oracle gain ((T) in REAL_VERIFIED_NUMBERS.md). The best earlier classifiers reached
0.762 (text features) and 0.853-0.863 (with encoder features). ULILLE's router reached 0.9867 on PAN 2026 validation.

Pre-registered (2026-11-16): prediction (completeness critic): test accuracy 0.88-0.95. Decision on the OFFICIAL
metric (pooled two-class macro-F1, tier average; computed by --score-only, because gate_lab prints positive-class F1):
if the 9 predicted-tier heads (hard flags) average >= 0.8678 (the original DeSAF) -> "with a learned router, the
tier-aware head is at least as good as tier-free DeSAF without any oracle"; if >= 0.8749 (half the oracle gain over the
original) -> "a learned router recovers about half the oracle gain or more"; otherwise "a learned router does not
recover the gain". Accuracy >= 0.95 alone is not a claim.

Run from D:\\sass-tad (GPU):
    python scripts/tier_router.py
then, for each encoder, into a SEPARATE output root (results/gate_lab/*/predicted_tier* hold the earlier runs behind
section 11 and must not be overwritten):
    python scripts/gate_lab.py --name seed43_gate --final concat_tier --test-tier-file checkpoints/stage2_cache/router_test_tierpred.npz --out-dir results/gate_lab_router
    (same for seed44_gate and seed45_gate)
then score on the official metric:
    python scripts/tier_router.py --score-only
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.utils.data_loader import load_pan2025_full  # noqa: E402

TIERS = ["easy", "medium", "hard"]
MODEL = "microsoft/deberta-v3-small"


def load(split, data_root):
    texts, labels, ids = [], [], []
    for k, t in enumerate(TIERS):
        for d in load_pan2025_full(data_root, tier=t, split=split):
            texts.append("\n".join(d.sentences)); labels.append(k); ids.append(d.doc_id)
    return texts, np.array(labels, dtype=np.int64), np.array(ids)


def encode(texts, tok, max_len):
    """Head + tail: the first and last (max_len - 2) / 2 tokens of each document."""
    out = []
    body = max_len - 2
    for txt in texts:
        ids = tok(txt, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > body:
            h = body // 2
            ids = ids[:h] + ids[-(body - h):]
        out.append(tok.build_inputs_with_special_tokens(ids))
    return out


def batches(enc, order, bs, pad_id, device):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        yield idx, ids.to(device), att.to(device)


@torch.no_grad()
def predict(model, enc, bs, pad_id, device):
    model.eval()
    order = np.argsort([len(e) for e in enc])                      # length-sorted for speed; results put back in place
    probs = np.zeros((len(enc), 3), np.float32)
    for idx, ids, att in batches(enc, order, bs, pad_id, device):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=att).logits
        probs[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs


def confusion(y, p):
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def show(name, y, probs):
    p = probs.argmax(1); cm = confusion(y, p)
    print(f"  {name}: accuracy {np.mean(p == y):.4f}; confusion (rows true, cols predicted; easy/medium/hard):")
    for i, t in enumerate(TIERS):
        print(f"    {t:7s} " + " ".join(f"{c:5d}" for c in cm[i]) + f"   recall {cm[i, i] / max(cm[i].sum(), 1):.3f}")
    return float(np.mean(p == y)), cm.tolist()


def macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


def score_only(a):
    """Official metric for the router-fed heads written by gate_lab.py --out-dir results/gate_lab_router."""
    root = Path(a.router_out_dir)
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    print(f"Official pooled two-class macro-F1 (tier average) of the router-fed heads under {root}")
    print(f"References: original DeSAF 0.8678; half the oracle gain 0.8749; oracle tier head 0.8820; "
          f"earlier hard-argmax predicted tiers 0.8532")
    for sub, pattern in (("predicted_tier", "final_concat_tier_s{s}_test_predtier.npz"),
                         ("predicted_tier_soft", "final_concat_tier_s{s}_test_predtier.npz"),
                         ("predicted_tier_trained", "final_concat_tier_s{s}_test.npz"),
                         ("predicted_tier_trained_soft", "final_concat_tier_s{s}_test.npz")):
        runs = []
        for n in names:
            for s in (1, 2, 3):
                f = root / n / sub / pattern.format(s=s)
                if f.exists():
                    z = np.load(f)
                    runs.append((n, s, [macro(z[f"{t}__labels"].astype(int), z[f"{t}__preds"].astype(int)) for t in TIERS]))
        if not runs:
            continue
        m = np.mean([r[2] for r in runs], axis=0)
        print(f"\n  {sub}: {len(runs)} heads" + ("" if len(runs) == 3 * len(names) else "  (WARNING: expected "
                                                                                   f"{3 * len(names)})"))
        for n, s, sc in runs:
            print(f"    {n} head {s}: " + " / ".join(f"{x:.4f}" for x in sc) + f" / avg {np.mean(sc):.4f}")
        avg = float(m.mean())
        verdict = ("recovers about half the oracle gain or more" if avg >= 0.8749 else
                   "at least as good as tier-free DeSAF" if avg >= 0.8678 else "does not recover the gain")
        print(f"    mean {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f} / avg {avg:.4f} -> {verdict}"
              + ("  (pre-registered decision)" if sub == "predicted_tier" else "  (secondary)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    ap.add_argument("--cache-name", default="seed43_gate", help="any stage-2 cache: only its document order is used")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--out", default="checkpoints/stage2_cache/router_test_tierpred.npz")
    ap.add_argument("--score-only", action="store_true", help="score the router-fed gate_lab heads, no training")
    ap.add_argument("--router-out-dir", default="results/gate_lab_router")
    ap.add_argument("--names", default="seed43_gate,seed44_gate,seed45_gate")
    a = ap.parse_args()
    if a.score_only:
        score_only(a); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = {s: load(s, a.data_root) for s in ("train", "validation", "test")}
    enc = {s: encode(data[s][0], tok, a.max_len) for s in data}
    cache = Path(a.cache_dir)
    caches = {}
    for split in ("test", "validation"):
        cz = np.load(cache / f"{a.cache_name}_{split}.npz")
        missing = set(str(x) for x in cz["doc_ids"]) - set(data[split][2])
        if missing:
            raise SystemExit(f"{len(missing)} {split} cache documents are not in the loaded {split} split, "
                             f"e.g. {sorted(missing)[0]}; check --cache-dir / --cache-name before training")
        caches[split] = cz
    # epoch selection on a stratified 10% slice of TRAIN (validation stays fully out of sample)
    rs = np.random.RandomState(a.seed)
    ytr = data["train"][1]
    dev_idx = np.concatenate([rs.permutation(np.flatnonzero(ytr == k))[:int(round(0.1 * np.sum(ytr == k)))] for k in range(3)])
    fit_idx = np.setdiff1d(np.arange(len(ytr)), dev_idx)
    for s in data:
        print(f"{s}: {len(data[s][1])} documents; tokens per document (after head+tail) median "
              f"{int(np.median([len(e) for e in enc[s]]))}, {np.mean([len(e) >= a.max_len for e in enc[s]]):.0%} cut")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(device)
    model.deberta.embeddings.word_embeddings.weight.requires_grad_(False)   # 4 GB GPU: keeps ~1.5 GB free
    if not a.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    n_micro = math.ceil(len(fit_idx) / a.batch)
    steps = math.ceil(n_micro / a.accum) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_train = torch.as_tensor(data["train"][1], dtype=torch.long)
    enc_dev = [enc["train"][i] for i in dev_idx]
    print(f"Training on {len(fit_idx)} train documents; epoch chosen on {len(dev_idx)} held-out train documents")
    best, best_acc, history = None, -1.0, []
    rng = np.random.RandomState(a.seed)
    for ep in range(a.epochs):
        model.train()
        order = fit_idx[rng.permutation(len(fit_idx))]
        tot, n = 0.0, 0
        opt.zero_grad()
        for step, (idx, ids, att) in enumerate(batches(enc["train"], order, a.batch, tok.pad_token_id, device)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y_train[idx].to(device)) / a.accum
            scaler.scale(loss).backward()
            tot += loss.item() * a.accum; n += 1
            if (step + 1) % a.accum == 0 or step + 1 == n_micro:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            if (step + 1) % 500 == 0:
                print(f"  epoch {ep + 1} step {step + 1}: loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dp = predict(model, enc_dev, 8, tok.pad_token_id, device)
        acc = float(np.mean(dp.argmax(1) == ytr[dev_idx]))
        history.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "heldout_train_acc": acc})
        print(f"epoch {ep + 1}: train loss {tot / max(n, 1):.4f}, held-out train accuracy {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, best = acc, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best)
    print(f"\nChosen: the epoch with the best held-out TRAIN accuracy ({best_acc:.4f}). Validation and test predicted once.")
    vp = predict(model, enc["validation"], 8, tok.pad_token_id, device)
    tp = predict(model, enc["test"], 8, tok.pad_token_id, device)
    raw = Path(a.out).with_name(Path(a.out).stem + "_raw.npz")
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw, test_probs=tp, test_ids=data["test"][2], val_probs=vp, val_ids=data["validation"][2])
    torch.save(best, raw.with_name("tier_router_best.pt"))
    val_acc, val_cm = show("VALIDATION", data["validation"][1], vp)
    test_acc, test_cm = show("TEST", data["test"][1], tp)

    # order the outputs like the stage-2 caches, so gate_lab can check the document order
    out = {}
    for split, probs, key in (("test", tp, ""), ("validation", vp, "val_")):
        cz = caches[split]
        cids = [str(x) for x in cz["doc_ids"]]
        pos = {d: i for i, d in enumerate(data[split][2])}
        missing = [d for d in cids if d not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} {split} cache documents have no router prediction, e.g. {missing[0]}")
        P = probs[[pos[d] for d in cids]]
        out[f"{key}probs"] = P.astype(np.float32); out[f"{key}doc_ids"] = cz["doc_ids"]
        if split == "test":
            out["pred"] = P.argmax(1).astype(np.int64)
            acc_cache = float(np.mean(P.argmax(1) == cz["doc_tier"].astype(int)))
            print(f"  test accuracy re-checked in cache order: {acc_cache:.4f}")
    out["classifier"] = np.array("deberta-v3-small router (head+tail, PAN 2025 train)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"val_acc": val_acc, "test_acc": test_acc, "heldout_train_acc_best": best_acc, "val_confusion": val_cm, "test_confusion": test_cm,
         "history": history, "args": vars(a)}, indent=1), encoding="utf-8")
    print(f"\nSaved {a.out} ({time.time() - t0:.0f}s)")
    print("Next, for each encoder (separate output root; the earlier predicted-tier runs stay untouched):")
    for n in [x.strip() for x in a.names.split(",") if x.strip()]:
        print(f"  python -u scripts/gate_lab.py --name {n} --final concat_tier --test-tier-file {a.out} --out-dir {a.router_out_dir}")
    print("then:  python -u scripts/tier_router.py --score-only")


if __name__ == "__main__":
    main()
