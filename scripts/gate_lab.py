#!/usr/bin/env python3
"""
Fast fusion/gate experiments on encoder outputs cached by cache_stage2_features.py.

Every variant shares the same cached DeBERTa [CLS] vectors and the same head as
DeSAF (proj -> BiLSTM(128x2) -> MLP 256-64-1); only the fusion step differs.

Screening (default): 5-fold cross-validation by document on the official
VALIDATION set (stratified by tier). For each held-out fold the head trains on 3
folds and early-stops + picks its threshold on the 4th. Each head seed uses its
own fold split; a variant and its control always share folds for the same seed.
The test set is NOT touched. Caveat: the checkpoint chose its epoch and threshold
on validation, so absolute validation scores (especially 'original') are a bit
optimistic. Use the screen for gated-vs-control differences, not absolute levels.

--final v1,v2 trains on all validation (85/15 split for early stopping and the
threshold) and evaluates those variants on the official test set once. Pick the
variant and its control BEFORE running it, and make claims from that run only.

Usage:
    python scripts/gate_lab.py --name seed43_gate
    python scripts/gate_lab.py --name seed43_gate --final original,scaled_overlap,overlap_gate
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TIERS = ["easy", "medium", "hard"]
KNOTS = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.18, 0.25, 0.35, 0.5], dtype=np.float32)
THRESHOLDS = np.round(np.arange(0.10, 0.905, 0.02), 2)
SURF_NAMES = ["fw_dist", "trigram_dissim", "punct_dist"]
N_SURF = 3
DEBERTA_DIM = 768

# style: use the 193 stylometric diffs + 3 surface distances
# norm: raw (as DeSAF's concat), ln (LayerNorm over the 193 diffs, as DeSAF's gate;
#       surface distances standardised separately), z (every feature standardised
#       with training-fold statistics)
# gate: what the gate sees (h = [CLS], r = topic-overlap basis, t = tier one-hot,
#       const = a learned per-feature weight that ignores the input)
# act: gate range (sigmoid 0..1 can only scale down; tanh -1..1 can also flip sign)
# overlap/tier: also append those inputs to the fused vector, so each gated
# variant and its controls see exactly the same information.
VARIANTS = {
    "cls_only":            dict(style=False, norm=None,  gate=None,    act=None,      overlap=False, tier=False),
    "concat":              dict(style=True,  norm="raw", gate=None,    act=None,      overlap=False, tier=False),
    "concat_ln":           dict(style=True,  norm="ln",  gate=None,    act=None,      overlap=False, tier=False),
    "gate":                dict(style=True,  norm="ln",  gate="h",     act="sigmoid", overlap=False, tier=False),
    "concat_z":            dict(style=True,  norm="z",   gate=None,    act=None,      overlap=False, tier=False),
    "scaled_z":            dict(style=True,  norm="z",   gate="const", act="tanh",    overlap=False, tier=False),
    "signed_gate":         dict(style=True,  norm="z",   gate="h",     act="tanh",    overlap=False, tier=False),
    "concat_overlap":      dict(style=True,  norm="z",   gate=None,    act=None,      overlap=True,  tier=False),
    "scaled_overlap":      dict(style=True,  norm="z",   gate="const", act="tanh",    overlap=True,  tier=False),
    "overlap_gate":        dict(style=True,  norm="z",   gate="r",     act="tanh",    overlap=True,  tier=False),
    "overlap_h_gate":      dict(style=True,  norm="z",   gate="hr",    act="tanh",    overlap=True,  tier=False),
    "concat_overlap_tier": dict(style=True,  norm="z",   gate=None,    act=None,      overlap=True,  tier=True),
    "overlap_tier_gate":   dict(style=True,  norm="z",   gate="rt",    act="tanh",    overlap=True,  tier=True),
    # round 2: pull apart what overlap_tier_gate's gain comes from
    "concat_tier":         dict(style=True,  norm="raw", gate=None,    act=None,      overlap=False, tier=True),
    "scaled_overlap_tier": dict(style=True,  norm="z",   gate="const", act="tanh",    overlap=True,  tier=True),
    "tier_gate":           dict(style=True,  norm="z",   gate="t",     act="tanh",    overlap=True,  tier=True),
}
# Fair controls for each gated variant (same inputs and normalisation):
#   concat_*  no gate at all
#   scaled_*  a learned per-feature weight that ignores the input, so a gated variant
#             can't win just by muting noisy features; beating it means the gate's
#             input (overlap, [CLS], tier) is what helps.
CONTROLS = {
    "gate": ["concat_ln"],
    "signed_gate": ["concat_z", "scaled_z"],
    "overlap_gate": ["concat_overlap", "scaled_overlap"],
    "overlap_h_gate": ["concat_overlap", "scaled_overlap"],
    # tier_gate: same inputs, but its gate sees only the tier, not the overlap
    "overlap_tier_gate": ["concat_overlap_tier", "scaled_overlap_tier", "tier_gate"],
}
# Other differences worth reading (A minus B), mostly about the value of the tier flag.
EXTRA_PAIRS = [
    ("concat_tier", "concat"),
    ("concat_overlap_tier", "concat_overlap"),
    ("overlap_tier_gate", "concat_tier"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="cache name used in cache_stage2_features.py")
    p.add_argument("--cache-dir", default="checkpoints/stage2_cache")
    p.add_argument("--out-dir", default="results/gate_lab")
    p.add_argument("--variants", default="all", help="comma list, or 'all' (includes 'original')")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-docs", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--final", default="", help="comma list of variants to evaluate ONCE on test")
    p.add_argument("--fresh", action="store_true",
                   help="ignore finished screening runs saved in --out-dir and redo everything")
    p.add_argument("--redo", default="", help="comma list of variants to retrain even if saved")
    p.add_argument("--test-tier-file", default="",
                   help="with --final: predicted test tiers from predict_tier.py; tier-aware heads are "
                        "then also scored with the PREDICTED tier as input (same trained head)")
    p.add_argument("--soft-tier", action="store_true",
                   help="feed the predicted tier probabilities instead of the most likely tier")
    p.add_argument("--train-on-predicted-tier", action="store_true",
                   help="with --test-tier-file: train the heads on the validation documents' "
                        "out-of-fold PREDICTED tiers too, and test on predicted tiers")
    return p.parse_args()


def tent_basis(o):
    """Piecewise-linear basis over KNOTS: lets a gate learn any smooth curve of overlap."""
    o = np.clip(o, KNOTS[0], KNOTS[-1])
    idx = np.clip(np.searchsorted(KNOTS, o, side="right") - 1, 0, len(KNOTS) - 2)
    w_right = (o - KNOTS[idx]) / (KNOTS[idx + 1] - KNOTS[idx])
    out = np.zeros((len(o), len(KNOTS)), dtype=np.float32)
    rows = np.arange(len(o))
    out[rows, idx] = 1.0 - w_right
    out[rows, idx + 1] += w_right
    return out


class SplitData:
    def __init__(self, path, device):
        z = np.load(path)
        self.lengths = z["doc_lengths"].astype(np.int64)
        self.doc_tier = z["doc_tier"].astype(np.int64)
        self.n_docs = len(self.lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        n = int(self.offsets[-1])
        self.labels = z["labels"].astype(np.int64)
        if len(self.labels) != n:
            raise ValueError(f"{path}: {len(self.labels)} labels but doc_lengths sum to {n}")
        self.pair_tier = np.repeat(self.doc_tier, self.lengths)
        self.probs = z["probs"].astype(np.float32)
        self.ckpt_threshold = float(z["threshold"]) if "threshold" in z.files else None
        self.doc_ids = z["doc_ids"] if "doc_ids" in z.files else None
        d = np.concatenate([z["style"], z["surf"]], axis=1).astype(np.float32)
        self.style_dim = d.shape[1]
        feats = {
            "h": z["h"].astype(np.float32),
            "d": d,
            "r": tent_basis(z["overlap"].astype(np.float32)),
            "t": np.eye(3, dtype=np.float32)[self.pair_tier],
            "y": self.labels.astype(np.float32)[:, None],
        }
        self.pad_index = n  # an all-zero row appended at index n, used for padding
        self.feats = {
            k: torch.from_numpy(np.concatenate([v, np.zeros((1, v.shape[1]), np.float32)])).to(device)
            for k, v in feats.items()
        }
        self.device = device

    def pair_index(self, docs):
        return np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in docs])

    def tier_input(self, doc_tier_matrix):
        """Tensor for the model's tier input from per-document rows of shape (n_docs, 3)."""
        pair = np.repeat(doc_tier_matrix.astype(np.float32), self.lengths, axis=0)
        return torch.from_numpy(np.concatenate([pair, np.zeros((1, 3), np.float32)])).to(self.device)

    def feature_stats(self, docs):
        """Per-feature mean/sd of the raw diffs over these docs; constant features get sd=inf (-> 0)."""
        d = self.feats["d"][torch.from_numpy(self.pair_index(docs)).to(self.device)]
        mu, sd = d.mean(0), d.std(0)
        sd = torch.where(sd < 1e-6, torch.full_like(sd, float("inf")), sd)
        return mu, sd

    def batch(self, docs):
        lens = self.lengths[docs]
        idx = np.full((len(docs), int(lens.max())), self.pad_index, dtype=np.int64)
        for row, di in enumerate(docs):
            idx[row, :lens[row]] = np.arange(self.offsets[di], self.offsets[di + 1])
        idx_t = torch.from_numpy(idx).to(self.device)
        b = {k: v[idx_t] for k, v in self.feats.items()}
        b["y"] = b["y"].squeeze(-1)
        b["mask"] = (idx_t != self.pad_index).float()
        b["lengths"] = torch.from_numpy(lens)
        return b


class Stage2Head(nn.Module):
    def __init__(self, cfg, style_dim, mu, sd, hidden=128, dropout=0.3):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mu", mu.clone())
        self.register_buffer("sd", sd.clone())
        n_r = len(KNOTS)
        fused = DEBERTA_DIM
        self.ln = None
        self.gate_net = None
        self.gate_param = None
        if cfg["style"]:
            fused += style_dim
            if cfg["norm"] == "ln":
                self.ln = nn.LayerNorm(style_dim - N_SURF)
            if cfg["gate"] == "const":
                self.gate_param = nn.Parameter(torch.zeros(style_dim))
            elif cfg["gate"]:
                cond_dim = {"h": DEBERTA_DIM, "r": n_r, "hr": DEBERTA_DIM + n_r, "rt": n_r + 3, "t": 3}[cfg["gate"]]
                gate_hidden = style_dim if "h" in cfg["gate"] else 64
                self.gate_net = nn.Sequential(
                    nn.Linear(cond_dim, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, style_dim),
                )
        if cfg["overlap"]:
            fused += n_r
        if cfg["tier"]:
            fused += 3
        self.proj = nn.Sequential(
            nn.Linear(fused, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(hidden * 2, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def style_input(self, d_raw):
        norm = self.cfg["norm"]
        if norm == "raw":
            return d_raw
        dz = torch.clamp((d_raw - self.mu) / self.sd, -10.0, 10.0)
        if norm == "z":
            return dz
        return torch.cat([self.ln(d_raw[..., :-N_SURF]), dz[..., -N_SURF:]], -1)

    def gate_values(self, b):
        g = self.cfg["gate"]
        if g == "const":
            return torch.tanh(self.gate_param)
        cond = {
            "h": lambda: b["h"],
            "r": lambda: b["r"],
            "hr": lambda: torch.cat([b["h"], b["r"]], -1),
            "rt": lambda: torch.cat([b["r"], b["t"]], -1),
            "t": lambda: b["t"],
        }[g]()
        raw = self.gate_net(cond)
        return torch.sigmoid(raw) if self.cfg["act"] == "sigmoid" else torch.tanh(raw)

    def forward(self, b):
        parts = [b["h"]]
        if self.cfg["style"]:
            d = self.style_input(b["d"])
            if self.cfg["gate"]:
                d = d * self.gate_values(b)
            parts.append(d)
        if self.cfg["overlap"]:
            parts.append(b["r"])
        if self.cfg["tier"]:
            parts.append(b["t"])
        x = self.proj(torch.cat(parts, -1))
        packed = pack_padded_sequence(x, b["lengths"], batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(out).squeeze(-1)


def f1_score(preds, labels):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def best_threshold(probs, labels):
    best_t, best_f = 0.5, -1.0
    for t in THRESHOLDS:
        f = f1_score((probs >= t).astype(np.int64), labels)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def tier_scores(preds, labels, pair_tier):
    s = {t: f1_score(preds[pair_tier == i], labels[pair_tier == i]) for i, t in enumerate(TIERS)}
    s["macro"] = float(np.mean([s[t] for t in TIERS]))
    return s


@torch.no_grad()
def predict(model, data, docs, batch_docs):
    model.eval()
    out = []
    for i in range(0, len(docs), batch_docs):
        chunk = docs[i:i + batch_docs]
        p = torch.sigmoid(model(data.batch(chunk))).cpu().numpy()
        for row, di in enumerate(chunk):
            out.append(p[row, :data.lengths[di]])
    return np.concatenate(out).astype(np.float32)


def train_head(variant, data, train_docs, dev_docs, seed, args):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mu, sd = data.feature_stats(train_docs)
    model = Stage2Head(VARIANTS[variant], data.style_dim, mu, sd, args.hidden, args.dropout).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=data.device),
                                 reduction="none")
    dev_labels = data.labels[data.pair_index(dev_docs)]
    best_f, best_t, best_state, bad = -1.0, 0.5, None, 0
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(train_docs)
        for i in range(0, len(order), args.batch_docs):
            b = data.batch(order[i:i + args.batch_docs])
            loss = (lossf(model(b), b["y"]) * b["mask"]).sum() / b["mask"].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        t, f = best_threshold(predict(model, data, dev_docs, args.batch_docs), dev_labels)
        if f > best_f:
            best_f, best_t, bad = f, t, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_t


@torch.no_grad()
def gate_curve_text(model, variant):
    """Gate weight on the 3 surface features. The overall sign of a gate is arbitrary, so every
    value is shown relative to ONE shared reference (Easy, or the lowest overlap); a '*' marks a
    reversal relative to it, across overlap AND across tiers (both are meaningful because the
    layer after the gate is shared)."""
    g_kind = VARIANTS[variant]["gate"]
    if g_kind not in ("r", "rt", "t"):
        return ""
    model.eval()
    dev = next(model.parameters()).device
    n_k = len(KNOTS)
    tiers = [None] if g_kind == "r" else [0, 1, 2]
    blocks = []
    for t in tiers:
        b = {"r": torch.eye(n_k, device=dev)}
        if t is not None:
            b["t"] = torch.zeros(n_k, 3, device=dev)
            b["t"][:, t] = 1.0
        blocks.append(model.gate_values(b)[:, -N_SURF:].cpu().numpy())
    ref = np.where(blocks[0][0] < 0, -1.0, 1.0)  # Easy (or only block), lowest overlap
    ref_name = {"r": "overlap 0", "rt": "Easy at overlap 0", "t": "Easy"}[g_kind]
    lines = [f"    learned gate on surface features ({variant}), signs relative to {ref_name};",
             "    '*' = reversed relative to that reference:"]

    def cells(row):
        return "  ".join(f"{v:14.3f}{'*' if v < 0 else ' '}" for v in row * ref)

    header = "  ".join(f"{n:>15s}" for n in SURF_NAMES)
    if g_kind == "t":
        lines.append(f"      tier     {header}")
        for t, blk in zip(tiers, blocks):
            lines.append(f"      {TIERS[t]:7s}  {cells(blk[0])}")
        return "\n".join(lines)
    for t, blk in zip(tiers, blocks):
        if t is not None:
            lines.append(f"      tier = {TIERS[t]}")
        lines.append(f"      overlap  {header}")
        for k, knot in enumerate(KNOTS):
            lines.append(f"      {knot:7.2f}  {cells(blk[k])}")
    return "\n".join(lines)


def save_sig_npz(path, data, preds, probs):
    """Same keys as save_test_predictions.py, so compute_significance.py can read it.
    Written to a temp file first so a crash mid-write never leaves a half-written result."""
    out = {}
    for i, t in enumerate(TIERS):
        m = data.pair_tier == i
        out[f"{t}__preds"] = preds[m]
        out[f"{t}__probs"] = probs[m]
        out[f"{t}__labels"] = data.labels[m]
        out[f"{t}__lengths"] = data.lengths[data.doc_tier == i]
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def load_scores(path):
    """Scores of a finished screening run, or None if it is missing or unreadable."""
    try:
        z = np.load(path)
        s = {t: f1_score(z[f"{t}__preds"], z[f"{t}__labels"]) for t in TIERS}
    except Exception:
        return None
    s["macro"] = float(np.mean([s[t] for t in TIERS]))
    return s


def make_folds(doc_tier, k, seed):
    folds = np.zeros(len(doc_tier), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for t in range(3):
        idx = np.where(doc_tier == t)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % k
    return folds


def fmt(s):
    return "  ".join(f"{t}={s[t]:.4f}" for t in TIERS + ["macro"])


def run_cv(data, variants, seeds, args, out_dir):
    redo = {x.strip() for x in args.redo.split(",") if x.strip()}
    results = {}
    for v in variants:
        runs = []
        for s in ([0] if v == "original" else seeds):
            t0 = time.time()
            label = "checkpoint" if v == "original" else f"seed {s}"
            path = out_dir / f"cv_{v}_s{s}.npz"
            curve_path = out_dir / f"gate_curve_{v}.txt"
            sc = None if (args.fresh or v in redo) else load_scores(path)
            if sc is not None:
                if s == seeds[0] and curve_path.exists():
                    print(curve_path.read_text())
                runs.append(sc)
                print(f"  {v:20s} {label:11s} {fmt(sc)}  (finished earlier, loaded)", flush=True)
                continue
            folds = make_folds(data.doc_tier, args.folds, args.split_seed + s)
            oof_prob = np.zeros(len(data.labels), np.float32)
            oof_pred = np.zeros(len(data.labels), np.int64)
            for k in range(args.folds):
                hold = np.where(folds == k)[0]
                dev = np.where(folds == (k + 1) % args.folds)[0]
                train = np.where((folds != k) & (folds != (k + 1) % args.folds))[0]
                if v == "original":
                    dev_idx = data.pair_index(dev)
                    thr, _ = best_threshold(data.probs[dev_idx], data.labels[dev_idx])
                    p = data.probs[data.pair_index(hold)]
                else:
                    model, thr = train_head(v, data, train, dev, 1000 * s + k, args)
                    p = predict(model, data, hold, args.batch_docs)
                    if k == 0 and s == seeds[0]:
                        curve = gate_curve_text(model, v)
                        if curve:
                            print(curve)
                            curve_path.write_text(curve)
                idx = data.pair_index(hold)
                oof_prob[idx] = p
                oof_pred[idx] = (p >= thr).astype(np.int64)
            sc = tier_scores(oof_pred, data.labels, data.pair_tier)
            runs.append(sc)
            save_sig_npz(path, data, oof_pred, oof_prob)
            print(f"  {v:20s} {label:11s} {fmt(sc)}  ({time.time() - t0:.0f}s)", flush=True)
        results[v] = runs
    return results


def summarize(results, n_folds):
    print("\n" + "=" * 92)
    print(f"VALIDATION {n_folds}-FOLD CV  (mean +/- sd over head seeds; test set not used)")
    print("=" * 92)
    print(f"{'variant':22s}" + "".join(f"{t:>17s}" for t in TIERS + ["macro"]))
    for v, runs in results.items():
        row = f"{v:22s}"
        for t in TIERS + ["macro"]:
            vals = np.array([r[t] for r in runs])
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row += f"   {vals.mean():.4f}+/-{sd:.4f}"
        print(row)
    print("('original' = the checkpoint itself; optimistic here because it was tuned on validation)")

    def diff_line(a, b):
        line = f"  {a:19s} vs {b:20s}"
        for t in ["medium", "hard", "macro"]:
            d = np.array([x[t] - y[t] for x, y in zip(results[a], results[b])])
            same = len(d) > 1 and (np.all(d > 0) or np.all(d < 0))
            line += f"  {t}: {d.mean():+.4f} [{', '.join(f'{x:+.4f}' for x in d)}]{' same sign' if same else ''}"
        return line

    print("\nGated variant minus each fair control, per seed (same folds within a seed).")
    print("'same sign' = the difference has the same sign on every seed (a screening hint, not proof):")
    for v, ctrls in CONTROLS.items():
        for ctrl in ctrls:
            if v in results and ctrl in results:
                print(diff_line(v, ctrl))
    pairs = [(a, b) for a, b in EXTRA_PAIRS if a in results and b in results]
    if pairs:
        print("\nOther differences (A minus B):")
        for a, b in pairs:
            print(diff_line(a, b))


def cv_threshold_scores(probs, data, pair_fold, n_folds, per_tier):
    """Threshold(s) picked on the other folds' out-of-fold probabilities, applied to this fold."""
    preds = np.zeros(len(probs), dtype=np.int64)
    for f in range(n_folds):
        test = pair_fold == f
        groups = [data.pair_tier == i for i in range(3)] if per_tier else [np.ones(len(probs), bool)]
        for g in groups:
            tr, te = (~test) & g, test & g
            thr, _ = best_threshold(probs[tr], data.labels[tr])
            preds[te] = (probs[te] >= thr).astype(np.int64)
    return tier_scores(preds, data.labels, data.pair_tier)


def threshold_check(data, variants, seeds, args, out_dir):
    """How much of each variant's score is just calibration: one threshold vs. one per tier."""
    print("\n" + "=" * 92)
    print("THRESHOLD CHECK: one global threshold vs. one threshold per tier (post-hoc, chosen by")
    print("cross-validation over each run's out-of-fold probabilities; mean over seeds)")
    print("=" * 92)
    print(f"{'variant':22s}{'hard (global)':>15s}{'macro (global)':>16s}"
          f"{'hard (per-tier)':>17s}{'macro (per-tier)':>18s}{'macro gain':>12s}")
    for v in variants:
        g_runs, t_runs = [], []
        for s in ([0] if v == "original" else seeds):
            try:
                z = np.load(out_dir / f"cv_{v}_s{s}.npz")
                probs = np.zeros(len(data.labels), dtype=np.float32)
                for i, t in enumerate(TIERS):
                    probs[data.pair_tier == i] = z[f"{t}__probs"]
            except Exception:
                continue
            folds = make_folds(data.doc_tier, args.folds, args.split_seed + s)
            pair_fold = np.repeat(folds, data.lengths)
            g_runs.append(cv_threshold_scores(probs, data, pair_fold, args.folds, per_tier=False))
            t_runs.append(cv_threshold_scores(probs, data, pair_fold, args.folds, per_tier=True))
        if not g_runs:
            continue
        m = lambda runs, k: float(np.mean([r[k] for r in runs]))
        print(f"{v:22s}{m(g_runs, 'hard'):15.4f}{m(g_runs, 'macro'):16.4f}"
              f"{m(t_runs, 'hard'):17.4f}{m(t_runs, 'macro'):18.4f}"
              f"{m(t_runs, 'macro') - m(g_runs, 'macro'):+12.4f}")


def run_final(val, test, variants, seeds, args, out_dir, pred_tier=None):
    rng = np.random.RandomState(args.split_seed)
    dev_mask = np.zeros(val.n_docs, dtype=bool)
    for t in range(3):
        idx = np.where(val.doc_tier == t)[0]
        idx = idx[rng.permutation(len(idx))]
        dev_mask[idx[: int(round(0.15 * len(idx)))]] = True
    train, dev = np.where(~dev_mask)[0], np.where(dev_mask)[0]
    all_test = np.arange(test.n_docs)
    print("\n" + "=" * 92)
    print("FINAL: heads trained on validation, evaluated ONCE on the official test set")
    print("=" * 92)
    summary = {}
    for v in variants:
        runs = []
        for s in ([0] if v == "original" else seeds):
            if v == "original":
                if val.ckpt_threshold is not None:
                    thr = val.ckpt_threshold
                else:
                    dev_idx = val.pair_index(dev)
                    thr, _ = best_threshold(val.probs[dev_idx], val.labels[dev_idx])
                p = test.probs
            else:
                model, thr = train_head(v, val, train, dev, 7777 + s, args)
                p = predict(model, test, all_test, args.batch_docs)
                if s == seeds[0]:
                    curve = gate_curve_text(model, v)
                    if curve:
                        print(curve)
            preds = (p >= thr).astype(np.int64)
            sc = tier_scores(preds, test.labels, test.pair_tier)
            runs.append(sc)
            save_sig_npz(out_dir / f"final_{v}_s{s}_test.npz", test, preds, p)
            label = "checkpoint" if v == "original" else f"seed {s}"
            print(f"  {v:20s} {label:11s} thr={thr:.2f}  {fmt(sc)}", flush=True)
            if pred_tier is not None and v != "original" and VARIANTS[v]["tier"]:
                # Same trained head and threshold; only the tier INPUT changes. Scores are still
                # reported per TRUE tier, as PAN evaluates them.
                true_t = test.feats["t"]
                test.feats["t"] = pred_tier
                p2 = predict(model, test, all_test, args.batch_docs)
                test.feats["t"] = true_t
                preds2 = (p2 >= thr).astype(np.int64)
                sc2 = tier_scores(preds2, test.labels, test.pair_tier)
                summary.setdefault(v + " [predicted tier]", []).append(sc2)
                save_sig_npz(out_dir / f"final_{v}_s{s}_test_predtier.npz", test, preds2, p2)
                print(f"  {v + ' [pred. tier]':32s} thr={thr:.2f}  {fmt(sc2)}", flush=True)
        summary[v] = runs
    if "original" in variants and val.ckpt_threshold is not None:
        print("('original' uses the checkpoint's own threshold, so it should equal the F1 that")
        print(" save_test_predictions.py printed for this checkpoint)")
    return summary


def main():
    args = parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_names = ["original"] + list(VARIANTS)
    variants = all_names if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    final = [v.strip() for v in args.final.split(",") if v.strip()]
    redo = [v.strip() for v in args.redo.split(",") if v.strip()]
    unknown = [v for v in variants + final + redo if v not in all_names]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Choose from: {all_names}")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | cache: {cache_dir / args.name}_*.npz | head seeds: {seeds}")
    val = SplitData(cache_dir / f"{args.name}_validation.npz", device)
    print(f"Validation: {val.n_docs} docs, {len(val.labels)} boundaries, style dim {val.style_dim}")

    if final:
        test = SplitData(cache_dir / f"{args.name}_test.npz", device)
        pred_tier = None
        if args.test_tier_file:
            tp = np.load(args.test_tier_file)
            if test.doc_ids is not None and not np.array_equal(tp["doc_ids"], test.doc_ids):
                raise SystemExit("The tier file's documents are not in the same order as this test cache.")
            probs = tp["probs"].astype(np.float32)
            matrix = probs if args.soft_tier else np.eye(3, dtype=np.float32)[probs.argmax(1)]
            acc = float((probs.argmax(1) == test.doc_tier).mean())
            print(f"Predicted test tiers: {args.test_tier_file} | document-level accuracy {acc:.3f} | "
                  f"input = {'probabilities' if args.soft_tier else 'most likely tier'}")
            pred_tier = test.tier_input(matrix)
            sub = "predicted_tier"
            if args.train_on_predicted_tier:
                if "val_probs" not in tp.files:
                    raise SystemExit("This tier file has no validation predictions; rerun predict_tier.py.")
                if val.doc_ids is not None and not np.array_equal(tp["val_doc_ids"], val.doc_ids):
                    raise SystemExit("The tier file's validation documents are not in this cache's order.")
                vp = tp["val_probs"].astype(np.float32)
                v_matrix = vp if args.soft_tier else np.eye(3, dtype=np.float32)[vp.argmax(1)]
                v_acc = float((vp.argmax(1) == val.doc_tier).mean())
                print(f"Training heads on out-of-fold PREDICTED validation tiers (accuracy {v_acc:.3f}); "
                      "test also uses predicted tiers")
                val.feats["t"] = val.tier_input(v_matrix)
                test.feats["t"] = pred_tier
                pred_tier = None
                sub = "predicted_tier_trained"
            # Keep the earlier pre-registered final results untouched.
            out_dir = out_dir / (sub + ("_soft" if args.soft_tier else ""))
            out_dir.mkdir(parents=True, exist_ok=True)
        elif args.train_on_predicted_tier:
            raise SystemExit("--train-on-predicted-tier needs --test-tier-file")
        summary = run_final(val, test, final, seeds, args, out_dir, pred_tier)
        (out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nPer-boundary predictions saved in {out_dir} (final_*_test.npz) for compute_significance.py")
        return

    print(f"Screening {len(variants)} variants with {args.folds}-fold CV on validation only.\n")
    results = run_cv(val, variants, seeds, args, out_dir)
    summarize(results, args.folds)
    threshold_check(val, list(results), seeds, args, out_dir)
    (out_dir / "cv_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nOut-of-fold predictions saved in {out_dir} (cv_*.npz) for compute_significance.py")


if __name__ == "__main__":
    main()
