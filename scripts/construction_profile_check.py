#!/usr/bin/env python3
"""
Does any construction reproduce PAN 2025 Hard's PROFILE, not only its signs?  CPU, a few minutes.

Checks the constructions built by construction_experiment.py (topic_length and topic_rawstyle, with random and
topic_style as references) against the criteria fixed before the run (completeness critic, 2026-11-09), using the same
profile and lag code as pan_extra_checks.py:
  C1 content-overlap AUC <= 0.35                         (PAN 2025 Hard validation 0.277)
  C2 |log length ratio| AUC <= 0.35                      (0.287)
  C3 fw_dist unmatched <= 0.45 AND >= 0.47 at exact length (0.411 -> 0.490)
  C4 punct_dist unmatched <= 0.47 AND >= 0.50 at exact length (0.419 -> 0.546)
  C5 trigram <= 0.46 at length x topic                   (0.427)
  C6 lag-1 content <= 0.40 AND lags 2-3 within 0.05 of chance (0.283; 0.482 / 0.482)
  C7 change-rate ratio, pairs with a sentence < 8 words / both >= 8 words, <= 0.5 (0.39)
"Reproduces PAN Hard's profile" only if one construction meets C1-C5. Run once; report both whatever the outcome.
Also printed (not a criterion): placement = AUC that the shorter sentence is longer at a change (PAN Hard 0.718).

Run from D:\\sass-tad after
    python scripts/construction_experiment.py ... --name reddit2025test_tl --constructions random,topic_style,topic_length,topic_rawstyle
    python scripts/construction_experiment.py ... --name c50_tl --constructions random,topic_style,topic_length,topic_rawstyle
with
    python scripts/construction_profile_check.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pan_extra_checks import lag_table, profile, content_vectorizer, read_jsonl  # noqa: E402
from pan_direction_followup import auc_fast  # noqa: E402

PAN_HARD = {"content": 0.277, "loglen": 0.287, "fw": (0.411, 0.490), "punct": (0.419, 0.546), "trigram_lt": 0.427,
            "lag": (0.283, 0.482, 0.482), "short_long_ratio": 0.39, "placement": 0.718}


def short_long(docs):
    y, wmin = [], []
    for s, lab in docs:
        w = [max(len(x.split()), 1) for x in s]
        for i, l in enumerate(lab):
            y.append(l); wmin.append(min(w[i], w[i + 1]))
    y, wmin = np.array(y), np.array(wmin, float)
    short = wmin < 8
    ratio = y[short].mean() / max(y[~short].mean(), 1e-12) if short.any() and (~short).any() else float("nan")
    return float(ratio), float(auc_fast(wmin, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructions-dir", default="results/construction")
    ap.add_argument("--names", default="reddit2025test_tl,c50_tl")
    ap.add_argument("--kinds", default="random,topic_style,topic_length,topic_rawstyle")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/construction/profile_check.json")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    report = {}
    for name in [n.strip() for n in a.names.split(",") if n.strip()]:
        summ_f = Path(a.constructions_dir) / f"{name}_summary.json"
        summ = json.loads(summ_f.read_text(encoding="utf-8"))["constructions"] if summ_f.exists() else {}
        cons = {k: read_jsonl(Path(a.constructions_dir) / f"{name}_{k}.jsonl") for k in kinds
                if (Path(a.constructions_dir) / f"{name}_{k}.jsonl").exists()}
        if not cons:
            print(f"{name}: no construction files found (skipped)"); continue
        vec = content_vectorizer([s for docs in cons.values() for d in docs for s in d[0]])
        print(f"\n{'=' * 120}\n{name}: profile and lag (same code as pan_extra_checks.py)\n{'=' * 120}")
        rows = {}
        for k, docs in cons.items():
            prof = profile(docs, vec, rng, a.boot, f"{name} {k}")
            lag = lag_table(docs, vec, rng, a.boot, f"{name} {k}")
            ratio, place = short_long(docs)
            auc = summ.get(k, {}).get("auc", {})
            content = auc.get("content overlap", {}).get("auc", float("nan"))
            loglen = auc.get("|log length ratio|", {}).get("auc", float("nan"))
            fw_u, tri_u, pu_u = prof["unmatched"]["auc"]
            fw_e, _, pu_e = prof["exact length"]["auc"]
            tri_lt = prof["length x topic"]["auc"][1]
            l1, l2, l3 = [x["auc"] for x in lag["content_lag"]] if lag else (float("nan"),) * 3
            crit = {"C1": content <= 0.35, "C2": loglen <= 0.35, "C3": fw_u <= 0.45 and fw_e >= 0.47,
                    "C4": pu_u <= 0.47 and pu_e >= 0.50, "C5": tri_lt <= 0.46,
                    "C6": l1 <= 0.40 and abs(l2 - 0.5) < 0.05 and abs(l3 - 0.5) < 0.05, "C7": ratio <= 0.5}
            vals = {"content": content, "loglen": loglen, "fw": [fw_u, fw_e], "punct": [pu_u, pu_e], "trigram_lt": tri_lt,
                    "lag": [l1, l2, l3], "short_long_ratio": ratio, "placement": place}
            rows[k] = {"values": vals, "criteria": {c: bool(v) for c, v in crit.items()},
                       "meets_C1_C5": all(crit[c] for c in ("C1", "C2", "C3", "C4", "C5"))}
        print(f"\n  CRITERIA vs PAN 2025 Hard (validation reference in brackets)")
        hdr = ["C1 content", "C2 |loglen|", "C3 fw u->len", "C4 punct u->len", "C5 tri LxT", "C6 lag 1/2/3",
               "C7 short/long", "placement"]
        print(f"  {'construction':16s}" + "".join(f"{h:>22s}" for h in hdr) + "   C1-C5?")
        ref = PAN_HARD
        print(f"  {'[PAN 2025 Hard]':16s}{ref['content']:>22.3f}{ref['loglen']:>22.3f}"
              f"{'%.3f->%.3f' % ref['fw']:>22s}{'%.3f->%.3f' % ref['punct']:>22s}{ref['trigram_lt']:>22.3f}"
              f"{'%.3f/%.3f/%.3f' % ref['lag']:>22s}{ref['short_long_ratio']:>22.2f}{ref['placement']:>22.3f}")
        for k, r in rows.items():
            v, c = r["values"], r["criteria"]
            mark = lambda ok: "+" if ok else "-"
            cells = [f"{v['content']:.3f} {mark(c['C1'])}", f"{v['loglen']:.3f} {mark(c['C2'])}",
                     f"{v['fw'][0]:.3f}->{v['fw'][1]:.3f} {mark(c['C3'])}", f"{v['punct'][0]:.3f}->{v['punct'][1]:.3f} {mark(c['C4'])}",
                     f"{v['trigram_lt']:.3f} {mark(c['C5'])}", f"{v['lag'][0]:.3f}/{v['lag'][1]:.3f}/{v['lag'][2]:.3f} {mark(c['C6'])}",
                     f"{v['short_long_ratio']:.2f} {mark(c['C7'])}", f"{v['placement']:.3f}"]
            print(f"  {k:16s}" + "".join(f"{x:>22s}" for x in cells) + f"   {'YES' if r['meets_C1_C5'] else 'no'}")
        print("  (+ = criterion met, - = not met; 'u->len' = unmatched -> exact matched length)")
        report[name] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
