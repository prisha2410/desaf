#!/usr/bin/env python3
"""
Real dataset statistics table for pan2025_full: documents, boundaries, and
class balance per tier/split. No training, no GPU -- just reads the data.

Usage:
    python scripts/dataset_stats_table.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_pan2025_full

TIERS = ["easy", "medium", "hard"]
SPLITS = ["train", "validation", "test"]


def stats_for(docs):
    n_docs = len(docs)
    n_boundaries = sum(len(d.boundary_labels) for d in docs)
    n_changes = sum(sum(d.boundary_labels) for d in docs)
    change_rate = n_changes / n_boundaries if n_boundaries else 0.0
    avg_paras = sum(len(d.sentences) for d in docs) / n_docs if n_docs else 0.0
    return n_docs, n_boundaries, n_changes, change_rate, avg_paras


def main():
    print(f"{'Tier':<8}{'Split':<12}{'Docs':>8}{'Boundaries':>12}{'Changes':>10}{'Change%':>10}{'Avg paras/doc':>16}")
    print("-" * 78)

    grand_totals = {"docs": 0, "boundaries": 0, "changes": 0}

    for tier in TIERS:
        for split in SPLITS:
            docs = load_pan2025_full("data/raw", tier=tier, split=split)
            n_docs, n_bound, n_change, rate, avg_p = stats_for(docs)
            print(f"{tier:<8}{split:<12}{n_docs:>8}{n_bound:>12}{n_change:>10}{rate*100:>9.1f}%{avg_p:>16.2f}")
            grand_totals["docs"] += n_docs
            grand_totals["boundaries"] += n_bound
            grand_totals["changes"] += n_change
        print()

    print("-" * 78)
    print(f"{'TOTAL':<20}{grand_totals['docs']:>8}{grand_totals['boundaries']:>12}"
         f"{grand_totals['changes']:>10}{grand_totals['changes']/grand_totals['boundaries']*100:>9.1f}%")


if __name__ == "__main__":
    main()