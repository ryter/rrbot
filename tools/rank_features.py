#!/usr/bin/env python3
"""Какой признак различает ранги.

Берёт CSV трекера (tools/rank_tracker.py) и считает, насколько каждый
признак разделяет «ранг 1» и «ранг ≥ 2»: разнесение медиан, доля верных
решений по порогу, AUC.

    .venv/bin/python tools/rank_features.py <csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

FEATURES = ("r", "fill", "stretch", "area", "rmax", "sw", "sh", "sa", "round_pedestal")


def auc(pos, neg):
    """AUC по Mann-Whitney: вероятность, что случайный pos > случайный neg."""
    if not pos or not neg:
        return float("nan")
    values = np.array(pos + neg, dtype=float)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # средние ранги для равных значений
    unique, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    np.add.at(counts, inverse, 1)
    ranks = (sums / counts)[inverse]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(pos, neg):
    """Лучший порог и доля верных решений."""
    values = sorted(set(pos) | set(neg))
    best = (0.0, 0.5, "—")
    for threshold in values:
        correct = sum(1 for v in pos if v >= threshold) + \
                  sum(1 for v in neg if v < threshold)
        accuracy = correct / (len(pos) + len(neg))
        if accuracy > best[0]:
            best = (accuracy, threshold, "выше" if pos and
                    np.median(pos) >= np.median(neg) else "ниже")
    return best


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("данных нет")
        return 1
    rank1 = [r for r in rows if int(r["rank"]) == 1]
    rank2 = [r for r in rows if int(r["rank"]) >= 2]
    print(f"ранг 1: {len(rank1)}, ранг ≥2: {len(rank2)}\n")
    print(f"{'признак':16} {'AUC':>6} {'порог':>10} {'верных':>8}  разделяет")
    scored = []
    for feature in FEATURES:
        pos = [float(r[feature]) for r in rank2]
        neg = [float(r[feature]) for r in rank1]
        a = auc(pos, neg)
        accuracy, threshold, direction = best_threshold(pos, neg)
        scored.append((abs(a - 0.5), feature, a, accuracy, threshold, direction))
    for _, feature, a, accuracy, threshold, direction in sorted(scored, reverse=True):
        print(f"{feature:16} {a:6.3f} {threshold:10.3f} {accuracy * 100:7.1f}%  "
              f"ранг2 {direction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
