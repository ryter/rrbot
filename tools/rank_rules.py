#!/usr/bin/env python3
"""Оценка правил определения ранга по размеченным данным.

Правило «пешка первого ранга» проверяется на данных трекера
(tools/rank_tracker.py). Для каждого правила считается полнота и
ложные срабатывания: доля пешек ранга 1, принятых за первый ранг, и доля
пешек ранга ≥2, принятых за первый ранг.

    .venv/bin/python tools/rank_rules.py <csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

RULES = {
    "текущее: fill ≥ 0.88": lambda f, s: f >= 0.88,
    "только stretch ≤ 1.15": lambda f, s: s <= 1.15,
    "fill ≥ 0.80 и stretch ≤ 1.20": lambda f, s: f >= 0.80 and s <= 1.20,
    "fill ≥ 0.85 и stretch ≤ 1.15": lambda f, s: f >= 0.85 and s <= 1.15,
    "fill ≥ 0.88 и stretch ≤ 1.10": lambda f, s: f >= 0.88 and s <= 1.10,
    "fill ≥ 0.90 и stretch ≤ 1.12": lambda f, s: f >= 0.90 and s <= 1.12,
    "fill ≥ 0.92 и stretch ≤ 1.15": lambda f, s: f >= 0.92 and s <= 1.15,
    "stretch ≤ 1.05": lambda f, s: s <= 1.05,
    "fill ≥ 0.93": lambda f, s: f >= 0.93,
}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("данных нет")
        return 1
    fills = [float(r["fill"]) for r in rows]
    stretches = [float(r["stretch"]) for r in rows]
    ranks = [int(r["rank"]) for r in rows]
    first = [i for i, r in enumerate(ranks) if r == 1]
    higher = [i for i, r in enumerate(ranks) if r >= 2]
    print(f"ранг 1: {len(first)}, ранг ≥2: {len(higher)}\n")
    print(f"{'правило':36} {'полнота':>8} {'ложных':>8} {'F1':>6}")
    scored = []
    for name, rule in RULES.items():
        flags = np.array([bool(rule(fills[i], stretches[i])) for i in range(len(rows))])
        tp = int(flags[first].sum())
        fn = len(first) - tp
        fp = int(flags[higher].sum())
        recall = tp / max(1, len(first))
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        scored.append((f1, name, recall, fp, len(higher)))
    for f1, name, recall, fp, total_higher in sorted(scored, reverse=True):
        print(f"{name:36} {recall * 100:7.1f}% {fp:8d} {f1:6.3f}"
              f"   (из {total_higher} старших принято {fp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
