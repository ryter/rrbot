#!/usr/bin/env python3
"""Ранг как относительный размер.

Ранг 2 нарисован крупнее ранга 1. Но абсолютный размер зависит от типа
юнита, поэтому пешку надо сравнивать с соседями по кадру: если она крупнее
медианы своей половины поля — она, вероятно, старшего ранга.

    .venv/bin/python tools/rank_size.py <csv>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("данных нет")
        return 1
    frames = defaultdict(list)
    for row in rows:
        frames[(row["round"], row["step"])].append(row)
    ratios = {1: [], 2: []}
    for _, group in frames.items():
        radii = np.array([float(r["r"]) for r in group])
        if len(radii) < 3:
            continue
        median = np.median(radii)
        for row in group:
            ratio = float(row["r"]) / median
            rank = int(row["rank"])
            ratios.setdefault(rank, []).append(ratio)
    for rank in sorted(ratios):
        values = np.array(ratios[rank])
        if not len(values):
            continue
        print(f"ранг {rank}: пешек {len(values):5d}  "
              f"r/медиана: мин {values.min():.2f} медиана {np.median(values):.2f} "
              f"макс {values.max():.2f}")
    print()
    # порог «крупнее медианы на X %»
    for extra in (0.0, 0.05, 0.10, 0.15, 0.20):
        threshold = 1.0 + extra
        pos = np.array(ratios.get(2, []))
        neg = np.array(ratios.get(1, []))
        if not len(pos) or not len(neg):
            continue
        correct = (pos >= threshold).sum() + (neg < threshold).sum()
        print(f"порог r ≥ {threshold:.2f}·медиана: верных "
              f"{correct / (len(pos) + len(neg)) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
