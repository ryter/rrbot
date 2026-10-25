#!/usr/bin/env python3
"""Один тип юнита на разных рангах: отличается ли форма основания?

Типы пешек различаются по палитре (как в merge.cpp: гистограмма HSV по кругу
основания, косинус). Собираем пешек из трекера, группируем по палитре (это
типы), и внутри каждой группы смотрим, как форма основания зависит от ранга.

Если внутри типа форма одинакова — форма кодирует ТИП, а не ранг.

    .venv/bin/python tools/rank_shape_types.py <csv> <каталог кадров>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
HIST_H, HIST_S, HIST_V = 18, 4, 4


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y), int(r)) for x, y, r in
            re.findall(r"своя (\d+),(\d+) радиус (\d+)", out)]


def palette(image, cx, cy, radius):
    """Гистограмма HSV по кругу основания (как merge.cpp)."""
    inner = max(4, radius - 3)
    patch = image[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
    if patch.shape[0] < 2 * radius or patch.shape[1] < 2 * radius:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = np.zeros(HIST_H * HIST_S * HIST_V, np.float32)
    cyy, cxx = np.mgrid[0:2 * radius, 0:2 * radius]
    mask = (cxx - radius) ** 2 + (cyy - radius) ** 2 <= inner * inner
    pixels = hsv[mask]
    for pixel in pixels:
        h = min(HIST_H - 1, int(pixel[0]) * HIST_H // 180)
        s = min(HIST_S - 1, int(pixel[1]) * HIST_S // 256)
        v = min(HIST_V - 1, int(pixel[2]) * HIST_V // 256)
        hist[h * HIST_S * HIST_V + s * HIST_V + v] += 1
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    by_frame: dict[str, list] = {}
    for row in csv.DictReader(csv_path.open()):
        by_frame.setdefault(row["frame"], []).append(row)

    # пешки с палитрой: (палитра, ранг, fill, stretch, r)
    items = []
    for frame_name, group in by_frame.items():
        has_high = any(int(r["rank"]) >= 2 for r in group)
        if not has_high:
            continue
        frame = directory / frame_name
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        arena = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                               capture_output=True, text=True).stdout
        m = re.search(r"арена: своя (-?\d+),(-?\d+) ", arena)
        if not m:
            continue
        ax, ay = int(m.group(1)), int(m.group(2))
        pawns = pawns_of(frame)
        for row in group:
            px = int(row["x"]) + ax
            py = int(row["y"]) + ay
            best = None
            best_d = 60
            for p in pawns:
                d = max(abs(p[0] - px), abs(p[1] - py))
                if d < best_d:
                    best_d, best = d, p
            if not best:
                continue
            pal = palette(image, best[0], best[1], best[2])
            if pal is None:
                continue
            items.append((pal, int(row["rank"]), float(row["fill"]),
                          float(row["stretch"]), int(row["r"])))
    if not items:
        print("данных нет")
        return 1
    print(f"пешек: {len(items)}")

    # группируем по палитре (жадное сопоставление по косинусу)
    groups: list[list] = []
    for item in items:
        placed = False
        for group in groups:
            if cosine(group[0][0], item[0]) >= 0.95:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    print(f"групп (типов): {len(groups)}\n")
    print(f"{'тип':>4} {'пешек':>7} {'ранг1: fill':>14} {'ранг≥2: fill':>14} "
          f"{'форм':>8}")
    for index, group in enumerate(sorted(groups, key=len, reverse=True)):
        if len(group) < 2:
            continue
        by_rank = defaultdict(list)
        for _, rank, fill, stretch, _ in group:
            by_rank[rank].append(fill)
        high = by_rank.get(2, []) + by_rank.get(3, [])
        def fmt(values):
            return f"{np.mean(values):.3f} (n={len(values)})" if values else "—"
        print(f"{index:>4} {len(group):>7} {fmt(by_rank.get(1, [])):>14} "
              f"{fmt(high):>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
