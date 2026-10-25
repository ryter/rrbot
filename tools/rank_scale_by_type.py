#!/usr/bin/env python3
"""Ранг = масштаб относительно эталона той же карты.

Форма основания кодирует ТИП карты (внутри одного типа заполненность
одинакова на всех рангах). А раг в Rush Royale — это масштаб юнита:
ранг 2 нарисован крупнее ранга 1 той же карты.

Проверяем: внутри одного типа (по палитре) сравниваем радиус основания
у ранга 1 и ранга ≥2. Масштаб камеры убираем делением на ширину арены.

    .venv/bin/python tools/rank_scale_by_type.py <csv> <каталог кадров>
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
    inner = max(4, radius - 3)
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    patch = image[y0:cy + radius, x0:cx + radius]
    if patch.shape[0] < 2 * radius or patch.shape[1] < 2 * radius:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = np.zeros(HIST_H * HIST_S * HIST_V, np.float32)
    cyy, cxx = np.mgrid[0:2 * radius, 0:2 * radius]
    mask = (cxx - radius) ** 2 + (cyy - radius) ** 2 <= inner * inner
    for pixel in hsv[mask]:
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
        out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                             capture_output=True, text=True).stdout
        m = re.search(r"арена: своя (-?\d+),(-?\d+) (\d+)x(\d+)", out)
        if not m:
            continue
        ax, ay, aw = int(m.group(1)), int(m.group(2)), int(m.group(3))
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
            items.append((pal, int(row["rank"]), best[2] / aw, best[2]))
    if not items:
        print("данных нет")
        return 1
    groups: list[list] = []
    for item in items:
        for group in groups:
            if cosine(group[0][0], item[0]) >= 0.95:
                group.append(item)
                break
        else:
            groups.append([item])
    print(f"пешек: {len(items)}, типов: {len(groups)}\n")
    print(f"{'тип':>4} {'ранг1: r/арена':>18} {'ранг≥2: r/арена':>18} {'больше?':>9}")
    for index, group in enumerate(sorted(groups, key=len, reverse=True)):
        one, high = [], []
        for _, rank, norm, radius in group:
            (one if rank == 1 else high).append(norm)
        if not one or not high:
            continue
        bigger = sum(1 for h in high if h > np.mean(one))
        print(f"{index:>4} {np.mean(one):18.4f} (n={len(one)}) "
              f"{np.mean(high):18.4f} (n={len(high)}) {bigger}/{len(high):>6}")
    print()
    all_one = [v for _, rank, v, _ in items if rank == 1]
    all_high = [v for _, rank, v, _ in items if rank >= 2]
    print(f"все ранги 1: r/арена медиана {np.median(all_one):.4f} "
          f"(n={len(all_one)})")
    print(f"все ранги ≥2: r/арена медиана {np.median(all_high):.4f} "
          f"(n={len(all_high)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
