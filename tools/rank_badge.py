#!/usr/bin/env python3
"""Знак над пешкой: есть ли он и что в нём.

Над пешками встречаются белые надписи вроде «1/6», «2/6». Если первое
число — ранг, это и есть готовый признак. Скрипт для каждой пешки из
трекера ищет белую надпись над головой и сводит по рангам.

    .venv/bin/python tools/rank_badge.py <csv> <каталог кадров>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def probe(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    pawns = []
    for side, x, y, r in re.findall(r"(своя|чужая) (\d+),(\d+) радиус (\d+)", out):
        if side == "своя":
            pawns.append((int(x), int(y), int(r)))
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2))), pawns


def badge(image, cx, cy, radius):
    """Белые пиксели над головой пешки — предполагаемый знак ранга."""
    half = int(radius * 1.6)
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
    y0, y1 = max(0, cy - int(radius * 2.4)), max(0, cy - int(radius * 0.7))
    if x1 <= x0 or y1 <= y0:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = (gray > 210).astype(np.uint8)
    if mask.sum() < 40:
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = 0
    area = 0
    for i in range(1, count):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < 12 or h < 6 or h > 40 or w > 60:
            continue
        blobs += 1
        area += stats[i, cv2.CC_STAT_AREA]
    if not blobs:
        return None
    return {"blobs": blobs, "area": area}


def main() -> int:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    by_frame: dict[str, list] = {}
    for row in csv.DictReader(csv_path.open()):
        by_frame.setdefault(row["frame"], []).append(row)
    found = {1: [0, 0], 2: [0, 0]}
    samples = []
    for frame_name, group in by_frame.items():
        frame = directory / frame_name
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        arena, pawns = probe(frame)
        if not arena:
            continue
        ax, ay = arena
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
            mark = badge(image, best[0], best[1], best[2])
            rank = int(row["rank"])
            key = rank if rank == 1 else 2
            found[key][1] += 1
            if mark:
                found[key][0] += 1
                if key == 2 and len(samples) < 12:
                    samples.append((frame_name, best, mark))
    print(f"ранг 1:  надпись у {found[1][0]} из {found[1][1]}")
    print(f"ранг ≥2: надпись у {found[2][0]} из {found[2][1]}")
    for frame_name, pawn, mark in samples:
        print(f"  {frame_name} пешка {pawn[:2]} блобов {mark['blobs']} площадь {mark['area']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
