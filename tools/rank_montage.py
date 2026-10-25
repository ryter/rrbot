#!/usr/bin/env python3
"""Монтаж пешек по рангам: видно ли разницу глазами.

Берёт CSV трекера (tools/rank_tracker.py), для каждой пешки кадрирует
область вокруг основания и собирает два монтажа — «ранг 1» и «ранг ≥2».
Над головой пешки оставляется место: там может быть знак ранга.

    .venv/bin/python tools/rank_montage.py <csv> <каталог кадров> [максимум]
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
TILE = 96
ZOOM = 2


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


def main() -> int:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    by_frame: dict[str, list] = {}
    for row in csv.DictReader(csv_path.open()):
        by_frame.setdefault(row["frame"], []).append(row)

    groups = {1: [], 2: []}
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
            rank = int(row["rank"])
            key = rank if rank == 1 else 2
            if len(groups[key]) >= limit:
                continue
            px = int(row["x"]) + ax
            py = int(row["y"]) + ay
            half = TILE // 2
            x0 = max(0, min(px - half, image.shape[1] - TILE))
            y0 = max(0, min(py - half + 18, image.shape[0] - TILE))
            crop = image[y0 : y0 + TILE, x0 : x0 + TILE]
            big = cv2.resize(crop, (TILE * ZOOM, TILE * ZOOM), interpolation=cv2.INTER_CUBIC)
            cv2.drawMarker(big, ((px - x0) * ZOOM, (py - y0) * ZOOM), (0, 0, 255),
                           cv2.MARKER_CROSS, 10, 1)
            cv2.putText(big, f"r{rank}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            groups[key].append(big)

    for key, title in ((1, "rank1"), (2, "rank2plus")):
        tiles = groups[key]
        if not tiles:
            print(f"{title}: пусто")
            continue
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        th = tiles[0].shape[0]
        sheet = np.zeros((rows * th, cols * th, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * th : (r + 1) * th, c * th : (c + 1) * th] = tile
        out = Path(f"/tmp/opencode/rank_{title}.png")
        cv2.imwrite(str(out), sheet)
        print(f"{title}: {len(tiles)}, файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
