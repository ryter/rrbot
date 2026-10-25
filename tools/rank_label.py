#!/usr/bin/env python3
"""Надпись над пешкой («N/6») — это ранг?

В снимках видны белые надписи вида «1/6», «2/6» прямо на пешках. Если
первое число — ранг, это и есть готовый признак: у только что призванной
пешки ранг 1, у результата слияния ранг 2.

Скрипт собирает кропы «надписи» для своих пешек и сводит их с рангом,
выведенным из лога боя (трекер tools/rank_tracker.py).

    .venv/bin/python tools/rank_label.py <лог> <каталог кадров>
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


def arena_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    m = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def label_crop(image, cx, cy, radius):
    """Кроп верхней части пешки, где стоит надпись."""
    half = int(radius * 1.3)
    x0 = max(0, min(cx - half, image.shape[1] - 2 * half))
    y0 = max(0, cy - int(radius * 2.0))
    if y0 + 2 * half > image.shape[0] or x0 + 2 * half > image.shape[1]:
        return None
    return image[y0:y0 + 2 * half, x0:x0 + 2 * half]


def has_text(crop):
    """Есть ли в кропе белый текст (несколько ярких глифов)."""
    if crop is None:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray > 225).astype(np.uint8)
    if mask.sum() < 60:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    glyphs = 0
    for i in range(1, count):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 4 <= w <= 30 and 8 <= h <= 34 and stats[i, cv2.CC_STAT_AREA] >= 25:
            glyphs += 1
    return glyphs >= 2


def main() -> int:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    rows = list(csv.DictReader(csv_path.open()))
    by_frame: dict[str, list] = {}
    for row in rows:
        by_frame.setdefault(row["frame"], []).append(row)
    stats = {1: [0, 0], 2: [0, 0]}
    tiles = []
    for frame_name, group in by_frame.items():
        frame = directory / frame_name
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        arena = arena_of(frame)
        if not arena:
            continue
        ax, ay = arena
        for row in group:
            px = int(row["x"]) + ax
            py = int(row["y"]) + ay
            radius = int(row["r"])
            crop = label_crop(image, px, py, radius)
            rank = int(row["rank"])
            key = rank if rank == 1 else 2
            stats[key][1] += 1
            if has_text(crop):
                stats[key][0] += 1
                if len(tiles) < 24:
                    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                     interpolation=cv2.INTER_CUBIC)
                    cv2.putText(big, f"r{rank}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                    tiles.append(big)
    print(f"ранг 1:  надпись у {stats[1][0]} из {stats[1][1]}")
    print(f"ранг ≥2: надпись у {stats[2][0]} из {stats[2][1]}")
    if tiles:
        cols = 6
        rows_n = (len(tiles) + cols - 1) // cols
        th = max(t.shape[0] for t in tiles)
        sheet = np.zeros((rows_n * th, cols * th, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            tile = cv2.resize(tile, (th, th))
            sheet[r * th:(r + 1) * th, c * th:(c + 1) * th] = tile
        cv2.imwrite("/tmp/opencode/rank_labels.png", sheet)
        print("монтаж: /tmp/opencode/rank_labels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
