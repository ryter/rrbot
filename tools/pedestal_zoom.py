#!/usr/bin/env python3
"""Крупный план оснований пешек: кроп 60×60 вокруг центра, увеличение ×4.

Так видно, что именно детектор зовёт «основанием» — круг, ромб, овал.

    .venv/bin/python tools/pedestal_zoom.py <кадр> [максимум пешек]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
ZOOM = 4
CROP = 60


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    pawns = []
    for side, x, y, r, round_ in re.findall(
            r"(своя|чужая) (\d+),(\d+) радиус (\d+) (круглое|НЕ круглое)", out):
        pawns.append((int(x), int(y), int(r), side == "своя", round_ == "круглое"))
    return pawns


def main() -> int:
    frame = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    image = cv2.imread(str(frame))
    if image is None:
        print("кадр не прочитан")
        return 1
    tiles = []
    for x, y, r, ours, round_ in pawns_of(frame):
        half = CROP // 2
        x0 = max(0, min(x - half, image.shape[1] - CROP))
        y0 = max(0, min(y - half, image.shape[0] - CROP))
        crop = image[y0 : y0 + CROP, x0 : x0 + CROP]
        big = cv2.resize(crop, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(big, ((x - x0) * ZOOM, (y - y0) * ZOOM), (0, 0, 255),
                       cv2.MARKER_CROSS, 14, 1)
        cv2.putText(big, f"{'С' if ours else 'Ч'} r{r} {'К' if round_ else 'Н'}",
                    (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(big)
        if len(tiles) >= limit:
            break
    if not tiles:
        print("пешек нет")
        return 1
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].shape[0]
    sheet = np.zeros((rows * th, cols * th, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : (r + 1) * th, c * th : (c + 1) * th] = t
    out = Path("/tmp/opencode/pedestal_zoom.png")
    cv2.imwrite(str(out), sheet)
    print(f"пешек: {len(tiles)}, файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
