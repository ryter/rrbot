#!/usr/bin/env python3
"""Вырезать основания пешек из кадров и собрать монтаж.

Нужен, чтобы глазами разложить пешки по рангам: у юнита первого ранга
основание круглое, у слитого — другая фигура.

    .venv/bin/python tools/rank_crops.py assets/eval_real/frames [максимум]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def pawns_of(frame: Path) -> list[tuple[int, int, int, str]]:
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    pawns = []
    for side, x, y, r in re.findall(r"(своя|чужая) (\d+),(\d+) радиус (\d+)", out):
        pawns.append((int(x), int(y), int(r), side))
    return pawns


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/eval_real/frames")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    tile = 110
    tiles = []
    for frame in sorted(directory.glob("*.png")):
        image = cv2.imread(str(frame))
        if image is None:
            continue
        for x, y, r, side in pawns_of(frame):
            half = tile // 2
            x0, y0 = x - half, y - half
            x0 = max(0, min(x0, image.shape[1] - tile))
            y0 = max(0, min(y0, image.shape[0] - tile))
            crop = image[y0 : y0 + tile, x0 : x0 + tile].copy()
            cv2.drawMarker(crop, (x - x0, y - y0), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(crop, f"{side[:1]} r{r}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(crop)
            if len(tiles) >= limit:
                break
        if len(tiles) >= limit:
            break
    if not tiles:
        print("пешек нет")
        return 1
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, crop in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = crop
    out = Path("/tmp/opencode/rank_crops.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"пешек: {len(tiles)}, монтаж: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
