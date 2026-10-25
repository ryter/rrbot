#!/usr/bin/env python3
"""Покажем пешки из неудачных попыток слияния: что игра отвергает.

    .venv/bin/python tools/failed_pairs.py <каталог_кадров> <лог> [сколько]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def main() -> int:
    directory = Path(sys.argv[1])
    log = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lines = log.read_text().split("\n")
    round_no = 1
    step = 0
    attempts = []
    last = None
    for line in lines:
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\), похожесть ([\d.]+)", line)
        if m:
            last = (round_no, step, int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)), float(m.group(5)))
        if "слияние вышло" in line and last:
            attempts.append((last, True))
            last = None
        elif "слияние не вышло" in line and last:
            attempts.append((last, False))
            last = None
    bad = [a for a, ok in attempts if not ok]
    print(f"неудачных попыток: {len(bad)}")
    tiles = []
    for (r, s, x1, y1, x2, y2, score) in bad[:limit]:
        frame = directory / f"b{r - 1}_{s}.png"
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        for (x, y) in ((x1, y1), (x2, y2)):
            # Крупно: тело юнита ( оно выше основания) вместе с основанием —
            # так видно знак ранга, если он есть.
            crop = image[max(0, y - 78):y + 42, max(0, x - 45):x + 45]
            if crop.shape[0] != 120 or crop.shape[1] != 90:
                continue
            crop = cv2.resize(crop, (135, 180), interpolation=cv2.INTER_LANCZOS4)
            cv2.putText(crop, f"{score:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255), 1)
            tiles.append(crop)
    if not tiles:
        print("кадров нет")
        return 1
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    out = Path("/tmp/opencode/failed_pairs.png")
    cv2.imwrite(str(out), np.vstack(rows))
    print("монтаж:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
