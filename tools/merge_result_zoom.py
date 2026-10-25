#!/usr/bin/env python3
"""Крупный план результата слияния (что стоит на месте цели после удачи).

На пешках в игре встречаются надписи вроде «1/6», «2/6» — надо понять,
что это. Если первое число — ранг, то это и есть готовый признак.

    .venv/bin/python tools/merge_result_zoom.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
HALF = 45


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y), int(r)) for x, y, r in
            re.findall(r"своя (\d+),(\d+) радиус (\d+)", out)]


def nearest(pawns, x, y, limit=50):
    best, best_d = None, limit
    for p in pawns:
        d = max(abs(p[0] - x), abs(p[1] - y))
        if d < best_d:
            best_d, best = d, p
    return best


def parse_log(log: Path):
    events, round_no, step, pending = [], 1, 0, None
    for line in log.read_text(errors="ignore").split("\n"):
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\)", line)
        if m:
            pending = (round_no, step, int(m.group(1)), int(m.group(2)),
                       int(m.group(3)), int(m.group(4)))
        if pending and "слияние вышло" in line:
            events.append((*pending, True))
            pending = None
        elif pending and "слияние не вышло" in line:
            events.append((*pending, False))
            pending = None
    return events


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    tiles = []
    for index, (r, s, x1, y1, x2, y2, ok) in enumerate(parse_log(log)):
        if not ok:
            continue
        after = directory / f"b{r - 1}_{s + 1}.png"
        if not after.exists():
            continue
        image = cv2.imread(str(after))
        if image is None:
            continue
        pawns = pawns_of(after)
        res = nearest(pawns, x2, y2)
        if not res:
            continue
        x, y = res[0], res[1]
        x0 = max(0, min(x - HALF, image.shape[1] - 2 * HALF))
        y0 = max(0, min(y - HALF, image.shape[0] - 2 * HALF))
        crop = image[y0 : y0 + 2 * HALF, x0 : x0 + 2 * HALF]
        big = cv2.resize(crop, (2 * HALF * 4, 2 * HALF * 4), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, f"#{index} r{res[2]}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        tiles.append(big)
    if not tiles:
        print("кадров нет")
        return 1
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].shape[0]
    sheet = np.zeros((rows * th, cols * th, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : (r + 1) * th, c * th : (c + 1) * th] = t
    out = Path("/tmp/opencode/merge_result_zoom.png")
    cv2.imwrite(str(out), sheet)
    print(f"кадров: {len(tiles)}, файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
