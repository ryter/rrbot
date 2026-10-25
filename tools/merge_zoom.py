#!/usr/bin/env python3
"""Крупный план пешки до и после успешного слияния — сравнить глазами.

По логу бота известно, где было слияние; на кадре «после» там стоит
результат (ранг +1). Крупный план показывает, чем он отличается.

    .venv/bin/python tools/merge_zoom.py <лог> <каталог кадров> [номер]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
HALF = 55


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


def zoom(image, x, y):
    h, w = image.shape[:2]
    x0 = max(0, min(x - HALF, w - 2 * HALF))
    y0 = max(0, min(y - HALF, h - 2 * HALF))
    crop = image[y0 : y0 + 2 * HALF, x0 : x0 + 2 * HALF]
    return cv2.resize(crop, (2 * HALF * 3, 2 * HALF * 3), interpolation=cv2.INTER_CUBIC)


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    want = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    events = [e for e in parse_log(log) if e[6]]
    rows = []
    for index, (r, s, x1, y1, x2, y2, _) in enumerate(events):
        if want >= 0 and index != want:
            continue
        before = directory / f"b{r - 1}_{s}.png"
        after = directory / f"b{r - 1}_{s + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        pb = pawns_of(before)
        pa = pawns_of(after)
        target = nearest(pb, x2, y2)
        result = nearest(pa, x2, y2)
        if not target or not result:
            continue
        zb = zoom(ib, *target[:2])
        za = zoom(ia, *result[:2])
        row = np.hstack([zb, np.full((zb.shape[0], 8, 3), 255, np.uint8), za])
        cv2.putText(row, f"#{index} r{target[2]}->r{result[2]}", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rows.append(row)
    if not rows:
        print("кадров нет")
        return 1
    sheet = np.vstack(rows)
    out = Path("/tmp/opencode/merge_zoom.png")
    cv2.imwrite(str(out), sheet)
    print(f"событий: {len(rows)}, файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
