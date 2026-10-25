#!/usr/bin/env python3
"""Связь формы основания с рангом: проверяем на парах, которые игра
отвергла.

Если пара отвергнута из-за разного ранга, то у одного юнита основание
круглое (ранг 1), а у другого — нет (ранг 2). Соберём статистику по
удачным и неудачным попыткам слияния.

    .venv/bin/python tools/rank_shape_pairs.py <каталог_кадров> <лог>
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pedestal_shape import shape_of  # noqa: E402

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def pawns_of(frame: Path) -> list[tuple[int, int]]:
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y)) for x, y, _ in re.findall(r"  своя (\d+),(\d+) радикс", out)]


def pawns_of(frame: Path) -> list[tuple[int, int]]:
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y)) for x, y, _ in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out)]


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop")
    log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/battle_loop.log")
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
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\)", line)
        if m:
            last = (round_no, step, int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)))
        if "слияние вышло" in line and last:
            attempts.append((last, True))
            last = None
        elif "слияние не вышло" in line and last:
            attempts.append((last, False))
            last = None
    stats = {True: Counter(), False: Counter()}
    for (r, s, x1, y1, x2, y2), ok in attempts:
        frame = directory / f"b{r - 1}_{s}.png"
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        pawns = pawns_of(frame)
        def find(x, y):
            best = None
            best_d = 40
            for px, py in pawns:
                d = max(abs(px - x), abs(py - y))
                if d < best_d:
                    best_d = d
                    best = (px, py)
            return best
        p1, p2 = find(x1, y1), find(x2, y2)
        if not p1 or not p2:
            continue
        s1 = shape_of(image, *p1)
        s2 = shape_of(image, *p2)
        if not s1 or not s2:
            continue
        # Ранг 1 — основание круглое (заполненность ~0.9), ранг 2 и выше —
        # другая фигура (заполненность меньше).
        r1 = s1["fill"] >= 0.88
        r2 = s2["fill"] >= 0.88
        key = ("круг+круг" if (r1 and r2) else
               ("круг+не-круг" if (r1 or r2) else "не-круг+не-круг"))
        stats[ok][key] += 1
    print("успешные:", dict(stats[True]))
    print("неудачные:", dict(stats[False]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
