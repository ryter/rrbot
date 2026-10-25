#!/usr/bin/env python3
"""Монтаж «до/после» по слияниям из лога бота.

Идея: успешное слияние двух юнитов даёт юнита на ранг выше. Значит, на
кадре «после» юнит на месте цели — это заведомо не ранг 1 (если сливали
первые ранги). Сравнив форму основания до и после, можно понять, чем
ранг 2 отличается от ранга 1.

    .venv/bin/python tools/merge_before_after.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

TILE = 120


def parse(log: Path):
    """Список (раунд, шаг, src, dst, ok) по логу."""
    events = []
    round_no = 1
    step = 0
    pending = None
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


def crop(image, x, y, half=TILE // 2):
    h, w = image.shape[:2]
    x0 = max(0, min(x - half, w - TILE))
    y0 = max(0, min(y - half, h - TILE))
    return image[y0 : y0 + TILE, x0 : x0 + TILE]


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/battle_loop")
    events = parse(log)
    if not events:
        print("слияний в логе нет")
        return 1
    rows = []
    for r, s, x1, y1, x2, y2, ok in events[:12]:
        before = directory / f"b{r - 1}_{s}.png"
        after = directory / f"b{r - 1}_{s + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        cb = crop(ib, x2, y2)
        ca = crop(ia, x2, y2)
        sb = crop(ib, x1, y1)
        pair = np.hstack([sb, cb, ca])
        cv2.putText(pair, f"{'ok' if ok else 'fail'} s{s}", (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(pair)
    if not rows:
        print("нужных кадров нет")
        return 1
    width = rows[0].shape[1]
    sheet = np.zeros((len(rows) * TILE, width, 3), np.uint8)
    for i, row in enumerate(rows):
        sheet[i * TILE : (i + 1) * TILE] = row
    out = Path("/tmp/opencode/merge_before_after.png")
    cv2.imwrite(str(out), sheet)
    print(f"событий: {len(events)}, в монтаже: {len(rows)}, файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
