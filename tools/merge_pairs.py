#!/usr/bin/env python3
"""Пары пешек для слияния: ищем на своей половине одинаковых соседей.

Идея: не нужен эталон юнита. Сами пешки сравниваются попарно: у одинаковых
юнитов почти совпадает круглый портрет, у разных — нет. Позиции пешек берём
по круглым основаниям (Hough), они находятся надёжно.

    .venv/bin/python tools/merge_pairs.py /tmp/opencode/battle_loop/b7_20.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/lennistyle/rushroyale-lab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import prepare_real_frames as PRF  # noqa: E402

HALF_W = 400
HALF_H = 240
GAP = 42
BOARD_H = HALF_H * 2 + GAP
OUR_TOP = HALF_H + GAP
REF_RADIUS = PRF.REF_RADIUS


def canonical(image, arenas, units):
    radii = sorted(u[2] for u in units)
    factor = REF_RADIUS / max(1.0, float(radii[len(radii) // 2])) if radii else 1.0
    board = np.zeros((BOARD_H, HALF_W, 3), dtype=np.uint8)
    offsets = []
    for index, (x, y, w, h) in enumerate(arenas):
        half = image[y : y + h, x : x + w]
        resized = cv2.resize(half, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                             interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
        top = 0 if index == 0 else OUR_TOP
        left = max(0, (HALF_W - resized.shape[1]) // 2)
        src_x, src_y = max(0, -left), max(0, -top)
        dst_x, dst_y = max(0, left), max(0, top)
        width = min(HALF_W - dst_x, resized.shape[1] - src_x)
        height = min(BOARD_H - dst_y, resized.shape[0] - src_y)
        if width > 0 and height > 0:
            board[dst_y : dst_y + height, dst_x : dst_x + width] = \
                resized[src_y : src_y + height, src_x : src_x + width]
        offsets.append((left, top, x, y, w, h))
    return board, factor, offsets


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/opencode/battle_loop/b7_20.png")
    image = cv2.imread(str(path))
    if image is None:
        print("кадр не прочитан")
        return 1
    arenas = sorted(PRF.find_arenas(image), key=lambda a: a[1])
    units = PRF.find_units(image, arenas)
    if len(arenas) < 2 or not units:
        print(f"арены {arenas}, оснований {len(units)}")
        return 1
    board, factor, offsets = canonical(image, arenas, units)

    def to_board(px, py):
        for index, (left, top, x, y, w, h) in enumerate(offsets):
            if x - 12 <= px <= x + w + 12 and y - 12 <= py <= y + h + 12:
                return int(round((px - x) * factor + left)), int(round((py - y) * factor + top))
        return None

    ours = []
    for cx, cy, radius in units:
        pos = to_board(cx, cy)
        if pos is None:
            continue
        bx, by = pos
        if by < OUR_TOP + 20:      # только своя половина
            continue
        ours.append((bx, by))

    print(f"{path.name}: своих пешек {len(ours)}: {ours}")

    # Портреты: круг радиусом REF_RADIUS вокруг основания.
    patch = int(REF_RADIUS * 2)
    patches = []
    for bx, by in ours:
        x0, y0 = bx - patch // 2, by - patch // 2
        if x0 < 0 or y0 < 0 or x0 + patch > board.shape[1] or y0 + patch > board.shape[0]:
            patches.append(None)
            continue
        patches.append(board[y0 : y0 + patch, x0 : x0 + patch])
    print("сравнение портретов (NCC):")
    for i in range(len(ours)):
        row = []
        for j in range(len(ours)):
            if i == j or patches[i] is None or patches[j] is None:
                row.append("  --  ")
                continue
            score = cv2.matchTemplate(patches[i], patches[j], cv2.TM_CCOEFF_NORMED)[0][0]
            row.append(f"{score:6.3f}")
        print(f"  {i} {ours[i]}: " + " ".join(row))

    # Соседние клетки: шаг сетки ~78 пикселей.
    step = 78
    print("кандидаты на слияние (соседи по клетке):")
    for i in range(len(ours)):
        for j in range(i + 1, len(ours)):
            dx = abs(ours[i][0] - ours[j][0])
            dy = abs(ours[i][1] - ours[j][1])
            if dx > step * 1.4 or dy > step * 1.4:
                continue
            if patches[i] is None or patches[j] is None:
                continue
            score = cv2.matchTemplate(patches[i], patches[j], cv2.TM_CCOEFF_NORMED)[0][0]
            print(f"  {ours[i]} <-> {ours[j]}: расстояние ({dx},{dy}), NCC {score:.3f}"
                  f"{'  ПАРА' if score > 0.8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
