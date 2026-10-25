#!/usr/bin/env python3
"""Геометрия поля для слияния: где стоят пешки и как они выглядят.

Показывает по боевому кадру:
* половины арены и канонический кадр (400x522);
* основания юнитов (круги) в координатах кадра и канонического поля;
* сетку поля 3x5 на своей половине и найденные пешки.

Запуск:
    .venv/bin/python tools/merge_geometry.py /tmp/opencode/battle_loop/b7_20.png
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


def canonical_python(image, arenas, units):
    """Канонический кадр так, как его делает prepare_real_frames.py."""
    radii = sorted(u[2] for u in units)
    factor = PRF.REF_RADIUS / max(1.0, float(radii[len(radii) // 2])) if radii else 1.0
    board = np.zeros((BOARD_H, HALF_W, 3), dtype=np.uint8)
    offsets = []
    for index, (x, y, w, h) in enumerate(arenas):
        half = image[y : y + h, x : y + h * 0 + h]
        half = image[y : y + h, x : x + w]
        resized = cv2.resize(half, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                             interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
        top = 0 if index == 0 else HALF_H + GAP
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
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/tmp/opencode/battle_loop/b7_20.png")
    image = cv2.imread(str(path))
    if image is None:
        print("кадр не прочитан:", path)
        return 1
    arenas = PRF.find_arenas(image)
    units = PRF.find_units(image, arenas)
    print(f"{path.name}: кадр {image.shape[1]}x{image.shape[0]}, арены {arenas}, "
          f"оснований {len(units)}")
    if not arenas or not units:
        print("арена или основания не найдены")
        return 1
    board, factor, offsets = canonical_python(image, arenas, units)
    print(f"масштаб {factor:.3f}, половины (left,top,x,y,w,h): {offsets}")

    def to_board(px, py):
        for index, (left, top, x, y, w, h) in enumerate(offsets):
            if x - 12 <= px <= x + w + 12 and y - 12 <= py <= y + h + 12:
                return (int(round((px - x) * factor + left)),
                        int(round((py - y) * factor + top)), index)
        return None

    preview = board.copy()
    print("основания (кадр -> канон, половина):")
    for cx, cy, radius in units:
        pos = to_board(cx, cy)
        if pos is None:
            print(f"  ({cx},{cy}) r={radius}: вне арены")
            continue
        bx, by, index = pos
        side = "своя" if index == 1 else "чужая"
        print(f"  ({cx},{cy}) r={radius} -> ({bx},{by}) {side}")
        cv2.circle(preview, (bx, by), max(3, int(radius * factor)), (0, 0, 255), 1)
        cv2.putText(preview, str((bx // 40, by // 40)), (bx - 8, by), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)

    out = Path("/tmp/opencode/canonical_board.png")
    cv2.imwrite(str(out), cv2.resize(preview, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST))
    print("канонический кадр с основаниями:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
