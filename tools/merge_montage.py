#!/usr/bin/env python3
"""Монтаж своей половины поля по кадрам боя: видно, что происходит с пешками.

Своя половина ищется по плиткам (как в боте), поэтому кадр кадрируется
правильно даже когда камера ходит.

    .venv/bin/python tools/merge_montage.py /tmp/opencode/battle_loop b0 0 8
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


def own_half(image):
    """Прямоугник своей половины арены (нижняя)."""
    arenas = PRF.find_arenas(image)
    if len(arenas) < 2:
        return None
    # Нижняя половина — та, что ниже по кадру.
    arenas = sorted(arenas, key=lambda a: a[1])
    x, y, w, h = arenas[-1]
    return x, y, w, h


def main() -> int:
    directory = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else "b0"
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    last = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out = Path("/tmp/opencode/montage.png")
    tiles = []
    for step in range(first, last + 1):
        path = directory / f"{prefix}_{step}.png"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        rect = own_half(image)
        if rect is None:
            print(f"{path.name}: арена не найдена")
            continue
        x, y, w, h = rect
        crop = image[y : y + h, x : x + w]
        crop = cv2.resize(crop, (420, 240))
        cv2.putText(crop, str(step), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(crop)
    if not tiles:
        print("кадров нет")
        return 1
    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i : i + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out), np.vstack(rows))
    print("монтаж:", out, len(tiles), "кадров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
