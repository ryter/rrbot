#!/usr/bin/env python3
"""Уровень карты в колоде: меняется ли после слияний?

На картах в колоде написано «L.1» — это уровень карты. Если игра при
слиянии повышает уровень карты, то подпись станет «L.2» — вот и готовый
признак ранга: у юнита ранга N подпись карты L.N.

Скрипт вырезает полосу колоды по кадрам боя и собирает монтаж, чтобы
увидеть, меняются ли подписи.

    .venv/bin/python tools/deck_level_probe.py <каталог кадров>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Полоса колоды в пикселях устройства (941x1030): карты с подписью «L.N».
DECK_RECT = (0, 850, 941, 1030)


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp")
    frames = sorted(directory.glob("b*.png"))
    if not frames:
        print("кадров нет")
        return 1
    x0, y0, x1, y1 = DECK_RECT
    tiles = []
    for frame in frames[::2]:
        image = cv2.imread(str(frame))
        if image is None:
            continue
        crop = image[y0:y1, x0:x1]
        crop = cv2.resize(crop, (470, 90), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, frame.stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1)
        tiles.append(crop)
        if len(tiles) >= 12:
            break
    sheet = np.vstack(tiles)
    out = Path("/tmp/opencode/deck_levels.png")
    cv2.imwrite(str(out), sheet)
    print(f"кадров: {len(tiles)}, монтаж: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
