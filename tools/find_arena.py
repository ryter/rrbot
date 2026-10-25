#!/usr/bin/env python3
"""Определение геометрии арены по боевому кадру.

Ищет зелёное игровое поле (сетка плиток) и полосу маны, чтобы задать
калибровку для реальной игры.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def green_field(canvas: np.ndarray) -> np.ndarray:
    """Маска зелёных плиток поля."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 60, 90), (95, 255, 255))


def main() -> int:
    parser = argparse.ArgumentParser(description="Геометрия арены")
    parser.add_argument("frame")
    args = parser.parse_args()

    canvas = cv2.imread(args.frame)
    if canvas is None:
        print("не открыть", file=sys.stderr)
        return 1
    print(f"кадр: {canvas.shape[1]}x{canvas.shape[0]}")

    mask = green_field(canvas)
    # Строками считаем участки, где зелёного много.
    rows = mask.sum(axis=1)
    strong_rows = np.flatnonzero(rows > canvas.shape[1] * 0.35)
    if len(strong_rows) == 0:
        print("поле не найдено")
        return 1
    # Разбиваем на непрерывные полосы (две арены).
    groups: list[list[int]] = [[int(strong_rows[0])]]
    for r in strong_rows[1:]:
        if r - groups[-1][-1] <= 4:
            groups[-1].append(int(r))
        else:
            groups.append([int(r)])
    print("полосы поля (y):")
    for g in groups:
        band = mask[g[0] : g[-1] + 1]
        cols = np.flatnonzero(band.sum(axis=0) > (g[-1] - g[0]) * 0.35)
        if len(cols):
            print(f"  y {g[0]}..{g[-1]}  x {cols[0]}..{cols[-1]}")

    # Полоса маны: ищем зелёную заливку в нижней трети.
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    strip = hsv[640:720, 60:200]
    mana = cv2.inRange(strip, (35, 90, 60), (85, 255, 255))
    if mana.sum():
        cols = np.flatnonzero(mana.any(axis=0))
        print(f"полоса маны: x {60 + cols[0]}..{60 + cols[-1]} (в зоне 640..720)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
