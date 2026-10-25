#!/usr/bin/env python3
"""Определение геометрии игрового поля на снимке экрана.

Анализирует кадр боя и находит:
  * границы игрового холста (внутри деревянной рамки телефона);
  * полосу реки — по ней определяется верх и низ поля;
  * полосу маны и карточки юнитов в HUD.

Результат печатается в формате, пригодном для ``config/default.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def find_canvas(win: np.ndarray) -> tuple[int, int, int, int]:
    """Границы игрового холста: ищем тёмно-коричневую рамку телефона."""
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    # Дерево/рамка: тёмный, насыщенный красно-коричневый.
    mask = cv2.inRange(hsv, (5, 60, 40), (25, 255, 190))
    cols = mask.sum(axis=0)
    rows = mask.sum(axis=1)
    # Рамка — вертикальные полосы с большим количеством тёмных пикселей.
    col_candidates = np.flatnonzero(cols > win.shape[0] * 0.5)
    row_candidates = np.flatnonzero(rows > win.shape[1] * 0.5)
    if len(col_candidates) < 2 or len(row_candidates) < 2:
        return 0, 0, win.shape[1], win.shape[0]
    left = int(col_candidates[0])
    right = int(col_candidates[-1])
    top = int(row_candidates[0])
    bottom = int(row_candidates[-1])
    return left, top, right, bottom


def find_river(canvas: np.ndarray) -> tuple[int, int] | None:
    """Найти реку: горизонтальная полоса бирюзового цвета."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    # Бирюза/циан: оттенок ~80-95 в шкале OpenCV.
    mask = cv2.inRange(hsv, (75, 60, 40), (100, 255, 255))
    rows = mask.sum(axis=1)
    width = canvas.shape[1]
    strong = np.flatnonzero(rows > width * 0.55)
    if len(strong) == 0:
        return None
    return int(strong[0]), int(strong[-1])


def find_mana_bar(canvas: np.ndarray) -> tuple[int, int, int, int] | None:
    """Найти полосу маны: насыщенная жёлто-голубая заливка в HUD."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 120, 120), (35, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 20 and 4 <= h <= 40 and area > 100:
            if best is None or area > best[4]:
                best = (int(x), int(y), int(w), int(h), int(area))
    if best is None:
        return None
    x, y, w, h, _ = best
    return x, y, w, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Геометрия игрового поля")
    parser.add_argument("image", help="снимок экрана (png)")
    parser.add_argument("--window", default="967,38,941,1030", help="x,y,w,h окна Chrome")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("не удалось открыть изображение", file=sys.stderr)
        return 1
    x, y, w, h = (int(v) for v in args.window.split(","))
    win = img[y : y + h, x : x + w]
    print(f"окно: {win.shape[1]}x{win.shape[0]}")

    left, top, right, bottom = find_canvas(win)
    print(f"рамка/холст (в окне): x {left}..{right}, y {top}..{bottom}")

    canvas = win[top:bottom, left:right]
    print(f"холст: {canvas.shape[1]}x{canvas.shape[0]}")

    river = find_river(canvas)
    if river:
        print(f"река в холсте: y {river[0]}..{river[1]}")
    mana = find_mana_bar(canvas)
    if mana:
        print(f"полоса маны (в холсте): {mana}")

    result = {
        "window": [x, y, w, h],
        "canvas_in_window": [left, top, right, bottom],
        "river_in_canvas": list(river) if river else None,
        "mana_in_canvas": list(mana) if mana else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
