#!/usr/bin/env python3
"""Диагностика: совпадает ли геометрия живого кадра с канонической.

Эталоны юнитов для мобильного клиента собраны в канонических кадрах
(`assets/eval_real/frames`, 400x522: верхняя половина арены, полоса волны,
нижняя половина). Канонический кадр собирается так (`tools/prepare_real_frames.py`):

* обе половины арены находятся по плиткам;
* каждая половина приводится к общему масштабу — по медианному радиусу
  оснований юнитов (REF_RADIUS = 31 px);
* верхняя половина кладётся в строки 0.., нижняя — в строки 282.., полоса
  волны между ними.

А C++-бот ищет арену иначе: всю арену целиком приводит к 400x522 одним
преобразованием. Геометрии отличаются: масштаб по ширине арены вместо
масштаба по юнитам, другая ширина полосы волны, другая высота половин.

Проверяем: (1) насколько радиус оснований совпадает с шириной половины
арены (если совпадает, масштаб можно брать по арене — без поиска кругов);
(2) насколько расходятся позиции юнитов при двух способах построения
канонического кадра.
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
    factor = PRF.REF_RADIUS / max(1.0, float(radii[len(radii) // 2]))
    board = np.zeros((BOARD_H, HALF_W, 3), dtype=np.uint8)
    for index, (x, y, w, h) in enumerate(arenas):
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
    return board, factor


def canonical_cpp(image, arena_rect):
    """Канонический кадр так, как делает C++-бот: вся арена целиком."""
    x, y, w, h = arena_rect
    half = image[y : y + h, x : x + w]
    return cv2.resize(half, (400, 522), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    frames = sorted(Path("/tmp/opencode/battle_frames").glob("*.png"))
    if not frames:
        print("нет кадров")
        return 1
    print(f"кадров: {len(frames)}\n")
    ratios = []
    shifts = []
    for path in frames[::5]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        arenas = PRF.find_arenas(image)
        if not arenas:
            continue
        units = PRF.find_units(image, arenas)
        if not units:
            continue
        radii = [u[2] for u in units]
        median_radius = sorted(radii)[len(radii) // 2]
        widths = [a[2] for a in arenas]
        # Радиус основания к ширине половины арены.
        ratio = median_radius / max(1.0, float(sum(widths) / len(widths)))
        ratios.append(ratio)

        # Позиции юнитов в обоих вариантах канонического кадра.
        board_py, factor = canonical_python(image, arenas, units)
        full = (min(a[0] for a in arenas), min(a[1] for a in arenas),
                max(a[0] + a[2] for a in arenas) - min(a[0] for a in arenas),
                max(a[1] + a[3] for a in arenas) - min(a[1] for a in arenas))
        board_cpp = canonical_cpp(image, full)
        # Ищем те же юниты на обоих кадрах: грубый сдвиг — по совпадению
        # шаблонов не делаем, сравниваем положение круглых оснований.
        for board, name in ((board_py, "python"), (board_cpp, "cpp")):
            found = PRF.find_units(board, PRF.find_arenas(board))
            shifts.append((name, len(found), len(units)))
    if ratios:
        ratios = np.array(ratios)
        print(f"радиус основания / ширина половины арены: {ratios.mean():.4f} "
              f"(разброс {ratios.std():.4f}, мин {ratios.min():.4f}, макс {ratios.max():.4f})")
    print("\nпо кадрам: (вариант, найдено юнитов в каноническом кадре, было в живом)")
    for name, found, total in shifts[:12]:
        print(f"  {name}: {found} из {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
