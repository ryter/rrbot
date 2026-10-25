#!/usr/bin/env python3
"""Сбор эталонов юнитов мобильного клиента из карточек колоды.

На экране итогов боя показан�� колода обоих игроков: в каждой карточке —
чистое изображение юнита (без травы, без круглого основания и без
ценника). Это лучший источник эталонов: спрайт на поле из-за соседей
перекрыт.

Карточки ищутся по рамкам: в ряду карточек рамки — это вертикальные
границы ярких прямоугольников. Внутри каждой рамки вырезается квадрат с
картинкой юнита (верхняя часть карточки, без полоски уровня).

Запуск:

    .venv/bin/python tools/collect_unit_cards.py --out assets/real_emulator \\
        /tmp/opencode/live3.png /tmp/opencode/fin2.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Ряды карточек на экране итогов (в пикселях кадра 941x1030).
ROWS = {
    "opponent": (135, 245),
    "player": (690, 795),
}


def card_bounds(image: np.ndarray, row: tuple[int, int]) -> list[tuple[int, int]]:
    """Границы карточек в ряду: (левый, правый) крайние x."""
    top, bottom = row
    band = image[top:bottom]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    density = edges.sum(axis=0)
    if density.size == 0:
        return []
    # Рамки карточек — вертикальные полосы края.
    strong = density > density.max() * 0.35
    groups: list[list[int]] = []
    for x, flag in enumerate(strong):
        if not flag:
            continue
        if groups and x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    # Края карточек: пары «левая-правая» с шириной больше 40 px.
    boxes: list[tuple[int, int]] = []
    for index in range(len(groups) - 1):
        left = groups[index][0]
        right = groups[index + 1][0]
        if 40 <= right - left <= 90:
            boxes.append((left, right))
    return boxes


def cut_art(image: np.ndarray, box: tuple[int, int]) -> np.ndarray | None:
    left, right = box
    width = right - left
    if width < 30:
        return None
    # Картинка юнита — верхняя часть карточки, квадрат со стороной ~0.8 ширины.
    side = int(width * 0.82)
    x0 = left + (width - side) // 2
    y0 = image.shape[0] * 0  # заполним ниже
    return (x0, side, width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор эталонов из карточек колоды")
    parser.add_argument("frames", nargs="+", help="кадры экрана итогов (PNG)")
    parser.add_argument("--out", default=str(ROOT / "assets" / "real_emulator"))
    parser.add_argument("--prefix", default="card")
    parser.add_argument("--preview", default="/tmp/opencode/cards.png")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    patches: list[np.ndarray] = []
    for path in args.frames:
        image = cv2.imread(path)
        if image is None:
            print(f"не прочитан: {path}", file=sys.stderr)
            continue
        for row_name, row in ROWS.items():
            top, bottom = row
            for index, (left, right) in enumerate(card_bounds(image, row)):
                width = right - left
                side = int(width * 0.8)
                x0 = left + (width - side) // 2
                y0 = top + int((bottom - top) * 0.06)
                if x0 < 0 or y0 < 0 or x0 + side > image.shape[1] or y0 + side > image.shape[0]:
                    continue
                patch = image[y0 : y0 + side, x0 : x0 + side].copy()
                # Внутри должен быть юнит: много насыщенных цветов.
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                colored = cv2.inRange(hsv, (0, 80, 60), (180, 255, 255))
                if float(colored.mean()) < 0.3:
                    continue
                patches.append(patch)

    if not patches:
        print("карточки не найдены", file=sys.stderr)
        return 1

    # Разные размеры сравнивать нельзя — приводим к общему квадрату.
    side_common = min(p.shape[0] for p in patches)
    resized = []
    for patch in patches:
        if patch.shape[0] < side_common:
            continue
        resized.append(cv2.resize(patch, (side_common, side_common), interpolation=cv2.INTER_AREA))
    unique: list[np.ndarray] = []
    for patch in resized:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if all(
            float(np.abs(gray.astype(np.int16) - cv2.cvtColor(u, cv2.COLOR_BGR2GRAY).astype(np.int16)).mean())
            > 18
            for u in unique
        ):
            unique.append(patch)

    for index, patch in enumerate(unique):
        cv2.imwrite(str(out / f"{args.prefix}_{index:02d}.png"), patch)
    print(f"сохранено карточек: {len(unique)} -> {out}")

    if unique:
        side = max(p.shape[0] for p in unique)
        cols = 5
        rows = (len(unique) + cols - 1) // cols
        sheet = np.zeros((rows * side, cols * side, 3), np.uint8)
        for index, patch in enumerate(unique):
            r, c = divmod(index, cols)
            sheet[r * side : (r + 1) * side, c * side : (c + 1) * side] = patch[:side, :side]
        cv2.imwrite(args.preview, sheet)
        print(f"превью: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
