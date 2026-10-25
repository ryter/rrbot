"""Генерация спрайтов юнитов и эталонных шаблонов.

Ключевые решения, которые делают детектор работоспособным:

1. **Шаблон содержит только тело юнита** (силуэт + глиф), без фона поля.
   Иначе совпадение ломается о разный фон (трава, река, мосты).
2. **У каждого типа юнита свой силуэт** (круг, шестиугольник, квадрат,
   треугольник). По форме контура тип определяется надёжнее, чем по
   маленькому глифу внутри — глиф слишком мелкий относительно площади.
3. **Цвет тела нейтрально-серый в шаблоне.** Шаблонное совпадение
   (TM_CCOEFF_NORMED) инвариантно к изменению яркости и контраста, поэтому
   один шаблон работает и для синих, и для красных юнитов. Сторона
   определяется отдельно — по оттенку насыщенных пикселей внутри диска.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Стороны в BGR. Синий — свои, красный — противник.
ALLY_BGR = (230, 90, 40)
ENEMY_BGR = (40, 40, 230)

# Фон поля (BGR) — низкая насыщенность, чтобы не путаться с юнитами.
GRASS_BGR = (96, 140, 100)
GRASS_ALT_BGR = (90, 132, 94)

TILE = 36  # размер холста спрайта (диаметр тела = 36 пикселей)
RADIUS = 17  # радиус тела юнита в канонических пикселях

GLYPH_COLOR = (245, 245, 245)
DARK = (28, 28, 28)
NEUTRAL_BODY = (170, 170, 170)

# Насыщенность тела юнита (по HSV) заметно выше фона поля.
UNIT_SATURATION_MIN = 120


# --------------------------------------------------------------- силуэты типов
def _body_mask(kind: str, size: int, radius: int) -> np.ndarray:
    """Бинарная маска тела юнита заданного типа."""
    mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    if kind == "golem":
        # Скруглённый квадрат.
        side = int(radius * 1.55)
        cv2.rectangle(mask, (c - side // 2, c - side // 2), (c + side // 2, c + side // 2), 255, -1)
    elif kind == "warden":
        # Ромб.
        pts = np.array([[c, c - radius], [c + radius, c], [c, c + radius], [c - radius, c]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    elif kind == "samurai":
        # Шестиугольник.
        pts = np.array(
            [
                [c, c - radius],
                [c + int(radius * 0.87), c - int(radius * 0.5)],
                [c + int(radius * 0.87), c + int(radius * 0.5)],
                [c, c + radius],
                [c - int(radius * 0.87), c + int(radius * 0.5)],
                [c - int(radius * 0.87), c - int(radius * 0.5)],
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
    else:  # knight, archer, bomber — круг
        cv2.circle(mask, (c, c), radius, 255, -1)
    return mask


# ------------------------------------------------------------------- глифы
def _glyph_knight(cv: np.ndarray) -> None:
    cv2.rectangle(cv, (14, 8), (21, 28), GLYPH_COLOR, -1)
    cv2.rectangle(cv, (10, 14), (25, 20), GLYPH_COLOR, -1)


def _glyph_archer(cv: np.ndarray) -> None:
    cv2.ellipse(cv, (18, 18), (8, 12), 0, -55, 55, GLYPH_COLOR, 2)
    cv2.line(cv, (18, 7), (18, 29), GLYPH_COLOR, 2)


def _glyph_golem(cv: np.ndarray) -> None:
    cv2.rectangle(cv, (10, 10), (25, 25), GLYPH_COLOR, -1)
    cv2.rectangle(cv, (15, 15), (20, 21), DARK, -1)


def _glyph_bomber(cv: np.ndarray) -> None:
    cv2.circle(cv, (17, 21), 7, GLYPH_COLOR, -1)
    cv2.line(cv, (19, 14), (24, 7), GLYPH_COLOR, 2)


def _glyph_samurai(cv: np.ndarray) -> None:
    cv2.line(cv, (9, 9), (26, 26), GLYPH_COLOR, 3)
    cv2.line(cv, (26, 9), (9, 26), GLYPH_COLOR, 3)


def _glyph_warden(cv: np.ndarray) -> None:
    pts = np.array([[18, 6], [28, 14], [18, 30], [8, 14]], dtype=np.int32)
    cv2.fillPoly(cv, [pts], GLYPH_COLOR)


GLYPHS: dict[str, Callable[[np.ndarray], None]] = {
    "knight": _glyph_knight,
    "archer": _glyph_archer,
    "golem": _glyph_golem,
    "bomber": _glyph_bomber,
    "samurai": _glyph_samurai,
    "warden": _glyph_warden,
}

# Сколько пикселей тела (без глифа) должно остаться для замера цвета.
SAMPLE_RING = (0.45, 0.85)


def glyph_tile(kind: str, size: int = TILE) -> np.ndarray:
    """Холст с глифом типа юнита (чёрный фон, светлый глиф)."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    GLYPHS.get(kind, _glyph_knight)(canvas)
    return canvas


def draw_unit(
    image: np.ndarray,
    kind: str,
    cx: float,
    cy: float,
    body_color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    """Нарисовать юнита на изображении (координаты в пикселях изображения)."""
    radius = max(3, int(round(RADIUS * scale)))
    size = max(6, int(round(TILE * scale)))
    x, y = int(round(cx)), int(round(cy))

    body = np.zeros((size, size, 3), dtype=np.uint8)
    mask = _body_mask(kind, size, radius)
    body[mask > 0] = body_color
    # Тёмный контур по границе силуэта.
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    body[contour > 0] = DARK

    tile = glyph_tile(kind, TILE)
    if size != TILE:
        interp = cv2.INTER_NEAREST if size < TILE else cv2.INTER_AREA
        tile = cv2.resize(tile, (size, size), interpolation=interp)
    gmask = (tile.any(axis=2)).astype(np.uint8) * 255
    body = np.where(gmask[:, :, None] > 0, tile, body)

    h, w = image.shape[:2]
    x0, y0 = x - size // 2, y - size // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    sw, sh = min(size - sx0, w - dx0), min(size - sy0, h - dy0)
    if sw > 0 and sh > 0:
        image[dy0 : dy0 + sh, dx0 : dx0 + sw] = body[sy0 : sy0 + sh, sx0 : sx0 + sw]


def unit_template(
    kind: str,
    size: int = TILE,
    body_color: tuple[int, int, int] = NEUTRAL_BODY,
) -> np.ndarray:
    """Шаблон юнита (только тело) в оттенках серого."""
    body = np.zeros((size, size, 3), dtype=np.uint8)
    radius = size // 2 - 1
    mask = _body_mask(kind, size, radius)
    body[mask > 0] = body_color
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    body[contour > 0] = DARK

    tile = glyph_tile(kind, size)
    gmask = (tile.any(axis=2)).astype(np.uint8) * 255
    body = np.where(gmask[:, :, None] > 0, tile, body)
    return cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)


def write_templates(out_dir: str | Path, kinds: list[str], size: int = TILE) -> list[Path]:
    """Записать серые шаблоны в каталог."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for kind in kinds:
        path = out / f"{kind}.png"
        cv2.imwrite(str(path), unit_template(kind, size))
        written.append(path)
    return written
