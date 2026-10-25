#!/usr/bin/env python3
"""Сбор эталонов юнитов с настоящего боевого кадра.

Юнит в игре — фигурка на цветном круге-подставке. Для распознавания
типа нужна сама фигурка, а не круг: подставки отличаются цветом по
уровню, а фигурка — по типу юнита.

Скрипт находит цветные круги, снимает с них фигурки и убирает дубликаты.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rrbot.live import FIELD_QUAD, LiveGame  # noqa: E402

# Фигурка занимает верхнюю часть круга-подставки.
SPRITE = 26
SPRITE_DY = -6  # смещение вверх от центра круга


def find_disks(field: np.ndarray) -> list[tuple[int, int, int]]:
    """Найти фигурки юнитов: коричневые/оранжевые пятна на поле.

    Поле зелёное и тоже насыщенное, поэтому по насыщенности не отличить;
    отличаем по оттенку: фигурки юнитов — от тёплых (5..30), трава — 60..90.
    """
    hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 110, 90), (32, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    disks: list[tuple[int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60 or w < 8 or h < 8:
            continue
        disks.append((x + w // 2, y + h // 2, max(w, h) // 2))
    return disks


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор эталонов юнитов с боевого кадра")
    parser.add_argument("frames", nargs="+")
    parser.add_argument("--out", default=str(ROOT / "assets" / "real"))
    parser.add_argument("--preview", default="/tmp/opencode/sprites_new.png")
    args = parser.parse_args()

    game = LiveGame()
    patches: list[np.ndarray] = []

    for path in args.frames:
        canvas = cv2.imread(path)
        if canvas is None:
            continue
        field = game.field_image(canvas)
        for cx, cy, r in find_disks(field):
            if r < 5 or r > 22:
                continue
            x0 = cx - SPRITE // 2
            y0 = cy + SPRITE_DY - SPRITE // 2
            if x0 < 0 or y0 < 0 or x0 + SPRITE > field.shape[1] or y0 + SPRITE > field.shape[0]:
                continue
            patch = field[y0 : y0 + SPRITE, x0 : x0 + SPRITE]
            if patch.shape[:2] != (SPRITE, SPRITE):
                continue
            patches.append(patch)

    if not patches:
        print("подставки не найдены", file=sys.stderr)
        return 1

    unique: list[np.ndarray] = []
    for p in patches:
        # Патч должен содержать фигурку (тёплые тона), а не только подставку.
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
        warm = cv2.inRange(hsv, (5, 90, 70), (40, 255, 255))
        if float(warm.mean()) < 0.08:
            continue
        key = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        if all(
            float(np.abs(key.astype(np.int16) - cv2.cvtColor(u, cv2.COLOR_BGR2GRAY).astype(np.int16)).mean()) > 22
            for u in unique
        ):
            unique.append(p)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for i, p in enumerate(unique):
        cv2.imwrite(str(out / f"unit_{i:02d}.png"), cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))
    print(f"эталонов сохранено: {len(unique)} (из {len(patches)} находок)")

    cols = 6
    rows = (len(unique) + cols - 1) // cols
    sheet = np.zeros((rows * 30, cols * 30, 3), dtype=np.uint8)
    for i, p in enumerate(unique):
        r, c = divmod(i, cols)
        sheet[r * 30 + 2 : r * 30 + 28, c * 30 + 2 : c * 30 + 28] = p
    cv2.imwrite(args.preview, sheet)
    print(f"превью: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
