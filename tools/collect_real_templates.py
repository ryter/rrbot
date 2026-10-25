#!/usr/bin/env python3
"""Сбор настоящих шаблонов юнитов из боевых кадров.

Берёт область поля, находит в ней спрайты юнитов (округлые объекты с
насыщенными цветами) и сохраняет их как эталоны для детектора.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rrbot.live import LiveGame  # noqa: E402

SPRITE_SIZE = 34


def find_sprites(field: np.ndarray, min_area: int = 120) -> list[tuple[int, int, int, int]]:
    """Найти спрайты юнитов в кадре поля.

    Юниты в Rush Royale стоят на круглых основаниях, поэтому отбираем
    объекты, у которых ширина и высота близки, а заполнение bounding box
    высокое. Так из выборки уходят стрелки и иконки интерфейса.
    """
    hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    # Спрайты юнитов яркие и насыщенные; трава — зелёная и менее насыщенная.
    mask = cv2.inRange(hsv, (0, 130, 110), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    found: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 10 or h < 10:
            continue
        if w > 44 or h > 44:
            continue
        # Приблизительно круглый объект, заполняющий свой bounding box.
        if abs(w - h) > 0.3 * max(w, h):
            continue
        if area / float(w * h) < 0.55:
            continue
        found.append((int(x), int(y), int(w), int(h)))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор настоящих шаблонов юнитов")
    parser.add_argument("frames", nargs="+")
    parser.add_argument("--out", default=str(ROOT / "assets" / "real"))
    parser.add_argument("--preview", default="/tmp/opencode/real_sprites.png")
    args = parser.parse_args()

    game = LiveGame()
    patches: list[np.ndarray] = []
    for path in args.frames:
        canvas = cv2.imread(path)
        if canvas is None:
            continue
        field = game.field_image(canvas)
        for x, y, w, h in find_sprites(field):
            cx, cy = x + w // 2, y + h // 2
            x0 = max(0, cx - SPRITE_SIZE // 2)
            y0 = max(0, cy - SPRITE_SIZE // 2)
            crop = field[y0 : y0 + SPRITE_SIZE, x0 : x0 + SPRITE_SIZE]
            if crop.shape[0] != SPRITE_SIZE or crop.shape[1] != SPRITE_SIZE:
                continue
            patches.append(crop)

    if not patches:
        print("спрайты не найдены", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches):
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(out / f"real_{i:02d}.png"), gray)
    print(f"сохранено {len(patches)} шаблонов в {out}")

    cols = 6
    rows = (len(patches) + cols - 1) // cols
    sheet = np.zeros((rows * 40, cols * 40, 3), dtype=np.uint8)
    for i, p in enumerate(patches):
        r, c = divmod(i, cols)
        sheet[r * 40 + 3 : r * 40 + 37, c * 40 + 3 : c * 40 + 37] = p
    cv2.imwrite(args.preview, sheet)
    print(f"превью: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
