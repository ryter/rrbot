#!/usr/bin/env python3
"""Собрать регрессионный набор для чтения энергии.

Вырезает область числа энергии из боевых кадров и записывает рядом
файл с ожидаемыми значениями. Кадры полного размера в репозиторий не
кладут — много места, нужен только кусок с цифрами.

    .venv/bin/python tools/make_energy_fixtures.py \\
        --frames /tmp/opencode/frames2 --labels /tmp/opencode/labels.txt \\
        --out assets/test_energy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Область числа энергии в координатах холста мобильного клиента.
# Должна совпадать с Layout::energy в cpp/src/layout.cpp.
ENERGY = (318, 877, 61, 30)
CANVAS = (950, 1034)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор тестовых вырезов энергии")
    parser.add_argument("--frames", required=True, help="каталог с кадрами")
    parser.add_argument("--labels", required=True,
                        help="файл со строками «имя_кадра.png значение»")
    parser.add_argument("--out", default=str(ROOT / "assets" / "test_energy"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels: list[tuple[str, str]] = []
    for line in Path(args.labels).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"пропущена строка: {line}", file=sys.stderr)
            continue
        labels.append((parts[0], parts[1]))

    frames_dir = Path(args.frames)
    written = 0
    for name, value in labels:
        frame = frames_dir / name
        image = cv2.imread(str(frame))
        if image is None:
            print(f"нет кадра: {frame}", file=sys.stderr)
            continue
        # Тот же размер, что и холст у бота, и та же область: вырез
        # должен быть ровно тем, что бот видит в бою.
        canvas = cv2.resize(image, CANVAS, interpolation=cv2.INTER_AREA)
        x, y, w, h = ENERGY
        crop = canvas[y : y + h, x : x + w]
        target = out / (Path(name).stem + ".png")
        cv2.imwrite(str(target), crop)
        written += 1
    print(f"вырезов записано: {written} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
