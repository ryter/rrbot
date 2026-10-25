#!/usr/bin/env python3
"""Обучение распознавателя цифр по кадру HUD.

Показывает увеличенную область с числом и просит назвать цифры, после чего
сохраняет эталоны в ``assets/digits``.

Запуск:
    python tools/label_digits.py /tmp/opencode/m_07.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rrbot.digits import DigitReader  # noqa: E402
from rrbot.live import LiveGame  # noqa: E402

# Область с числом энергии в координатах холста.
ENERGY_REGION = (138, 682, 38, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение распознавателя цифр")
    parser.add_argument("frame", help="кадр с числом (PNG в координатах холста)")
    parser.add_argument("--region", default="132,682,52,16")
    parser.add_argument("--value", default="", help="что написано на экране (например 120)")
    args = parser.parse_args()

    canvas = cv2.imread(args.frame)
    if canvas is None:
        print("не открыть кадр", file=sys.stderr)
        return 1
    x, y, w, h = (int(v) for v in args.region.split(","))
    region = canvas[y : y + h, x : x + w]

    reader = DigitReader()
    glyphs = reader._glyphs(region)
    print(f"найдено глифов: {len(glyphs)}")
    if not glyphs:
        print("глифы не найдены — проверьте область", file=sys.stderr)
        return 1

    sheet = np.hstack(
        [cv2.resize(g, (g.shape[1] * 8, g.shape[0] * 8), interpolation=cv2.INTER_NEAREST) for g in glyphs]
    )
    preview = "/tmp/opencode/digit_glyphs.png"
    cv2.imwrite(preview, sheet)
    print(f"превью глифов: {preview}")

    if not args.value:
        print("укажите --value, чтобы сохранить эталоны")
        return 0
    value = args.value.strip()
    if len(value) != len(glyphs):
        print(f"в значении {len(value)} символов, а глифов {len(glyphs)}", file=sys.stderr)
        return 1
    for glyph, ch in zip(glyphs, value):
        path = reader.learn(glyph, ch)
        print(f"  сохранено {ch} -> {path}")
    print(f"известные цифры: {sorted(reader.known)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
