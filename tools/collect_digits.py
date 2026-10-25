#!/usr/bin/env python3
"""Сбор неизвестных цифр из игровых кадров.

Ищет в кадрах глифы, которые распознаватель не опознал, и собирает их
в контактный лист: цифры подписать вручную командой
``tools/label_digits.py`` или через ``--value``.
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
from rrbot.live import ENERGY_REGION, LiveGame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор неизвестных цифр")
    parser.add_argument("frames", nargs="+", help="кадры холста (PNG)")
    parser.add_argument("--out", default="/tmp/opencode/digit_samples.png")
    args = parser.parse_args()

    reader = DigitReader()
    game = LiveGame()
    samples: list[np.ndarray] = []

    for path in args.frames:
        canvas = cv2.imread(path)
        if canvas is None:
            continue
        x, y, w, h = ENERGY_REGION
        region = canvas[y : y + h, x : x + w]
        for glyph in reader._glyphs(region):
            name, score = reader._match(glyph)
            if name == "?":
                samples.append(glyph)

    if not samples:
        print("неизвестных цифр не найдено")
        return 0

    sheet = np.zeros((len(samples) * 20, 120, 3), dtype=np.uint8)
    for i, g in enumerate(samples):
        big = cv2.resize(g, (16, 18), interpolation=cv2.INTER_NEAREST)
        sheet[i * 20 + 1 : i * 20 + 19, 4 : 20] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(args.out, sheet)
    print(f"неизвестных глифов: {len(samples)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
