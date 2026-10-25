#!/usr/bin/env python3
"""Разметка цифр по кадрам с известными значениями.

Скрипт получает список кадров и соответствующие им прочитанные значения
(например, «60»), сопоставляет глифы с цифрами значения и сохраняет эталоны
для тех цифр, которых ещё не было.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from rrbot.digits import DigitReader  # noqa: E402
from rrbot.live import ENERGY_REGION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Разметка цифр по известным значениям")
    parser.add_argument("--frames", nargs="+", required=True, help="кадры холста")
    parser.add_argument("--values", nargs="+", required=True, help="значения энергии в кадрах")
    parser.add_argument("--skip", type=int, default=0, help="сколько первых кадров пропустить")
    args = parser.parse_args()

    if len(args.frames) != len(args.values):
        print("кадров и значений должно быть поровну", file=sys.stderr)
        return 1

    reader = DigitReader()
    added: dict[str, Path] = {}
    for path, value in zip(args.frames, args.values):
        if not value or value == "-":
            continue
        canvas = cv2.imread(path)
        if canvas is None:
            continue
        x, y, w, h = ENERGY_REGION
        region = canvas[y : y + h, x : x + w]
        glyphs = reader._glyphs(region)
        if len(glyphs) != len(value):
            print(f"{Path(path).name}: глифов {len(glyphs)}, в значении {len(value)} — пропуск")
            continue
        for glyph, ch in zip(glyphs, value):
            if not ch.isdigit():
                continue
            out = reader.learn(glyph, ch)
            added[ch] = out
            print(f"{Path(path).name}: цифра «{ch}» -> {out}")
    print(f"выучено цифр: {sorted(added)}; всего известно: {sorted(reader.known)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
