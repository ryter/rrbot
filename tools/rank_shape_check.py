#!/usr/bin/env python3
"""Проверка признака ранга по живым кадрам: у юнита ранга 1 основание
круглое, у слитого (ранга 2) — «капля».

Пешки берутся из проверочной программы (cpp/build/merge_probe), форма
основания считается здесь.

    .venv/bin/python tools/rank_shape_check.py /tmp/opencode/battle_loop
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pedestal_shape import shape_of  # noqa: E402

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def pawns_of(frame: Path) -> list[tuple[int, int]]:
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y)) for x, y, _ in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out)]


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop")
    frames = sorted(directory.glob("*.png"))[::3]
    stretches = []
    fills = []
    for frame in frames:
        image = cv2.imread(str(frame))
        if image is None:
            continue
        for x, y in pawns_of(frame):
            s = shape_of(image, x, y)
            if not s:
                continue
            stretches.append(s["stretch"])
            fills.append(s["fill"])
    if not stretches:
        print("данных нет")
        return 1
    stretches = np.array(stretches)
    fills = np.array(fills)
    print(f"пешек: {len(stretches)}")
    print(f"вытянутость: min {stretches.min():.2f}  медиана {np.median(stretches):.2f}  "
          f"max {stretches.max():.2f}")
    print(f"заполненность: min {fills.min():.2f}  медиана {np.median(fills):.2f}  "
          f"max {fills.max():.2f}")
    print("гистограмма вытянутости:")
    for lo in np.arange(0.9, 1.8, 0.05):
        n = int(((stretches >= lo) & (stretches < lo + 0.05)).sum())
        if n:
            print(f"  {lo:.2f}-{lo + 0.05:.2f}: {'#' * min(n, 60)} ({n})")
    print("гистограмма заполненности:")
    for lo in np.arange(0.5, 1.05, 0.05):
        n = int(((fills >= lo) & (fills < lo + 0.05)).sum())
        if n:
            print(f"  {lo:.2f}-{lo + 0.05:.2f}: {'#' * min(n, 60)} ({n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
