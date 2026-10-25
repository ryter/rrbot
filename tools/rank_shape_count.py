#!/usr/bin/env python3
"""Сколько на поле юнитов с не-круглым основанием?

Проверка гипотезы «форма основания = ранг»:

  * каждое успешное слияние рождает одного юнита старшего ранга;
  * если ранг кодируется формой (круг — ранг 1, не-круг — ранг ≥2), то
    число пешек с не-круглым основанием должно расти ровно на число
    успешных слияний, а с кружком — падать на число слияний.

По кадрам боя считаем обе величины и сравниваем.

    .venv/bin/python tools/rank_shape_count.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    round_count = other_count = 0
    for side, shape in re.findall(r"  (своя|чужая) \d+,\d+ радиус \d+ (круглое|НЕ круглое)", out):
        if side != "своя":
            continue
        if shape == "круглое":
            round_count += 1
        else:
            other_count += 1
    return round_count, other_count


def parse_log(log: Path):
    """Число успешных и неудачных слияний к каждому шагу."""
    per_step = {}
    pending, step, round_no = None, 0, 1
    for line in log.read_text(errors="ignore").split("\n"):
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\)", line)
        if m:
            pending = {"round": round_no, "step": step}
        if pending and "слияние вышло" in line:
            per_step[(round_no, step)] = per_step.get((round_no, step), 0) + 1
            pending = None
        elif pending and "слияние не вышло" in line:
            pending = None
    return per_step


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    merges = parse_log(log)
    print(f"{'кадр':>10} {'шаг':>5} {'успешных':>9} {'круглых':>9} {'не-круглых':>11}")
    for frame in sorted(directory.glob("b*.png"),
                        key=lambda p: (int(re.search(r"b(\d+)_", p.name).group(1)),
                                       int(p.name.split("_")[1].split(".")[0]))):
        m = re.match(r"b(\d+)_(\d+)\.png", frame.name)
        if not m:
            continue
        round_no = int(m.group(1)) + 1
        step = int(m.group(2))
        round_count, other_count = pawns_of(frame)
        done = sum(v for (r, s), v in merges.items() if r == round_no and s <= step)
        print(f"{frame.stem:>10} {step:5d} {done:9d} {round_count:9d} {other_count:11d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
