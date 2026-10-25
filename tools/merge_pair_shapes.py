#!/usr/bin/env python3
"""Правило слияния: совпадает ли форма оснований у пары?

Слияние в игре разрешено только для юнитов одного типа и одного ранга.
Если ранг кодирует форма подставки (круг — ранг 1, квадрат — ранг 2, овал
— ранг 3), то у успешных пар формы должны совпадать, а у неудачных —
нет.

Важно: формы меряются по точным координатам пешек из лога, а не по списку
найденных пешек — детектор ищет только круглые основания, и слитые юниты
в его список не попадают.

    .venv/bin/python tools/merge_pair_shapes.py <лог> <каталог кадров>
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


def arena_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    m = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_log(log: Path):
    events, pending, step, round_no = [], None, 0, 1
    for line in log.read_text(errors="ignore").split("\n"):
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\)", line)
        if m:
            pending = {"round": round_no, "step": step,
                       "src": (int(m.group(1)), int(m.group(2))),
                       "dst": (int(m.group(3)), int(m.group(4))),
                       "ok": None}
        if pending and ("слияние вышло" in line or "слияние не вышло" in line):
            pending["ok"] = "слияние вышло" in line
            events.append(pending)
            pending = None
    return events


def kind(shape):
    if not shape:
        return "нет"
    if shape["fill"] >= 0.88 and shape["stretch"] <= 1.12:
        return "круг"
    if shape["fill"] >= 0.68:
        return "квадрат"
    return "овал"


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    stats = {True: Counter(), False: Counter()}
    for event in parse_log(log):
        frame = directory / f"b{event['round'] - 1}_{event['step']}.png"
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        arena = arena_of(frame)
        if not arena:
            continue
        # обе пешки в координатах кадра: лог пишет координаты кадра
        ss = shape_of(image, *event["src"])
        sd = shape_of(image, *event["dst"])
        if not ss or not sd:
            continue
        k1, k2 = kind(ss), kind(sd)
        key = f"{k1}+{k2}" if k1 <= k2 else f"{k2}+{k1}"
        stats[event["ok"]][key] += 1
    print("успешные слияния:")
    for key, count in stats[True].most_common():
        print(f"  {key:16} {count}")
    print("неудачные:")
    for key, count in stats[False].most_common():
        print(f"  {key:16} {count}")
    print()
    for ok in (True, False):
        same = sum(v for k, v in stats[ok].items() if k.split("+")[0] == k.split("+")[1])
        print(f"{'успешных' if ok else 'неудачных'}: одинаковая форма "
              f"{same} из {sum(stats[ok].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
