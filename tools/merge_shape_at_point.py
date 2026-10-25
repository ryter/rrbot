#!/usr/bin/env python3
"""Форма основания на месте цели: ДО и ПОСЛЕ слияния.

Важная поправка: детектор пешек ищет только КРУГЛЫЕ основания (Hough), так
что юниты с не-круглой подставкой в его список не попадают. Поэтому
измерять форму «по найденной пешке» нельзя — сравнивал бы не то.

Здесь форма меряется прямо по координатам цели из лога, с компенсацией
сдвига камеры (начало арены).

    .venv/bin/python tools/merge_shape_at_point.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
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


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    print(f"{'исход':>6} | {'fill до':>8} {'fill после':>10} {'раст до':>8} "
          f"{'раст после':>10} | {'круг? до':>9} {'круг? после':>11}")
    changed = same = 0
    for event in parse_log(log):
        before = directory / f"b{event['round'] - 1}_{event['step']}.png"
        after = directory / f"b{event['round'] - 1}_{event['step'] + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        arena_b = arena_of(before)
        arena_a = arena_of(after)
        if not arena_b or not arena_a:
            continue
        shift = (arena_a[0] - arena_b[0], arena_a[1] - arena_b[1])
        tx, ty = event["dst"][0] + shift[0], event["dst"][1] + shift[1]
        sb = shape_of(ib, *event["dst"])
        sa = shape_of(ia, tx, ty)
        if not sb or not sa:
            continue
        print(f"{'ок' if event['ok'] else 'нет':>6} | "
              f"{sb['fill']:8.3f} {sa['fill']:10.3f} "
              f"{sb['stretch']:8.2f} {sa['stretch']:10.2f} | "
              f"{'да' if sb['fill'] >= 0.88 else 'нет':>9} "
              f"{'да' if sa['fill'] >= 0.88 else 'нет':>11}")
        if sb["fill"] >= 0.88 and sa["fill"] < 0.88:
            changed += 1
        elif abs(sb["fill"] - sa["fill"]) < 0.02:
            same += 1
    print()
    print(f"форма сменилась (круг -> не круг): {changed}, не изменилась: {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
