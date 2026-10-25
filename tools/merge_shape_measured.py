#!/usr/bin/env python3
"""Форма основания до и после слияния — со сдвигом камеры.

Раньше замеры путались: камера в бою ходит, а координаты цели в логе
относятся к кадру ДО слияния. Здесь сдвиг вычисляется по началу арены и
компенсируется, так что «до» и «после» — это одна и та же клетка.

Печатаем форму основания (заполненность, вытянутость) и радиус.

    .venv/bin/python tools/merge_shape_measured.py <лог> <каталог кадров>
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


def probe(frame: Path):
    """Пешки и начало своей половины арены."""
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) (\d+)x(\d+)", out)
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append((int(x), int(y), int(r)))
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2)), int(arena.group(3))), pawns


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
    print(f"{'исход':>10} {'цель':>10} | {'fill до':>8} {'fill после':>10} "
          f"{'раст до':>8} {'раст после':>10} | {'r до':>5} {'r после':>7}")
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
        arena_b, pawns_b = probe(before)
        arena_a, pawns_a = probe(after)
        if not arena_b or not arena_a:
            continue
        shift = (arena_a[0] - arena_b[0], arena_a[1] - arena_b[1])
        # цель на кадре «после» (сдвиг камеры компенсирован)
        tx, ty = event["dst"][0] + shift[0], event["dst"][1] + shift[1]
        near_b = min(pawns_b, key=lambda p: max(abs(p[0] - event["dst"][0]),
                                                abs(p[1] - event["dst"][1])))
        near_a = min(pawns_a, key=lambda p: max(abs(p[0] - tx), abs(p[1] - ty)))
        if max(abs(near_a[0] - tx), abs(near_a[1] - ty)) > 45:
            continue
        if max(abs(near_b[0] - event["dst"][0]), abs(near_b[1] - event["dst"][1])) > 45:
            continue
        sb = shape_of(ib, near_b[0], near_b[1])
        sa = shape_of(ia, near_a[0], near_a[1])
        if not sb or not sa:
            continue
        print(f"{'ок' if event['ok'] else 'нет':>10} {'':>10} | "
              f"{sb['fill']:8.3f} {sa['fill']:10.3f} "
              f"{sb['stretch']:8.2f} {sa['stretch']:10.2f} | "
              f"{near_b[2]:5d} {near_a[2]:7d}")
        if sb["fill"] >= 0.88 and sa["fill"] < 0.88:
            changed += 1
        elif abs(sb["fill"] - sa["fill"]) < 0.02:
            same += 1
    print()
    print(f"форма изменилась (круг -> не круг): {changed}, не изменилась: {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
