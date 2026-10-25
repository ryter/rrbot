#!/usr/bin/env python3
"""Надёжное измерение формы основания при слиянии.

Камера в бою ходит сильно (до 70 px), поэтому компенсировать сдвиг по
«началу арены» нельзя — её детектор даёт разные прямоугольники. Сдвиг
оценивается по совпавшим пешкам: пешки сопоставляются между кадрами жадным
поиском, сдвиг — медиана их перемещений.

Дальше по логу боя для каждого успешного слияния берётся клетка цели и
меряется форма основания ДО и ПОСЛЕ.

    .venv/bin/python tools/merge_shape_robust.py <лог> <каталог кадров>
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


def pawns_of(frame: Path):
    """Все свои пешки: координаты кадра (в т.ч. с не-круглым основанием)."""
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append((int(x), int(y), int(r)))
    return pawns


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


def camera_shift(before, after, limit=45):
    """Сдвиг камеры между кадрами по совпавшим пешкам (медиана)."""
    used = set()
    deltas = []
    for pawn in after:
        best, best_d = None, limit
        for index, old in enumerate(before):
            if index in used:
                continue
            d = max(abs(old[0] - pawn[0]), abs(old[1] - pawn[1]))
            if d < best_d:
                best_d, best = d, index
        if best is not None:
            used.add(best)
            deltas.append((after[0][0] - before[best][0], after[0][1] - before[best][1]))
    if len(deltas) < 2:
        return (0, 0)
    arr = np.array(deltas)
    return (int(np.median(arr[:, 0])), int(np.median(arr[:, 1])))


def nearest(pawns, x, y, limit=50):
    best, best_d = None, limit
    for pawn in pawns:
        d = max(abs(pawn[0] - x), abs(pawn[1] - y))
        if d < best_d:
            best_d, best = d, pawn
    return best


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
    print(f"{'исход':>6} | {'fill до':>8} {'fill после':>10} {'раст до':>8} "
          f"{'раст после':>10} | {'форма до':>10} {'форма после':>11}")
    counts = {True: {}, False: {}}
    for event in parse_log(log):
        before = directory / f"b{event['round'] - 1}_{event['step']}.png"
        after = directory / f"b{event['round'] - 1}_{event['step'] + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        pawns_b = pawns_of(before)
        pawns_a = pawns_of(after)
        if not pawns_b or not pawns_a:
            continue
        shift = camera_shift(pawns_b, pawns_a)
        tx, ty = event["dst"][0] + shift[0], event["dst"][1] + shift[1]
        target_b = nearest(pawns_b, *event["dst"])
        target_a = nearest(pawns_a, tx, ty)
        if not target_b or not target_a:
            continue
        sb = shape_of(ib, target_b[0], target_b[1])
        sa = shape_of(ia, target_a[0], target_a[1])
        if not sb or not sa:
            continue
        print(f"{'ок' if event['ok'] else 'нет':>6} | "
              f"{sb['fill']:8.3f} {sa['fill']:10.3f} "
              f"{sb['stretch']:8.2f} {sa['stretch']:10.2f} | "
              f"{kind(sb):>10} {kind(sa):>11}")
        key = (kind(sb), kind(sa))
        counts[event["ok"]][key] = counts[event["ok"]].get(key, 0) + 1
    print()
    for ok in (True, False):
        title = "успешные" if ok else "неудачные"
        print(f"{title}:")
        for key, count in sorted(counts[ok].items(), key=lambda kv: -kv[1]):
            print(f"  {key[0]:8} -> {key[1]:8} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
