#!/usr/bin/env python3
"""Разметка рангов по слияниям и сбор признаков пешек.

Грунтовная правда берётся из игры:
  * успешное слияние -> юнит на месте цели имеет ранг на 1 выше, чем был;
  * пешка, только что призванная, имеет ранг 1.

Поэтому по логу боя («сливаю: … на …» / «слияние вышло») и кадрам можно
разметить пешки как ранг 1 (свежепризванные) и ранг 2 (результат удачного
слияния), а потом посмотреть, какие признаки их различают.

    .venv/bin/python tools/rank_labels.py <лог> <каталог кадров>
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


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    pawns = []
    for side, x, y, r, round_ in re.findall(
            r"(своя|чужая) (\d+),(\d+) радиус (\d+) (круглое|НЕ круглое)", out):
        pawns.append((int(x), int(y), int(r), side == "своя", round_ == "круглое"))
    return pawns


def parse_log(log: Path):
    """[(раунд, шаг, src, dst, ok)] — по логу бота."""
    events = []
    round_no = 1
    step = 0
    pending = None
    for line in log.read_text(errors="ignore").split("\n"):
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\)", line)
        if m:
            pending = (round_no, step, int(m.group(1)), int(m.group(2)),
                       int(m.group(3)), int(m.group(4)))
        if pending and "слияние вышло" in line:
            events.append((*pending, True))
            pending = None
        elif pending and "слияние не вышло" in line:
            events.append((*pending, False))
            pending = None
    return events


def nearest(pawns, x, y, limit=45):
    best = None
    best_d = limit
    for p in pawns:
        d = max(abs(p[0] - x), abs(p[1] - y))
        if d < best_d:
            best_d = d
            best = p
    return best


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/battle_loop")
    events = parse_log(log)
    print(f"слияний в логе: {len(events)}")
    if not events:
        return 1
    rows = []
    for r, s, x1, y1, x2, y2, ok in events:
        if not ok:
            continue
        before = directory / f"b{r - 1}_{s}.png"
        after = directory / f"b{r - 1}_{s + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        pb = pawns_of(before)
        pa = pawns_of(after)
        # Цель до слияния — ранг N (как правило 1), результат после — ранг N+1.
        target_before = nearest(pb, x2, y2)
        result_after = nearest(pa, x2, y2)
        if not target_before or not result_after:
            continue
        sb = shape_of(ib, target_before[0], target_before[1])
        sa = shape_of(ia, result_after[0], result_after[1])
        if not sb or not sa:
            continue
        rows.append({
            "src": (x1, y1), "dst": (x2, y2),
            "r_before": target_before[2], "r_after": result_after[2],
            "round_before": target_before[4], "round_after": result_after[4],
            "fill_before": sb["fill"], "fill_after": sa["fill"],
            "stretch_before": sb["stretch"], "stretch_after": sa["stretch"],
            "area_before": sb["area"], "area_after": sa["area"],
        })
    if not rows:
        print("разметить нечего")
        return 1
    print(f"размечено слияний: {len(rows)}")
    print()
    hdr = (f"{'fill до':>8} {'fill после':>10} {'раст до':>8} {'раст после':>9} "
           f"{'r до':>5} {'r после':>7}  круг?")
    print(hdr)
    for row in rows:
        print(f"{row['fill_before']:8.3f} {row['fill_after']:10.3f} "
              f"{row['stretch_before']:8.2f} {row['stretch_after']:9.2f} "
              f"{row['r_before']:5d} {row['r_after']:7d}  "
              f"{'да' if row['round_before'] else 'нет'}->"
              f"{'да' if row['round_after'] else 'нет'}")
    print()
    print("сводка:")
    for key in ("fill_before", "fill_after", "stretch_before", "stretch_after",
                "r_before", "r_after"):
        values = [row[key] for row in rows]
        print(f"  {key:16}: мин {min(values):.3f} медиана {np.median(values):.3f} "
              f"макс {max(values):.3f}")
    print()
    print("разбивка по заполненности (порог 0.88):")
    counter = Counter()
    for row in rows:
        counter[("до", row["round_before"])] += 1
        counter[("после", row["round_after"])] += 1
    for key in sorted(counter):
        print(f"  {key}: {counter[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
