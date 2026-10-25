#!/usr/bin/env python3
"""Статистика поиска арены в «Подземельях» (Co-Op) по сохранённым кадрам.

Прогоняет бинарник merge_probe по всем боевым кадрам каталога и считает:
* в какой доле кадров арена нашлась (комнаты Co-Op или цветные плитки PvP);
* согласованы ли границы своей комнаты от кадра к кадру;
* сколько своих пешек видно в среднем.

    .venv/bin/python tools/arena_coop_stats.py <каталог_кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
PROBE = CPP / "build" / "merge_probe"


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop")
    frames = sorted(directory.glob("b*.png"))
    if not frames:
        print("кадров нет:", directory)
        return 1
    found = 0
    nofound = 0
    rooms = Counter()
    ours_counts = []
    unfound = []
    for frame in frames:
        out = subprocess.run([str(PROBE), str(frame)], capture_output=True,
                             text=True).stdout + subprocess.run(
                                 [str(PROBE), str(frame)],
                                 capture_output=True, text=True).stderr
        m = re.search(r"арена: своя (\d+),(\d+) (\d+)x(\d+); чужая (\d+),(\d+)", out)
        n = re.search(r"пешек всего (\d+), своих (\d+)", out)
        if m:
            found += 1
            rooms[(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                   int(m.group(4)))] += 1
            if n:
                ours_counts.append(int(n.group(2)))
        else:
            nofound += 1
            unfound.append(frame.name)
    total = found + nofound
    print(f"кадров: {total}, арена найдена: {found} ({found * 100 // total}%), "
          f"не найдена: {nofound}")
    if ours_counts:
        ours_counts.sort()
        print(f"своих пешек: мин {ours_counts[0]}, медиана "
              f"{ours_counts[len(ours_counts) // 2]}, макс {ours_counts[-1]}")
    print("своя комната (x,y WxH -> сколько кадров):")
    for room, count in rooms.most_common(6):
        print(f"  {room} -> {count}")
    if unfound:
        print("без арены:", " ".join(unfound[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
