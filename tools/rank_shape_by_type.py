#!/usr/bin/env python3
"""Форма основания: тип юнита или ранг?

Сопоставляем тип пешки (по зрению) с формой её основания. Если форма
кодирует тип, у одного типа она одинакова во всех рангах — и правило
«круглое основание = ранг 1» бессмысленно.

    .venv/bin/python tools/rank_shape_by_type.py <csv> <каталог кадров>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def vision_types(frame: Path):
    """Типы юнитов и их позиции (из --vision)."""
    out = subprocess.run([str(CPP / "build" / "rrbot"), "--backend", "adb",
                          "--frame", str(frame), "--vision"],
                         capture_output=True, text=True).stdout
    found = []
    for side, name, card, x, y in re.findall(
            r"(свой|чужой) (\w+) карточка=(\d+) \((\d+),(\d+)\)", out):
        found.append((side == "свой", name, int(x), int(y)))
    return found


def main() -> int:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank_track.csv")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    by_frame: dict[str, list] = {}
    for row in csv.DictReader(csv_path.open()):
        by_frame.setdefault(row["frame"], []).append(row)
    groups: dict[str, list] = defaultdict(list)
    for frame_name, group in by_frame.items():
        # кадры со старшими рангами плюс часть обычных
        has_high = any(int(r["rank"]) >= 2 for r in group)
        if not has_high and len(group) < 8:
            continue
        frame = directory / frame_name
        if not frame.exists():
            continue
        types = vision_types(frame)
        if not types:
            continue
        arena = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                               capture_output=True, text=True).stdout
        m = re.search(r"арена: своя (-?\d+),(-?\d+) ", arena)
        if not m:
            continue
        ax, ay = int(m.group(1)), int(m.group(2))
        for row in group:
            px = int(row["x"]) + ax
            py = int(row["y"]) + ay
            # трекер пишет координаты арены; зрение — кадра
            best = None
            best_d = 60
            for ours, name, x, y in types:
                if not ours:
                    continue
                d = max(abs(x - px), abs(y - py))
                if d < best_d:
                    best_d, best = d, name
            if best is None:
                continue
            groups[best].append((int(row["rank"]), float(row["fill"]),
                                 float(row["stretch"]), int(row["r"])))
    print(f"{'тип':16} {'пешек':>7} {'ранг1: fill':>18} {'ранг≥2: fill':>18}")
    for name, rows in sorted(groups.items()):
        by_rank = defaultdict(list)
        for rank, fill, stretch, r in rows:
            by_rank[rank].append(fill)
        def fmt(values):
            if not values:
                return "—"
            return f"{np.mean(values):.3f} (n={len(values)})"
        print(f"{name:16} {len(rows):7d} {fmt(by_rank.get(1, [])):>18} "
              f"{fmt(by_rank.get(2, []) + by_rank.get(3, [])):>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
