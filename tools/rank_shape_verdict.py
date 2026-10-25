#!/usr/bin/env python3
"""Форма основания: тип юнита или ранг?

Решающий опыт: игра не сливает юнита ранга 1 с юнитом ранга 2, а такие
пары выглядят одинаково. Значит, если в основе кодируется ранг, то
успешные пары будут однотипными по форме, а неудачные — разными.

По логу боя собираем все попытки слияния и для каждой — форму основания
источника и цели.

    .venv/bin/python tools/rank_shape_verdict.py <лог> <каталог кадров>
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
    return [(int(x), int(y), int(r)) for x, y, r in
            re.findall(r"своя (\d+),(\d+) радиус (\d+)", out)]


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


def nearest(pawns, x, y, limit=50):
    best, best_d = None, limit
    for p in pawns:
        d = max(abs(p[0] - x), abs(p[1] - y))
        if d < best_d:
            best_d, best = d, p
    return best


def bucket(shape):
    """Грубая форма: круг / овал-щит / прочее."""
    if shape["stretch"] <= 1.12:
        return "круг"
    if shape["stretch"] <= 1.30:
        return "овал"
    return "вытянутый"


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    stats = {True: Counter(), False: Counter()}
    detail = {True: [], False: []}
    for event in parse_log(log):
        frame = directory / f"b{event['round'] - 1}_{event['step']}.png"
        if not frame.exists():
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        pawns = pawns_of(frame)
        src = nearest(pawns, *event["src"])
        dst = nearest(pawns, *event["dst"])
        if not src or not dst:
            continue
        s1 = shape_of(image, src[0], src[1])
        s2 = shape_of(image, dst[0], dst[1])
        if not s1 or not s2:
            continue
        b1, b2 = bucket(s1), bucket(s2)
        key = f"{b1}+{b2}" if b1 <= b2 else f"{b2}+{b1}"
        stats[event["ok"]][key] += 1
        detail[event["ok"]].append((b1, b2, round(s1["fill"], 3), round(s2["fill"], 3),
                                    round(s1["stretch"], 2), round(s2["stretch"], 2)))
    print("успешные слияния (форма источника + форма цели):")
    for key, count in stats[True].most_common():
        print(f"  {key:22} {count}")
    print("неудачные:")
    for key, count in stats[False].most_common():
        print(f"  {key:22} {count}")
    print()
    same_ok = sum(v for k, v in stats[True].items() if k.split("+")[0] == k.split("+")[1])
    tot_ok = sum(stats[True].values())
    same_bad = sum(v for k, v in stats[False].items() if k.split("+")[0] == k.split("+")[1])
    tot_bad = sum(stats[False].values())
    print(f"одинаковая форма: успешных {same_ok}/{tot_ok}, неудачных {same_bad}/{tot_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
