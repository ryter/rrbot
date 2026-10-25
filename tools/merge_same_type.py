#!/usr/bin/env python3
"""Результат слияния: тот же тип юнита или другая карта?

Сравниваем палитру пешки-цели до слияния и пешки на её месте после.
По палитре (гистограмма HSV по кругу основания) один тип даёт косинус
~0.99, разные типы — 0.2-0.3.

    .venv/bin/python tools/merge_same_type.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
HIST_H, HIST_S, HIST_V = 18, 4, 4


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y), int(r)) for x, y, r in
            re.findall(r"своя (\d+),(\d+) радиус (\d+)", out)]


def palette(image, cx, cy, radius):
    inner = max(4, radius - 3)
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    patch = image[y0:cy + radius, x0:cx + radius]
    if patch.shape[0] < 2 * radius or patch.shape[1] < 2 * radius:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = np.zeros(HIST_H * HIST_S * HIST_V, np.float32)
    cyy, cxx = np.mgrid[0:2 * radius, 0:2 * radius]
    mask = (cxx - radius) ** 2 + (cyy - radius) ** 2 <= inner * inner
    for pixel in hsv[mask]:
        h = min(HIST_H - 1, int(pixel[0]) * HIST_H // 180)
        s = min(HIST_S - 1, int(pixel[1]) * HIST_S // 256)
        v = min(HIST_V - 1, int(pixel[2]) * HIST_V // 256)
        hist[h * HIST_S * HIST_V + s * HIST_V + v] += 1
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def cosine(a, b):
    if a is None or b is None:
        return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest(pawns, x, y, limit=50):
    best, best_d = None, limit
    for p in pawns:
        d = max(abs(p[0] - x), abs(p[1] - y))
        if d < best_d:
            best_d, best = d, p
    return best


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
    print(f"{'исход':>8} {'цель':>8} {'похожесть до':>13} {'после':>8}  вердикт")
    same, different = 0, 0
    for event in parse_log(log):
        before = directory / f"b{event['round'] - 1}_{event['step']}.png"
        after = directory / f"b{event['round'] - 1}_{event['step'] + 1}.png"
        if not before.exists() or not after.exists():
            continue
        ib = cv2.imread(str(before))
        ia = cv2.imread(str(after))
        if ib is None or ia is None:
            continue
        pb = pawns_of(before)
        pa = pawns_of(after)
        src = nearest(pb, *event["src"])
        tgt = nearest(pb, *event["dst"])
        res = nearest(pa, *event["dst"])
        if not src or not tgt or not res:
            continue
        p_src = palette(ib, src[0], src[1], src[2])
        p_tgt = palette(ib, tgt[0], tgt[1], tgt[2])
        p_res = palette(ia, res[0], res[1], res[2])
        sim_src_tgt = cosine(p_src, p_tgt)
        sim_tgt_res = cosine(p_tgt, p_res)
        verdict = "тот же тип" if sim_tgt_res >= 0.9 else "ДРУГОЙ тип"
        if sim_tgt_res >= 0.9:
            same += 1
        else:
            different += 1
        print(f"{sim_src_tgt:8.3f} {'':>8} {'':>13} {sim_tgt_res:8.3f}  {verdict}")
    print()
    print(f"тот же тип: {same}, другой: {different}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
