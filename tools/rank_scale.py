#!/usr/bin/env python3
"""Гипотеза «ранг = масштаб спрайта».

В Rush Royale юнит высокого ранга нарисован крупнее. Значит, ранг можно
определять по размеру спрайта, а не по форме основания.

Замер: у результата успешного слияния ранг на 1 выше, чем у исходных
пешек. Сравниваем:
  * площадь и высоту спрайта (всё, что выше основания);
  * радиус основания.

    .venv/bin/python tools/rank_scale.py <лог> <каталог кадров>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pedestal_shape import is_tile  # noqa: E402

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def pawns_of(frame: Path):
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    return [(int(x), int(y), int(r)) for x, y, r in
            re.findall(r"своя (\d+),(\d+) радиус (\d+)", out)]


def sprite_size(image, cx, cy, radius):
    """Габариты спрайта над основанием: всё, что не плитка и выше центра."""
    half = int(radius * 2.2)
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
    y0, y1 = max(0, cy - int(radius * 2.6)), min(image.shape[0], cy)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = image[y0:y1, x0:x1]
    mask = ~is_tile(patch)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return {"width": width, "height": height, "area": int(mask.sum())}


def parse_log(log: Path):
    events, round_no, step, pending = [], 1, 0, None
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


def nearest(pawns, x, y, limit=50):
    best, best_d = None, limit
    for p in pawns:
        d = max(abs(p[0] - x), abs(p[1] - y))
        if d < best_d:
            best_d, best = d, p
    return best


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    rows = []
    for r, s, x1, y1, x2, y2, ok in parse_log(log):
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
        src = nearest(pb, x1, y1)
        tgt = nearest(pb, x2, y2)
        res = nearest(pa, x2, y2)
        if not src or not tgt or not res:
            continue
        sb = sprite_size(ib, src[0], src[1], src[2])
        tb = sprite_size(ib, tgt[0], tgt[1], tgt[2])
        ra = sprite_size(ia, res[0], res[1], res[2])
        if not sb or not tb or not ra:
            continue
        rows.append({
            "src_h": sb["height"], "src_w": sb["width"], "src_r": src[2],
            "tgt_h": tb["height"], "tgt_w": tb["width"], "tgt_r": tgt[2],
            "res_h": ra["height"], "res_w": ra["width"], "res_r": res[2],
        })
    if not rows:
        print("данных нет")
        return 1
    print(f"{'выс src':>8} {'выс tgt':>8} {'выс рез':>8} | "
          f"{'шир src':>8} {'шир рез':>8} | {'r tgt':>6} {'r рез':>6}")
    for row in rows:
        print(f"{row['src_h']:8d} {row['tgt_h']:8d} {row['res_h']:8d} | "
              f"{row['src_w']:8d} {row['res_w']:8d} | {row['tgt_r']:6d} {row['res_r']:6d}")
    print()
    for key in ("src_h", "tgt_h", "res_h", "src_w", "res_w", "tgt_r", "res_r"):
        values = [row[key] for row in rows]
        print(f"  {key:8}: мин {min(values):4d} медиана {int(np.median(values)):4d} "
              f"макс {max(values):4d}")
    taller = sum(1 for row in rows if row["res_h"] > row["tgt_h"])
    same = sum(1 for row in rows if row["res_h"] == row["tgt_h"])
    smaller = sum(1 for row in rows if row["res_h"] < row["tgt_h"])
    print()
    print(f"высота результата vs цели: выше {taller}, так же {same}, ниже {smaller}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
