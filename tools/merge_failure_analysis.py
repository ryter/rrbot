#!/usr/bin/env python3
"""Почему слияние не выходит: разбираем пары по кадрам перед попыткой.

Для каждой попытки слияния берём кадр, на котором бот её делал, и узнаём
у каждой пешки радиус основания и заполненность (по ней судят о ранге).
Сравниваем удачные и неудачные пары: если у неудачных пешки систематически
разного размера, значит дело в ранге (игра не принимает 1+2), а если нет —
в самом жесте.

    .venv/bin/python tools/merge_failure_analysis.py <каталог_кадров> <лог>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")
PROBE = CPP / "build" / "merge_probe"


def parse_log(path: Path):
    """Попытки слияния: (раунд, шаг, откуда, куда, похожесть, удалось)."""
    round_no = 1
    step = 0
    last = None
    out = []
    for line in path.read_text(errors="ignore").split("\n"):
        m = re.match(r"== раунд (\d+)/", line)
        if m:
            round_no = int(m.group(1))
        m = re.match(r"\[(\d+)\] энергия", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\), похожесть ([\d.]+)", line)
        if m:
            last = (round_no, step, int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)), float(m.group(5)))
        if "слияние вышло" in line and last:
            out.append((last, True))
            last = None
        elif "слияние не вышло" in line and last:
            out.append((last, False))
            last = None
    return out


def pawns_of(frame: Path):
    """Пешки кадра: (x, y) -> (радиус, заполненность)."""
    out = subprocess.run([str(PROBE), str(frame)], capture_output=True, text=True).stdout
    pawns = {}
    for x, y, radius, fill in re.findall(
            r"  своя (\d+),(\d+) радиус (\d+) круглое заполненность ([\d.]+)", out):
        pawns[(int(x), int(y))] = (int(radius), float(fill))
    for x, y, radius, fill in re.findall(
            r"  своя (\d+),(\d+) радиус (\d+) НЕ круглое заполненность ([\d.]+)", out):
        pawns[(int(x), int(y))] = (int(radius), float(fill))
    return pawns


def nearest(pawns, point, limit=30):
    best = None
    best_distance = limit
    for key, value in pawns.items():
        distance = max(abs(key[0] - point[0]), abs(key[1] - point[1]))
        if distance < best_distance:
            best_distance = distance
            best = (key, value)
    return best


def white_pixels(frame: Path, point, radius: int = 42) -> int:
    """Сколько белых (вспышек) пикселей вокруг пешки.

    Вспышки на юните — признак того, что юнит сейчас что-то делает
    (атакует, стреляет, получает урон). Если игра не сливает таких,
    это объяснит часть неудач.
    """
    image = cv2.imread(str(frame))
    if image is None:
        return -1
    x, y = point
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image.shape[1], x + radius)
    y1 = min(image.shape[0], y + radius)
    if x1 <= x0 or y1 <= y0:
        return -1
    patch = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 225), (180, 70, 255))
    return int(cv2.countNonZero(mask))


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/battle_loop")
    log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/battle_loop.log")
    attempts = parse_log(log)
    print(f"попыток слияния: {len(attempts)}")
    rows = []
    for (rnd, step, x1, y1, x2, y2, score), ok in attempts:
        frame = directory / f"b{rnd - 1}_{step}.png"
        if not frame.exists():
            continue
        pawns = pawns_of(frame)
        src = nearest(pawns, (x1, y1))
        dst = nearest(pawns, (x2, y2))
        if src is None or dst is None:
            continue
        (rs, fs), (rt, ft) = src[1], dst[1]
        rows.append({
            "ok": ok, "score": score, "r_src": rs, "r_dst": rt,
            "dr": abs(rs - rt), "fill_src": fs, "fill_dst": ft,
            "rel": abs(rs - rt) / max(rs, rt),
            "flash_src": white_pixels(frame, (x1, y1)),
            "flash_dst": white_pixels(frame, (x2, y2)),
        })
    if not rows:
        print("нет данных (кадры не найдены)")
        return 1
    for ok in (True, False):
        part = [row for row in rows if row["ok"] == ok]
        if not part:
            continue
        n = len(part)
        avg_rel = sum(row["rel"] for row in part) / n
        same_fill = sum(1 for row in part
                        if (row["fill_src"] >= 0.88) == (row["fill_dst"] >= 0.88)) / n
        big_diff = sum(1 for row in part if row["rel"] > 0.10) / n
        print(f"{'удалось' if ok else 'не вышло'}: {n}   "
              f"разница радиусов в среднем {avg_rel * 100:.1f} %, "
              f"разные основания {same_fill * 100:.0f} %, "
              f"размеры расходятся больше чем на 10 %: {big_diff * 100:.0f} %")
    print()
    print("разница радиусов по парам (удачные / неудачные):")
    for row in sorted(rows, key=lambda r: r["rel"]):
        mark = "ок  " if row["ok"] else "нет "
        print(f"  {mark} радиусы {row['r_src']:3d} и {row['r_dst']:3d} "
              f"(разница {row['rel'] * 100:5.1f} %), заполненность "
              f"{row['fill_src']:.2f}/{row['fill_dst']:.2f}, похожесть {row['score']:.2f}")
    # Какие правила отсекают неудачные пары и сколько при этом теряем
    # удачных: правило хорошее, если отсекает много неудачных и мало
    # удачных.
    print()
    print()
    print("вспышки (белые пиксели) у пешек пары:")
    for ok in (True, False):
        part = [r for r in rows if r["ok"] == ok and r["flash_src"] >= 0]
        if not part:
            continue
        avg_src = sum(r["flash_src"] for r in part) / len(part)
        avg_dst = sum(r["flash_dst"] for r in part) / len(part)
        busy = sum(1 for r in part if max(r["flash_src"], r["flash_dst"]) > 200) / len(part)
        print(f"  {'удалось' if ok else 'не вышло'}: вспышек у источника {avg_src:.0f}, "
              f"у цели {avg_dst:.0f}, пар с заметной вспышкой {busy * 100:.0f} %")
    print("проверка правил (остаётся успешных / отсекается неудачных):")
    ok_rows = [r for r in rows if r["ok"]]
    bad_rows = [r for r in rows if not r["ok"]]
    rules = [
        ("размеры пешек различаются не больше чем на 8 %", lambda r: r["rel"] <= 0.08),
        ("размеры пешек различаются не больше чем на 12 %", lambda r: r["rel"] <= 0.12),
        ("размеры пешек различаются не больше чем на 15 %", lambda r: r["rel"] <= 0.15),
        ("у обеих пешек круглое основание (заполненность >= 0.85)",
         lambda r: min(r["fill_src"], r["fill_dst"]) >= 0.85),
        ("у обеих пешек круглое основание (заполненность >= 0.80)",
         lambda r: min(r["fill_src"], r["fill_dst"]) >= 0.80),
        ("портреты очень похожи (>= 0.85)", lambda r: r["score"] >= 0.85),
        ("портреты похожи (>= 0.75)", lambda r: r["score"] >= 0.75),
        ("размеры <= 12 % и основания круглые",
         lambda r: r["rel"] <= 0.12 and min(r["fill_src"], r["fill_dst"]) >= 0.85),
        ("размеры <= 12 % и портреты >= 0.85",
         lambda r: r["rel"] <= 0.12 and r["score"] >= 0.85),
        ("размеры <= 12 % и портреты >= 0.75",
         lambda r: r["rel"] <= 0.12 and r["score"] >= 0.75),
        ("размеры <= 15 % и портреты >= 0.75",
         lambda r: r["rel"] <= 0.15 and r["score"] >= 0.75),
        ("размеры <= 20 % и портреты >= 0.80",
         lambda r: r["rel"] <= 0.20 and r["score"] >= 0.80),
        ("размеры <= 15 % и портреты >= 0.80",
         lambda r: r["rel"] <= 0.15 and r["score"] >= 0.80),
    ]
    for name, rule in rules:
        kept_ok = sum(1 for r in ok_rows if rule(r))
        blocked = sum(1 for r in bad_rows if not rule(r))
        print(f"  {name}: успешных осталось {kept_ok}/{len(ok_rows)}, "
              f"неудачных отсеяно {blocked}/{len(bad_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
