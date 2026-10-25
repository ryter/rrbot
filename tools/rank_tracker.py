#!/usr/bin/env python3
"""Трекер пешек: разметка рангов по логу боя.

Правда о рангах берётся из самой игры:
  * пешка, которой не было на прошлом кадре и которая появилась не на месте
    удачного слияния, — призванная, ранг 1;
  * пешка на месте цели сразу после удачного слияния — ранг на 1 выше, чем
    был у цели.

Позиции сравниваются в координатах арены (начало арены вычитается), поэтому
сдвиг камеры не путает трекер.

На выходе — CSV: кадр, позиция, ранг, признаки. По нему ищется признак,
который различает ранги.

    .venv/bin/python tools/rank_tracker.py <лог> <каталог кадров> <out.csv>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pedestal_shape import is_tile, shape_of  # noqa: E402

CPP = Path("/home/lennistyle/rushroyale-lab/cpp")


def probe(frame: Path):
    """Пешки своей стороны и начало арены на кадре."""
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    if not arena:
        return None, []
    ax, ay = int(arena.group(1)), int(arena.group(2))
    pawns = []
    for side, x, y, r, round_ in re.findall(
            r"(своя|чужая) (\d+),(\d+) радиус (\d+) (круглое|НЕ круглое)", out):
        if side != "своя":
            continue
        pawns.append({"x": int(x) - ax, "y": int(y) - ay,
                      "px": int(x), "py": int(y), "r": int(r),
                      "round": round_ == "круглое", "rank": None})
    return (ax, ay), pawns


def parse_log(log: Path):
    """Слияния по шагам: раунд, шаг, источник, цель, успех."""
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


HIST_H, HIST_S, HIST_V = 18, 4, 4


def palette(image, cx, cy, radius):
    """Палитра основания (гистограмма HSV) — по ней юниты одного типа
    совпадают (косинус ~0.99), разные типы — 0.2-0.3."""
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
        return 0.0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 0:
        return 0.0
    return float(a @ b / norm)


def sprite_features(image, cx, cy, radius):
    """Габариты спрайта: всё, что не плитка и выше центра основания."""
    half = int(radius * 2.4)
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
    y0, y1 = max(0, cy - int(radius * 2.8)), min(image.shape[0], cy)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = ~is_tile(image[y0:y1, x0:x1])
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return {"sw": int(xs.max() - xs.min() + 1),
            "sh": int(ys.max() - ys.min() + 1),
            "sa": int(mask.sum())}


def match(previous, current, limit=45):
    """Сопоставить пешки прошлого и текущего кадра (в координатах арены)."""
    used = set()
    pairs = {}
    for i, cur in enumerate(current):
        best, best_d = None, limit
        for j, old in enumerate(previous):
            if j in used:
                continue
            d = max(abs(old["x"] - cur["x"]), abs(old["y"] - cur["y"]))
            if d < best_d:
                best_d, best = d, j
        if best is not None:
            used.add(best)
            pairs[i] = best
    return pairs


def main() -> int:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/pvp.log")
    directory = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp")
    out_path = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/opencode/rank_track.csv")
    events = parse_log(log)

    frames = []
    for frame in directory.glob("b*.png"):
        m = re.match(r"b(\d+)_(\d+)\.png", frame.name)
        if m:
            frames.append((int(m.group(1)) + 1, int(m.group(2)), frame))
    frames.sort()

    known: dict[tuple[int, int], dict] = {}
    rows = []
    for round_no, step, frame in frames:
        arena, pawns = probe(frame)
        if not pawns:
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        prev = known.get((round_no, step - 1), {}).get("pawns", [])
        pairs = match(prev, pawns)
        for i, pawn in enumerate(pawns):
            if i in pairs:
                pawn["rank"] = prev[pairs[i]]["rank"]
            else:
                pawn["rank"] = 1     # призванная пешка
        # удачное слияние на прошлом шаге: цель получила ранг +1
        for event in events:
            if event["round"] != round_no or event["step"] != step - 1 or not event["ok"]:
                continue
            for pawn in pawns:
                if abs(pawn["px"] - event["dst"][0]) < 45 and \
                   abs(pawn["py"] - event["dst"][1]) < 45:
                    pawn["rank"] += 1
        known[(round_no, step)] = {"pawns": pawns}
        for pawn in pawns:
            shape = shape_of(image, pawn["px"], pawn["py"])
            feat = sprite_features(image, pawn["px"], pawn["py"], pawn["r"])
            if not shape or not feat:
                continue
            rows.append({
                "frame": frame.name, "round": round_no, "step": step,
                "x": pawn["x"], "y": pawn["y"], "r": pawn["r"],
                "rank": pawn["rank"],
                "round_pedestal": int(pawn["round"]),
                "fill": round(shape["fill"], 4),
                "stretch": round(shape["stretch"], 4),
                "area": shape["area"], "rmax": round(shape["rmax"], 1),
                "sw": feat["sw"], "sh": feat["sh"], "sa": feat["sa"],
            })
    if not rows:
        print("данных нет")
        return 1
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ranks: dict[int, int] = {}
    for row in rows:
        ranks[row["rank"]] = ranks.get(row["rank"], 0) + 1
    print(f"строк: {len(rows)}, рангов: {dict(sorted(ranks.items()))}, файл: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
