#!/usr/bin/env python3
"""Форма основания: ранг 1 (чистый набор) против результатов слияния.

Чистый набор: бой без слияний — на поле только призванные юниты, все
ранга 1. Набор слитых: результаты успешных слияний (ранг >= 2).

Для каждой пешки считаем признаки подставки (заполненность, вытянутость,
площадь) и сравниваем распределения.

    .venv/bin/python tools/rank_clean_compare.py <каталог без слияний> <лог со слияниями> <каталог со слияниями>
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
    out = subprocess.run([str(CPP / "build" / "merge_probe"), str(frame)],
                         capture_output=True, text=True).stdout
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append((int(x), int(y), int(r)))
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2))), pawns


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
                       "dst": (int(m.group(3)), int(m.group(4))),
                       "src": (int(m.group(1)), int(m.group(2))),
                       "ok": None}
        if pending and ("слияние вышло" in line or "слияние не вышло" in line):
            pending["ok"] = "слияние вышло" in line
            events.append(pending)
            pending = None
    return events


def collect(directory: Path, events=None):
    """Признаки пешек. Если events заданы — только пешки на месте цели
    успешного слияния (это слитые юниты)."""
    result = []
    for frame in sorted(directory.glob("b*.png")):
        m = re.match(r"b(\d+)_(\d+)\.png", frame.name)
        if not m:
            continue
        round_no = int(m.group(1)) + 1
        step = int(m.group(2))
        arena, pawns = pawns_of(frame)
        if not arena or not pawns:
            continue
        image = cv2.imread(str(frame))
        if image is None:
            continue
        wanted = None
        if events is not None:
            wanted = [e["dst"] for e in events
                      if e["round"] == round_no and e["step"] == step - 1 and e["ok"]]
            if not wanted:
                continue
        for x, y, r in pawns:
            if wanted is not None:
                if not any(max(abs(x - dx), abs(y - dy)) <= 45 for dx, dy in wanted):
                    continue
            shape = shape_of(image, x, y)
            if not shape:
                continue
            result.append({"fill": shape["fill"], "stretch": shape["stretch"],
                           "area": shape["area"], "rmax": shape["rmax"], "r": r})
    return result


def report(name, rows):
    print(f"\n{name}: пешек {len(rows)}")
    for key in ("fill", "stretch", "area", "r"):
        values = np.array([row[key] for row in rows])
        print(f"  {key:8}: мин {values.min():8.3f} 25% {np.percentile(values, 25):8.3f} "
              f"медиана {np.median(values):8.3f} 75% {np.percentile(values, 75):8.3f} "
              f"макс {values.max():8.3f}")
    circles = sum(1 for row in rows if row["fill"] >= 0.88)
    print(f"  круглых оснований (fill >= 0.88): {circles} из {len(rows)} "
          f"({circles / max(1, len(rows)) * 100:.0f} %)")


def auc_of(pos, neg):
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    clean_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rank1")
    log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/pvp.log")
    merged_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/opencode/pvp")
    events = parse_log(log)
    print(f"успешных слияний в логе: {sum(1 for e in events if e['ok'])}")
    clean = collect(clean_dir)
    merged = collect(merged_dir, events)
    report("чистый набор (все пешки — ранг 1)", clean)
    report("слитые юниты (ранг >= 2)", merged)
    if clean and merged:
        print("\nAUC по каждому признаку (0.5 = не различает):")
        for key in ("fill", "stretch", "area", "r"):
            pos = np.array([row[key] for row in merged])
            neg = np.array([row[key] for row in clean])
            print(f"  {key:8}: {auc_of(pos, neg):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
