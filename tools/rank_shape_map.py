#!/usr/bin/env python3
"""Карта рангов: как выглядит подставка юнита ранга 1, 2, 3.

Опыт: бот только призывает. Мы берём двух юнитов одной карты (по цвету
подставки и палитре) и сливаем их. Затем берём ещё одного такого же и
сливаем с результатом — получаем следующий ранг.

Форму основания меряем по точным координатам (детектор пешек ищет только
круги, поэтому слитые юниты в его список не попадают).

    .venv/bin/python tools/rank_shape_map.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/lennistyle/rushroyale-lab")
CPP = ROOT / "cpp"
WORK = Path("/tmp/opencode/rank_shape_map")

sys.path.insert(0, str(ROOT / "tools"))
from pedestal_shape import shape_of  # noqa: E402

HIST_H, HIST_S, HIST_V = 18, 4, 4


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as fh:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=fh)


def probe(image):
    path = WORK / "probe.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "merge_probe"), str(path)])
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append({"x": int(x), "y": int(y), "r": int(r)})
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2))), pawns


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
        return 0.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def ring_color(image, pawn):
    ring = []
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = int(pawn["x"] + (pawn["r"] - 5) * np.cos(angle))
        y = int(pawn["y"] + (pawn["r"] - 5) * np.sin(angle))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            ring.append(image[y, x])
    if not ring:
        return None
    return np.median(np.array(ring), axis=0)


def window_origin():
    out = sh(["hyprctl", "clients"])
    for block in out.split("Window "):
        if "waydroid.com.my.defense" not in block:
            continue
        at = re.search(r"at:\s*(-?\d+),(-?\d+)", block)
        if at:
            return int(at.group(1)), int(at.group(2))
    return 0, 0


def mouse_drag(x1, y1, x2, y2):
    sys.path.insert(0, str(ROOT))
    from rrbot.input import UinputMouseBackend
    ox, oy = window_origin()
    mouse = UinputMouseBackend()
    mouse.drag(ox + x1, oy + y1, ox + x2, oy + y2, steps=12, press_ms=150,
               step_ms=40, hold_ms=200)
    mouse.close()


def kind(shape):
    """Грубый вид подставки."""
    if not shape:
        return "нет"
    if shape["fill"] >= 0.88 and shape["stretch"] <= 1.12:
        return "круг"
    if shape["fill"] >= 0.72:
        return "скруглённый квадрат"
    return "овал/капля"


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[b]uild/rrbot"], capture_output=True)
    time.sleep(1)
    WORK.mkdir(parents=True, exist_ok=True)
    log = (WORK / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "200",
         "--rounds", "1", "--no-merge", "--record", str(WORK)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = WORK / "live.png"
    results = []
    for rank in (1, 2, 3):
        print(f"\n=== ищем двух юнитов одной карты (для ранга {rank}) ===")
        deadline = time.time() + 180
        choice = None
        while time.time() < deadline:
            time.sleep(2)
            shoot(live)
            image = cv2.imread(str(live))
            if image is None:
                continue
            arena, pawns = probe(image)
            if not arena or len(pawns) < 2:
                continue
            best = None
            for i, a in enumerate(pawns):
                pa = palette(image, a["x"], a["y"], a["r"])
                ca = ring_color(image, a)
                for b in pawns[i + 1:]:
                    d = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
                    if d > 110:
                        continue
                    pb = palette(image, b["x"], b["y"], b["r"])
                    cb = ring_color(image, b)
                    if ca is None or cb is None:
                        continue
                    if float(np.linalg.norm(ca - cb)) > 30 or cosine(pa, pb) < 0.97:
                        continue
                    if best is None or d < best[0]:
                        best = (d, a, b)
            if best:
                choice = best
                break
        if not choice:
            print("не дождались")
            break
        subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
        time.sleep(0.5)
        shoot(live)
        image = cv2.imread(str(live))
        arena, pawns = probe(image)
        _, a, b = choice
        source = min(pawns, key=lambda p: max(abs(p["x"] - a["x"]), abs(p["y"] - a["y"])))
        target = min(pawns, key=lambda p: max(abs(p["x"] - b["x"]), abs(p["y"] - b["y"])))
        shape_before = shape_of(image, target["x"], target["y"])
        print(f"цель {target['x']},{target['y']}: ДО {kind(shape_before)} "
              f"(fill {shape_before['fill']:.3f} stretch {shape_before['stretch']:.2f})")
        mouse_drag(source["x"], source["y"], target["x"], target["y"])
        time.sleep(3)
        shoot(WORK / f"rank{rank}_after.png")
        after = cv2.imread(str(WORK / f"rank{rank}_after.png"))
        arena_after, _ = probe(after)
        shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
        tx, ty = target["x"] + shift[0], target["y"] + shift[1]
        shape_after = shape_of(after, tx, ty)
        print(f"ПОСЛЕ {kind(shape_after)} "
              f"(fill {shape_after['fill']:.3f} stretch {shape_after['stretch']:.2f})"
              if shape_after else "ПОСЛЕ основание не найдено")
        results.append((rank, kind(shape_before), kind(shape_after)))
        cv2.imwrite(str(WORK / f"rank{rank}_before.png"), image)
        subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
        time.sleep(1)
    print("\nитог:")
    for rank, before, after in results:
        print(f"  ранг {rank}: было {before} -> стало {after}")
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
