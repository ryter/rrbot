#!/usr/bin/env python3
"""Чистый опыт: слияние пары и что стало с полем.

Показывает ВСЮ свою половину до и после слияния (сетка поверх), чтобы
видеть всё: исчез ли источник, что стоит на месте цели, изменилась ли
форма основания у той же карты.

Все координаты — из merge_probe (они в пикселях кадра 941×1030), так что
путаницы систем координат не будет.

    .venv/bin/python tools/merge_watch.py
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
WORK = Path("/tmp/opencode/merge_watch")
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
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) (\d+)x(\d+)", out)
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append({"x": int(x), "y": int(y), "r": int(r)})
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2)), int(arena.group(3))), pawns


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


def half(image, arena, scale=1):
    ax, ay, aw = arena
    return image[ay:ay + 260, ax:ax + aw]


def draw_half(image, arena, pawns, title):
    crop = half(image, arena).copy()
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                      interpolation=cv2.INTER_CUBIC)
    ax, ay, aw = arena
    for pawn in pawns:
        x = (pawn["x"] - ax) * 2
        y = (pawn["y"] - ay) * 2
        if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
            cv2.drawMarker(crop, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(crop, title, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return crop


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[b]uild/rrbot"], capture_output=True)
    time.sleep(1)
    WORK.mkdir(parents=True, exist_ok=True)
    log = (WORK / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "160",
         "--rounds", "1", "--no-merge", "--record", str(WORK)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = WORK / "live.png"
    print("ждём двух соседей с одинаковой подставкой...")
    deadline = time.time() + 300
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
        proc.kill()
        return 1
    subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
    time.sleep(0.5)
    shoot(live)
    image = cv2.imread(str(live))
    arena, pawns = probe(image)
    _, a, b = choice
    source = min(pawns, key=lambda p: max(abs(p["x"] - a["x"]), abs(p["y"] - a["y"])))
    target = min(pawns, key=lambda p: max(abs(p["x"] - b["x"]), abs(p["y"] - b["y"])))
    print(f"сливаем {source['x']},{source['y']} -> {target['x']},{target['y']}")
    cv2.imwrite(str(WORK / "before.png"), image)
    top = draw_half(image, arena, pawns, "ДО")
    mouse_drag(source["x"], source["y"], target["x"], target["y"])
    time.sleep(3)
    shoot(WORK / "after.png")
    after = cv2.imread(str(WORK / "after.png"))
    arena_after, pawns_after = probe(after)
    shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
    bottom = draw_half(after, arena_after, pawns_after, "ПОСЛЕ")
    print(f"сдвиг {shift}")
    print("пешки ДО:   ", [(p["x"], p["y"], p["r"]) for p in pawns])
    print("пешки ПОСЛЕ:", [(p["x"], p["y"], p["r"]) for p in pawns_after])
    height = max(top.shape[0], bottom.shape[0])
    sheet = np.zeros((height * 2, top.shape[1], 3), np.uint8)
    sheet[:top.shape[0]] = top
    sheet[height:height + bottom.shape[0]] = bottom
    cv2.imwrite("/tmp/opencode/merge_watch.png", sheet)
    print("монтаж: /tmp/opencode/merge_watch.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
