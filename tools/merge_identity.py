#!/usr/bin/env python3
"""Решающий опыт: та же карта после слияния или другая?

Берём двух своих юнитов с одинаковой подставкой (цвет кольца) и
одинаковой палитрой, сливаем их мышью и на месте цели сравниваем:

  * подставку (круг / квадрат / овал) — по точным координатам;
  * спрайт юнита (палитра тела) — та же карта или случайная из колоды.

    .venv/bin/python tools/merge_identity.py
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
WORK = Path("/tmp/opencode/merge_identity")
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


def body_palette(image, cx, cy, radius):
    """Палитра тела юнита (круг над основанием, без подставки)."""
    inner = max(4, radius - 8)
    patch = image[max(0, cy - 2 * radius):cy, max(0, cx - radius):cx + radius]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = np.zeros(HIST_H * HIST_S * HIST_V, np.float32)
    cyy, cxx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    # верхние 60 % патча — тело юнита
    mask = (cxx - patch.shape[1] // 2) ** 2 + (cyy - patch.shape[0]) ** 2 <= inner * inner
    for pixel in hsv[mask]:
        h = min(HIST_H - 1, int(pixel[0]) * HIST_H // 180)
        s = min(HIST_S - 1, int(pixel[1]) * HIST_S // 256)
        v = min(HIST_V - 1, int(pixel[2]) * HIST_V // 256)
        hist[h * HIST_S * HIST_V + s * HIST_V + v] += 1
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def base_palette(image, cx, cy, radius):
    inner = max(4, radius - 3)
    patch = image[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
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


def crop(image, x, y, half=70):
    h, w = image.shape[:2]
    x0 = max(0, min(x - half, w - 2 * half))
    y0 = max(0, min(y - half, h - 2 * half))
    return image[y0:y0 + 2 * half, x0:x0 + 2 * half]


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
            pa = base_palette(image, a["x"], a["y"], a["r"])
            ca = ring_color(image, a)
            for b in pawns[i + 1:]:
                d = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
                if d > 110:
                    continue
                pb = base_palette(image, b["x"], b["y"], b["r"])
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
    body_before = body_palette(image, target["x"], target["y"], target["r"])
    base_before = base_palette(image, target["x"], target["y"], target["r"])
    print(f"сливаем {source['x']},{source['y']} -> {target['x']},{target['y']}")
    cv2.imwrite(str(WORK / "before.png"), image)
    mouse_drag(source["x"], source["y"], target["x"], target["y"])
    time.sleep(3)
    shoot(WORK / "after.png")
    after = cv2.imread(str(WORK / "after.png"))
    arena_after, _ = probe(after)
    shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
    tx, ty = target["x"] + shift[0], target["y"] + shift[1]
    body_after = body_palette(after, tx, ty, target["r"])
    base_after = base_palette(after, tx, ty, target["r"])
    print(f"тело:  до -> после, похожесть {cosine(body_before, body_after):.3f}")
    print(f"подставка: до -> после, похожесть {cosine(base_before, base_after):.3f}")
    tiles = []
    for tag, img, point in (("ДО", image, (target["x"], target["y"])),
                            ("ПОСЛЕ", after, (tx, ty))):
        c = crop(img, point[0], point[1])
        big = cv2.resize(c, (420, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(big)
    cv2.imwrite("/tmp/opencode/merge_identity.png", np.hstack(tiles))
    print("монтаж: /tmp/opencode/merge_identity.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
