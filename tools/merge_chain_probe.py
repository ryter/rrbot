#!/usr/bin/env python3
"""Слияние по цепочке: как выглядит ранг 1, 2, 3.

Бот только призывает. Мы дважды подряд сливаем пешку с одинаковыми: первый
раз получаем ранг 2, второй — ранг 3. Снимки позволяют увидеть, как меняется
подставка (круг → скруглённый квадрат → ?).

    .venv/bin/python tools/merge_chain_probe.py
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
WORK = Path("/tmp/opencode/merge_chain")


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
    if not arena:
        return None, []
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append({"x": int(x), "y": int(y), "r": int(r)})
    return (int(arena.group(1)), int(arena.group(2))), pawns


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


def find_twins(image):
    """Две пешки с одинаковой подставкой (по цвету кольца) в соседних клетках."""
    arena, pawns = probe(image)
    if not arena or len(pawns) < 2:
        return None
    best = None
    for i, a in enumerate(pawns):
        for b in pawns[i + 1:]:
            d = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
            if d > 110:
                continue
            ca = ring_color(image, a)
            cb = ring_color(image, b)
            if ca is None or cb is None:
                continue
            if float(np.linalg.norm(ca - cb)) > 30:
                continue
            if best is None or d < best[0]:
                best = (d, a, b)
    return best


def ring_color(image, pawn):
    ring = []
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = int(pawn["x"] + (pawn["r"] - 4) * np.cos(angle))
        y = int(pawn["y"] + (pawn["r"] - 4) * np.sin(angle))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            ring.append(image[y, x])
    if not ring:
        return None
    return np.median(np.array(ring), axis=0)


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[b]uild/rrbot"], capture_output=True)
    time.sleep(1)
    WORK.mkdir(parents=True, exist_ok=True)
    log = (WORK / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "120",
         "--rounds", "1", "--no-merge", "--record", str(WORK)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = WORK / "live.png"
    print("ждём одинаковых соседей...")
    deadline = time.time() + 300
    twins = None
    while time.time() < deadline:
        time.sleep(2)
        shoot(live)
        image = cv2.imread(str(live))
        if image is None:
            continue
        twins = find_twins(image)
        if twins:
            break
    if not twins:
        print("не дождались")
        proc.kill()
        return 1
    subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
    time.sleep(0.5)

    shots = []
    for step in range(3):
        shoot(live)
        image = cv2.imread(str(live))
        arena, pawns = probe(image)
        pair = find_twins(image)
        if not pair:
            print(f"шаг {step}: пары нет")
            break
        _, a, b = pair
        # тянем дальнего в ближнего к базе
        source, target = (a, b) if a["y"] > b["y"] else (b, a)
        cv2.imwrite(str(WORK / f"chain_{step}_before.png"), image)
        print(f"шаг {step}: {source['x']},{source['y']} r{source['r']} -> "
              f"{target['x']},{target['y']} r{target['r']}")
        mouse_drag(source["x"], source["y"], target["x"], target["y"])
        time.sleep(3)
        shoot(WORK / f"chain_{step}_after.png")
        after = cv2.imread(str(WORK / f"chain_{step}_after.png"))
        shots.append(("ДО", image, target))
        shots.append(("ПОСЛЕ", after, target))
    if not shots:
        proc.kill()
        return 1
    tiles = []
    for tag, img, point in shots:
        c = crop(img, point["x"], point["y"])
        big = cv2.resize(c, (330, 330), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(big)
    cols = 2
    rows = (len(tiles) + 1) // 2
    sheet = np.zeros((rows * 330, cols * 330, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 330:(r + 1) * 330, c * 330:(c + 1) * 330] = tile
    cv2.imwrite("/tmp/opencode/merge_chain.png", sheet)
    print("монтаж: /tmp/opencode/merge_chain.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
