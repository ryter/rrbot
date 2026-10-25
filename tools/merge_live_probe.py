#!/usr/bin/env python3
"""Живой опыт: что получается при слиянии двух одинаковых юнитов?

Бот играет без слияний, мышью сливаем двух одинаковых пешек и смотрим:
  * какой юнит стоит на месте цели (тот же тип или другой);
  * как он нарисован (крупнее? другая подставка?);
  * что у него с подписью над головой.

Снимки до и после — в /tmp/opencode/merge_live_*.png.

    .venv/bin/python tools/merge_live_probe.py
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
FRAMES = Path("/tmp/opencode/merge_live")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as fh:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=fh)


def probe(frame: Path):
    out = sh([str(CPP / "build" / "merge_probe"), str(frame)])
    arena = re.search(r"арена: своя (\d+),(\d+) (\d+)x(\d+)", out)
    pawns = []
    for side, x, y, r, shape in re.findall(
            r"(своя|чужая) (\d+),(\d+) радиус (\d+) (круглое|НЕ круглое)", out):
        if side == "своя":
            pawns.append((int(x), int(y), int(r), shape == "круглое"))
    origin = (int(arena.group(1)), int(arena.group(2))) if arena else (0, 0)
    return {"pawns": pawns, "origin": origin}


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


def crop(image, x, y, half=60):
    h, w = image.shape[:2]
    x0 = max(0, min(x - half, w - 2 * half))
    y0 = max(0, min(y - half, h - 2 * half))
    return image[y0:y0 + 2 * half, x0:x0 + 2 * half]


def best_pair(frame: Path):
    """Самая похожая пара пешек (по палитре) из вывода merge_probe."""
    out = sh([str(CPP / "build" / "merge_probe"), str(frame)])
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append((int(x), int(y), int(r)))
    pairs = []
    for src, dst, x1, y1, x2, y2, score in re.findall(
            r"\[(\d+)->(\d+)\] \((\d+),(\d+)\) -> \((\d+),(\d+)\): похожесть ([\d.]+)", out):
        pairs.append((float(score), (int(x1), int(y1)), (int(x2), int(y2))))
    if not pairs:
        return None, pawns
    pairs.sort(reverse=True)
    return pairs[0], pawns


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[b]uild/rrbot"], capture_output=True)
    time.sleep(1)
    FRAMES.mkdir(parents=True, exist_ok=True)
    log = (FRAMES / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "80",
         "--rounds", "1", "--no-merge", "--record", str(FRAMES)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = Path("/tmp/opencode/merge_live_now.png")
    print("ждём пешки на поле...")
    deadline = time.time() + 240
    best = None
    while time.time() < deadline:
        time.sleep(2)
        shoot(live)
        pair, pawns = best_pair(live)
        if pair and pair[0] >= 0.95 and len(pawns) >= 3:
            best = pair
            break
    if not best:
        print("одинаковой пары не дождались")
        proc.kill()
        return 1
    subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
    shoot(live)
    info = probe(live)
    score, src, dst = best
    print(f"похожесть {score:.3f}, сливаем {src} -> {dst}")
    source = min(info["pawns"], key=lambda p: max(abs(p[0] - src[0]), abs(p[1] - src[1])))
    target = min(info["pawns"], key=lambda p: max(abs(p[0] - dst[0]), abs(p[1] - dst[1])))
    src, dst = source[:2], target[:2]
    print(f"пешки: источник {source}, цель {target}")

    before = FRAMES / "before.png"
    after = FRAMES / "after.png"
    shoot(before)
    mouse_drag(source[0], source[1], target[0], target[1])
    time.sleep(3)
    shoot(after)
    ib = cv2.imread(str(before))
    ia = cv2.imread(str(after))
    info_after = probe(after)
    shift = (info_after["origin"][0] - info["origin"][0],
             info_after["origin"][1] - info["origin"][1])
    print("пешки после:", info_after["pawns"])
    print(f"сдвиг камеры: {shift}")

    tiles = []
    for tag, image, point in (("цель ДО", ib, target), ("цель ПОСЛЕ", ia, target),
                              ("источник ДО", ib, source)):
        c = crop(image, point[0], point[1])
        big = cv2.resize(c, (360, 360), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(big)
    cv2.imwrite("/tmp/opencode/merge_live_result.png", np.hstack(tiles))
    print("монтаж: /tmp/opencode/merge_live_result.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
