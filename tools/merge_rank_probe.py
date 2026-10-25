#!/usr/bin/env python3
"""Что происходит при слиянии: та же карта или другая, и как меняется ранг.

Бот только призывает. Мы ищем двух соседей одной карты (карта определяется
по эталону, tools: cpp/build/rrbot --frame ... --vision), сливаем их мышью и
сравниваем карту и радиус основания на месте цели до и после.

    .venv/bin/python tools/merge_rank_probe.py
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
WORK = Path("/tmp/opencode/merge_rank")


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


def cards(image):
    """Карта юнита в каждой точке поля (по эталонам)."""
    path = WORK / "cards.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "rrbot"), "--backend", "adb", "--frame", str(path),
              "--vision"])
    result = []
    for side, name, x, y in re.findall(r"(свой|чужой) (\w+) карточка=\d+ \((\d+),(\d+)\)", out):
        result.append((side == "свой", name, int(x), int(y)))
    return result


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


def card_at(card_list, x, y, limit=45):
    best, best_d = None, limit
    for ours, name, cx, cy in card_list:
        if not ours:
            continue
        d = max(abs(cx - x), abs(cy - y))
        if d < best_d:
            best_d, best = d, name
    return best


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
    print("ждём двух соседей одной карты...")
    deadline = time.time() + 300
    pair = None
    while time.time() < deadline:
        time.sleep(2)
        shoot(live)
        image = cv2.imread(str(live))
        if image is None:
            continue
        arena, pawns = probe(image)
        card_list = cards(image)
        if not arena or len(pawns) < 2:
            continue
        best = None
        for i, a in enumerate(pawns):
            for b in pawns[i + 1:]:
                d = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
                if d > 110:
                    continue
                ca = card_at(card_list, a["x"], a["y"])
                cb = card_at(card_list, b["x"], b["y"])
                if ca and ca == cb:
                    if best is None or d < best[0]:
                        best = (d, a, b, ca)
        if best:
            pair = best
            break
    if not pair:
        print("не дождались")
        proc.kill()
        return 1
    subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
    time.sleep(0.5)
    shoot(live)
    image = cv2.imread(str(live))
    arena, pawns = probe(image)
    card_list = cards(image)
    _, a, b, name = pair
    source = min(pawns, key=lambda p: max(abs(p["x"] - a["x"]), abs(p["y"] - a["y"])))
    target = min(pawns, key=lambda p: max(abs(p["x"] - b["x"]), abs(p["y"] - b["y"])))
    print(f"карта {name}: источник {source['x']},{source['y']} r{source['r']}, "
          f"цель {target['x']},{target['y']} r{target['r']}")
    cv2.imwrite(str(WORK / "before.png"), image)
    mouse_drag(source["x"], source["y"], target["x"], target["y"])
    time.sleep(3)
    shoot(WORK / "after.png")
    after = cv2.imread(str(WORK / "after.png"))
    arena_after, pawns_after = probe(after)
    shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
    cards_after = cards(after)
    tx, ty = target["x"] + shift[0], target["y"] + shift[1]
    print(f"сдвиг камеры {shift}")
    print(f"карта на месте цели: ДО {card_at(card_list, target['x'], target['y'])}, "
          f"ПОСЛЕ {card_at(cards_after, tx, ty)}")
    print("пешки после:", [(p["x"], p["y"], p["r"]) for p in pawns_after])
    tiles = []
    for tag, img, point in (("ДО", image, target), ("ПОСЛЕ", after, (tx, ty))):
        c = crop(img, point["x"], point["y"])
        big = cv2.resize(c, (420, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(big)
    cv2.imwrite("/tmp/opencode/merge_rank.png", np.hstack(tiles))
    print("монтаж: /tmp/opencode/merge_rank.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
