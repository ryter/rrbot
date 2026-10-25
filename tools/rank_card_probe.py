#!/usr/bin/env python3
"""Решающий опыт по рангу: карта, ранг и масштаб.

Зрение бота определяет номер карты юнита («свой knight карточка=2»). План:

  1. ждём двух своих юнитов одной карты рядом;
  2. сливаем их мышью;
  3. смотрим, что стоит на месте цели: та же карта (тогда ранг = масштаб)
     или другая (тогда ранг вообще не на карте);
  4. если та же карта — сравниваем размер спрайта до и после: во сколько
     раз вырос юнит.

    .venv/bin/python tools/rank_card_probe.py
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
WORK = Path("/tmp/opencode/rank_card")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as fh:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=fh)


def units(image):
    """Свои юниты: позиция, радиус основания, номер карты, имя карты."""
    path = WORK / "units.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "merge_probe"), str(path)])
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    bases = [(int(x), int(y), int(r)) for x, y, r in
             re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out)]
    vision = sh([str(CPP / "build" / "rrbot"), "--backend", "adb", "--frame", str(path),
                 "--vision"])
    cards = []
    for side, name, card, x, y in re.findall(
            r"(свой|чужой) (\w+) карточка=(\d+) \((\d+),(\d+)\)", vision):
        if side == "свой":
            cards.append((name, int(card), int(x), int(y)))
    if not arena:
        return None, [], []
    return (int(arena.group(1)), int(arena.group(2))), bases, cards


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


def sprite_height(image, cx, cy, radius):
    """Высота спрайта над основанием (без зелёной плитки)."""
    x0, x1 = max(0, cx - int(radius * 2)), min(image.shape[1], cx + int(radius * 2))
    y0, y1 = max(0, cy - int(radius * 2.6)), cy
    if x1 <= x0 or y1 <= y0:
        return None
    patch = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    tile = (hsv[..., 0] >= 26) & (hsv[..., 0] <= 56) & (hsv[..., 1] > 130) & (hsv[..., 2] > 110)
    ys, xs = np.nonzero(~tile)
    if len(xs) == 0:
        return None
    return int(ys.max() - ys.min() + 1)


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
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "140",
         "--rounds", "1", "--no-merge", "--record", str(WORK)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = WORK / "live.png"
    print("ждём двух соседей одной карты...")
    deadline = time.time() + 300
    choice = None
    while time.time() < deadline:
        time.sleep(2)
        shoot(live)
        image = cv2.imread(str(live))
        if image is None:
            continue
        arena, bases, cards = units(image)
        if not arena or not cards:
            continue
        best = None
        for i, a in enumerate(cards):
            for b in cards[i + 1:]:
                if a[1] != b[1] or a[0] != b[0]:
                    continue
                d = max(abs(a[2] - b[2]), abs(a[3] - b[3]))
                if d > 110:
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
    arena, bases, cards = units(image)
    _, a, b = choice
    if not cards:
        print("карты не определились на свежем кадре")
        proc.kill()
        return 1
    src = min(cards, key=lambda c: max(abs(c[2] - a[2]), abs(c[3] - a[3])))
    dst = min(cards, key=lambda c: max(abs(c[2] - b[2]), abs(c[3] - b[3])))
    base_src = min(bases, key=lambda p: max(abs(p[0] - src[2]), abs(p[1] - src[3])))
    base_dst = min(bases, key=lambda p: max(abs(p[0] - dst[2]), abs(p[1] - dst[3])))
    h_src = sprite_height(image, src[2], src[3], base_src[2])
    h_dst = sprite_height(image, dst[2], dst[3], base_dst[2])
    print(f"источник {src} основание r{base_src[2]} спрайт {h_src}")
    print(f"цель    {dst} основание r{base_dst[2]} спрайт {h_dst}")
    cv2.imwrite(str(WORK / "before.png"), image)
    mouse_drag(src[2], src[3], dst[2], dst[3])
    time.sleep(3)
    shoot(WORK / "after.png")
    after = cv2.imread(str(WORK / "after.png"))
    arena_after, bases_after, cards_after = units(after)
    shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
    tx, ty = dst[2] + shift[0], dst[3] + shift[1]
    near = min(cards_after, key=lambda c: max(abs(c[2] - tx), abs(c[3] - ty)),
               default=None)
    print(f"сдвиг {shift}")
    print(f"на месте цели ПОСЛЕ: {near} (было {dst})")
    if near:
        base_near = min(bases_after, key=lambda p: max(abs(p[0] - tx), abs(p[1] - ty)),
                        default=None)
        h_near = sprite_height(after, tx, ty, base_near[2] if base_near else 30)
        print(f"основание r{base_near[2] if base_near else '?'} спрайт {h_near}")
        if h_dst and h_near:
            print(f"во сколько раз вырос спрайт: {h_near / h_dst:.2f}")
    tiles = []
    for tag, img, point in (("ДО", image, (dst[2], dst[3])), ("ПОСЛЕ", after, (tx, ty))):
        c = crop(img, point[0], point[1])
        big = cv2.resize(c, (420, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(big)
    cv2.imwrite("/tmp/opencode/rank_card.png", np.hstack(tiles))
    print("монтаж: /tmp/opencode/rank_card.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
