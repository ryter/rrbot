#!/usr/bin/env python3
"""Одна и та же карта на ранге 1 и ранге 2 — есть ли разница?

План:
  1. ждём двух своих юнитов одной карты рядом (карта — по зрению бота);
  2. сливаем их мышью: получаем юнита старшего ранга;
  3. берём ещё одного такого же юнита и пробуем слировать его с результатом.

Если игра откажет (разные ранги) — ранг в поле имеет какой-то признак.
Снимки и замеры формы/размера покажут, какой.

    .venv/bin/python tools/rank_same_card.py
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
WORK = Path("/tmp/opencode/rank_same_card")

sys.path.insert(0, str(ROOT / "tools"))
from pedestal_shape import shape_of  # noqa: E402


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as fh:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=fh)


def bases(image):
    path = WORK / "bases.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "merge_probe"), str(path)])
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) ", out)
    pawns = []
    for x, y, r in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns.append({"x": int(x), "y": int(y), "r": int(r)})
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2))), pawns


def cards(image):
    """Свои юниты по зрению: имя карты и позиция."""
    path = WORK / "cards.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "rrbot"), "--backend", "adb", "--frame", str(path),
              "--vision"])
    found = []
    for side, name, x, y in re.findall(
            r"(свой|чужой) (\w+) карточка=\d+ \((\d+),(\d+)\)", out):
        if side == "свой":
            found.append((name, int(x), int(y)))
    return found


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


def describe(image, x, y, r):
    shape = shape_of(image, x, y)
    if not shape:
        return "основание не найдено"
    return (f"r{r} fill {shape['fill']:.3f} stretch {shape['stretch']:.2f} "
            f"area {shape['area']}")


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
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "160",
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
        arena, pawns = bases(image)
        card_list = cards(image)
        if not arena or not pawns or len(card_list) < 2:
            continue
        best = None
        for i, a in enumerate(card_list):
            for b in card_list[i + 1:]:
                if a[0] != b[0]:
                    continue
                d = max(abs(a[1] - b[1]), abs(a[2] - b[2]))
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
    arena, pawns = bases(image)
    card_list = cards(image)
    if not card_list:
        print("карты не определились")
        proc.kill()
        return 1
    _, a, b = choice
    src = min(card_list, key=lambda c: max(abs(c[1] - a[1]), abs(c[2] - a[2])))
    dst = min(card_list, key=lambda c: max(abs(c[1] - b[1]), abs(c[2] - b[2])))
    base_dst = min(pawns, key=lambda p: max(abs(p["x"] - dst[1]), abs(p["y"] - dst[2])))
    print(f"карта {src[0]}: сливаем {src[1]},{src[2]} -> {dst[1]},{dst[2]}")
    print(f"цель ДО: {describe(image, dst[1], dst[2], base_dst['r'])}")
    cv2.imwrite(str(WORK / "before.png"), image)
    mouse_drag(src[1], src[2], dst[1], dst[2])
    time.sleep(3)
    shoot(WORK / "after.png")
    after = cv2.imread(str(WORK / "after.png"))
    arena_after, pawns_after = bases(after)
    shift = (arena_after[0] - arena[0], arena_after[1] - arena[1]) if arena_after else (0, 0)
    tx, ty = dst[1] + shift[0], dst[2] + shift[1]
    print(f"сдвиг {shift}")
    print(f"цель ПОСЛЕ: {describe(after, tx, ty, base_dst['r'])}")
    cards_after = cards(after)
    near = min(cards_after, key=lambda c: max(abs(c[1] - tx), abs(c[2] - ty)),
               default=None)
    print(f"карта на месте цели ПОСЛЕ: {near} (было {dst[0]})")

    # ищем ещё одного такого же и пробуем слировать его с результатом
    third = None
    for c in cards_after:
        if near and c[0] == near[0] and max(abs(c[1] - tx), abs(c[2] - ty)) > 60:
            third = c
            break
    if third is None:
        print("третьего такого же юнита нет")
    else:
        print(f"третий {third[0]} в {third[1]},{third[2]}: пробуем слить с результатом")
        mouse_drag(third[1], third[2], tx, ty)
        time.sleep(3)
        shoot(WORK / "after2.png")
        after2 = cv2.imread(str(WORK / "after2.png"))
        arena2, pawns2 = bases(after2)
        shift2 = (arena2[0] - arena_after[0], arena2[1] - arena_after[1]) if arena2 else (0, 0)
        tx2, ty2 = tx + shift2[0], ty + shift2[1]
        print(f"цель после второй попытки: {describe(after2, tx2, ty2, base_dst['r'])}")
        print(f"пешки после: {[(p['x'], p['y'], p['r']) for p in pawns2]}")
        cv2.imwrite(str(WORK / "after2.png"), after2)
    tiles = []
    for tag, img, point in (("ДО", image, (dst[1], dst[2])), ("ПОСЛЕ", after, (tx, ty))):
        c = crop(img, point[0], point[1])
        big = cv2.resize(c, (420, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(big, tag, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(big)
    cv2.imwrite("/tmp/opencode/rank_same_card.png", np.hstack(tiles))
    print("монтаж: /tmp/opencode/rank_same_card.png")
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
