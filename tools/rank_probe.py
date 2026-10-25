#!/usr/bin/env python3
"""Ранг юнита: опыты «что игра принимает» и как выглядит ранг.

Замер (24.10): подставка (основание) кодирует **ранг**, а не тип карты:

* ранг 1 — круглая подставка (заполненность ~0.87..0.96);
* ранг 2 — подставка «капля» (заполненность ~0.75..0.85);
* ранг 3 — подставка «щит» (заполненность ~0.65..0.72).

Один и тот же спрайт встречается и с круглой, и с каплеобразной подставкой —
значит, форма зависит от ранга, а не от карты. Слияние на месте цели меняет
подставку: было 0.94 -> стало 0.76.

Скрипт делает живые опыты мышью:

  1. 1 + 1 -> ранг 2 (как выглядит результат);
  2. 2 + 1 -> принимает ли игра;
  3. 2 + 2 -> принимает ли игра и как выглядит ранг 3.

    .venv/bin/python tools/rank_probe.py
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
WORK = Path("/tmp/opencode/rank_probe")

# Заполненность подставки -> ранг.
RANK1_MIN = 0.86
RANK2_MIN = 0.70


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> bool:
    with path.open("wb") as fh:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=fh)
    return path.stat().st_size > 1000


def probe(image):
    """Пешки своей половины: координаты, радиусы, заполненность."""
    path = WORK / "probe.png"
    cv2.imwrite(str(path), image)
    out = sh([str(CPP / "build" / "merge_probe"), str(path)])
    arena = re.search(r"арена: своя (-?\d+),(-?\d+) (\d+)x(\d+)", out)
    pawns = [
        {"x": int(x), "y": int(y), "r": int(r), "fill": float(f)}
        for x, y, r, f in re.findall(
            r"  своя (\d+),(\d+) радиус (\d+).*?заполненность ([\d.]+)", out)
    ]
    if not arena:
        return None, []
    return (int(arena.group(1)), int(arena.group(2)), int(arena.group(3))), pawns


def lattice(path):
    """Шаг и начало сетки клеток (для выравнивания кадров)."""
    out = sh([str(CPP / "build" / "board_probe"), path])
    m = re.search(r"шаг ([\d.]+), начало \(([-\d.]+),([-\d.]+)\)", out)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def rank_of(fill):
    if fill >= RANK1_MIN:
        return 1
    if fill >= RANK2_MIN:
        return 2
    return 3


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


def cell_center(lat, cx, cy):
    sp, px, py = lat
    return (px + round((cx - px) / sp) * sp, py + round((cy - py) / sp) * sp)


def nearest(pawns, point, radius=45):
    best = None
    for pawn in pawns:
        d = (pawn["x"] - point[0]) ** 2 + (pawn["y"] - point[1]) ** 2
        if d < radius * radius and (best is None or d < best[0]):
            best = (d, pawn)
    return None if best is None else best[1]


def wait_units(deadline=240):
    """Ждём, пока на своей половине есть пешки (и снимаем кадр)."""
    live = WORK / "live.png"
    while time.time() < deadline:
        time.sleep(2)
        if not shoot(live):
            continue
        image = cv2.imread(str(live))
        if image is None:
            continue
        arena, pawns = probe(image)
        if arena and pawns:
            return image, arena, pawns
    return None, None, None


def same_card(image, a, b):
    if abs(a["x"] - b["x"]) > 110 or abs(a["y"] - b["y"]) > 110:
        return False
    ca, cb = ring_color(image, a), ring_color(image, b)
    if ca is None or cb is None:
        return False
    return float(np.linalg.norm(ca - cb)) <= 30


def merge(proc, label, source, target, before_path, after_path):
    """Сливает source -> target и измеряет, что стало на месте цели."""
    subprocess.run(["kill", "-STOP", str(proc.pid)], capture_output=True)
    time.sleep(0.5)
    print(f"{label}: {source['x']},{source['y']} (ранг {rank_of(source['fill'])}, "
          f"заполненность {source['fill']:.2f}) -> "
          f"{target['x']},{target['y']} (ранг {rank_of(target['fill'])}, "
          f"заполненность {target['fill']:.2f})")
    mouse_drag(source["x"], source["y"], target["x"], target["y"])
    time.sleep(3)
    shoot(after_path)
    subprocess.run(["kill", "-CONT", str(proc.pid)], capture_output=True)
    after = cv2.imread(str(after_path))
    if after is None:
        print("   кадр после не снят")
        return None
    lat_after = lattice(after_path)
    if not lat_after:
        print("   сетка не найдена")
        return None
    point = cell_center(lat_after, target["x"], target["y"])
    _, pawns_after = probe(after)
    result = nearest(pawns_after, point)
    # Исчез ли источник?
    lat_before = lattice(before_path)
    source_point = cell_center(lat_before, source["x"], source["y"]) if lat_before else None
    source_gone = True
    if source_point:
        found = nearest(pawns_after, source_point)
        source_gone = found is None
    print(f"   на месте цели: "
          + (f"ранг {rank_of(result['fill'])}, заполненность {result['fill']:.2f}"
             if result else "ничего не найдено")
          + f"; источник {'исчез (слияние вышло)' if source_gone else 'остался (не вышло)'}")
    return {"point": point, "rank": rank_of(result["fill"]) if result else None,
            "gone": source_gone, "result": result}


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[b]uild/rrbot"], capture_output=True)
    time.sleep(1)
    WORK.mkdir(parents=True, exist_ok=True)
    log = (WORK / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot"), "--backend", "emulator", "--pvp", "--steps", "300",
         "--rounds", "1", "--no-merge", "--record", str(WORK)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    try:
        # --- опыт 1: два первых ранга -> ранг 2
        pair = None
        deadline = time.time() + 300
        while time.time() < deadline and pair is None:
            time.sleep(2)
            live = WORK / "live.png"
            if not shoot(live):
                continue
            image = cv2.imread(str(live))
            if image is None:
                continue
            arena, pawns = probe(image)
            if not arena:
                continue
            for i, a in enumerate(pawns):
                if rank_of(a["fill"]) != 1:
                    continue
                for b in pawns[i + 1:]:
                    if rank_of(b["fill"]) != 1:
                        continue
                    if same_card(image, a, b):
                        pair = (image, a, b)
                        break
                if pair:
                    break
        if not pair:
            print("не дождались пары первых рангов")
            return 1
        image, a, b = pair
        cv2.imwrite(str(WORK / "exp1_before.png"), image)
        source, target = (a, b) if a["y"] > b["y"] else (b, a)
        first = merge(proc, "1+1", source, target, WORK / "exp1_before.png",
                      WORK / "exp1_after.png")
        if not first:
            return 1

        # --- опыт 2: второй ранг + первый
        deadline = time.time() + 300
        done = False
        while time.time() < deadline and not done:
            time.sleep(2)
            live = WORK / "live.png"
            if not shoot(live):
                continue
            image = cv2.imread(str(live))
            if image is None:
                continue
            arena, pawns = probe(image)
            if not arena:
                continue
            lat = lattice(live)
            if not lat:
                continue
            rank2 = nearest(pawns, first["point"])
            if rank2 is None or rank_of(rank2["fill"]) == 1:
                continue
            firsts = [p for p in pawns if rank_of(p["fill"]) == 1]
            if not firsts:
                continue
            other = max(firsts, key=lambda p: (p["x"] - rank2["x"]) ** 2 +
                        (p["y"] - rank2["y"]) ** 2)
            cv2.imwrite(str(WORK / "exp2_before.png"), image)
            merge(proc, "2+1", rank2, other, WORK / "exp2_before.png",
                  WORK / "exp2_after.png")
            done = True

        # --- опыт 3: второй ранг + второй ранг
        deadline = time.time() + 300
        done = False
        while time.time() < deadline and not done:
            time.sleep(2)
            live = WORK / "live.png"
            if not shoot(live):
                continue
            image = cv2.imread(str(live))
            if image is None:
                continue
            arena, pawns = probe(image)
            if not arena:
                continue
            rank2s = [p for p in pawns if rank_of(p["fill"]) != 1]
            if len(rank2s) < 2:
                continue
            a, b = rank2s[0], rank2s[1]
            cv2.imwrite(str(WORK / "exp3_before.png"), image)
            merge(proc, "2+2", a, b, WORK / "exp3_before.png", WORK / "exp3_after.png")
            done = True
        return 0
    finally:
        proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
