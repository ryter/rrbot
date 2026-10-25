#!/usr/bin/env python3
"""Куда надо отпускать пешку при слиянии.

Слияние не выходит примерно в трети попыток. Одна из причин может быть
в том, что палец отпускают на основании юнита, а не на самом юните:
основание — это круг под ногами, а тело нарисовано выше. Игра ждёт отпускания
над юнитом.

Скрипт на живом бою находит пару одинаковых соседних пешек и пробует
перетаскивание в разные точки:

    основание (как сейчас в боте),
    тело (на 30 px выше основания),
    чуть ниже основания (контроль).

Результат: исчезла ли пешка-источник.

Запуск:
    .venv/bin/python tools/merge_drop_point.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path("/home/lennistyle/rushroyale-lab")
CPP = ROOT / "cpp"
LIVE = Path("/tmp/opencode/drop_live.png")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as subprocess_out:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=subprocess_out)


def probe(frame: Path) -> dict:
    out = sh([str(CPP / "build" / "merge_probe"), str(frame)])
    pawns = {}
    for x, y, radius in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns[(int(x), int(y))] = int(radius)
    pairs = []
    for a, b, c, d, score in re.findall(
            r"\((\d+),(\d+)\) -> \((\d+),(\d+)\): похожесть ([\d.]+)", out):
        pairs.append((int(a), int(b), int(c), int(d), float(score)))
    return {"pawns": pawns, "pairs": pairs}


def waydroid_window() -> tuple[int, int]:
    out = subprocess.run(["hyprctl", "clients"], capture_output=True, text=True).stdout
    for block in out.split("Window "):
        if "waydroid.com.my.defense" not in block:
            continue
        at = re.search(r"at:\s*(-?\d+),(-?\d+)", block)
        if at:
            return int(at.group(1)), int(at.group(2))
    return 0, 0


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    sys.path.insert(0, str(ROOT))
    from rrbot.input import UinputMouseBackend

    ox, oy = waydroid_window()
    mouse = UinputMouseBackend()
    mouse.drag(ox + x1, oy + y1, ox + x2, oy + y2, steps=12, press_ms=150,
               step_ms=40, hold_ms=200)
    mouse.close()


def nearest(pawns: dict, point: tuple[int, int], limit: int = 30):
    best = None
    best_distance = limit
    for key in pawns:
        distance = max(abs(key[0] - point[0]), abs(key[1] - point[1]))
        if distance < best_distance:
            best_distance = distance
            best = key
    return best


def main() -> int:
    variants = [("основание", 0), ("тело (на 30 выше)", -30), ("ниже основания", 25)]
    print("ищем пару одинаковых соседних пешек...")
    chosen = None
    deadline = time.time() + 120
    while time.time() < deadline and chosen is None:
        shoot(LIVE)
        info = probe(LIVE)
        for x1, y1, x2, y2, score in info["pairs"]:
            if score < 0.95:
                continue
            if abs(x1 - x2) <= 120 and abs(y1 - y2) <= 120:
                chosen = (x1, y1, x2, y2)
                break
    if not chosen:
        print("пара не появилась (бот должен играть в это время)")
        return 1
    x1, y1, x2, y2 = chosen
    print(f"пара: ({x1},{y1}) -> ({x2},{y2})")
    for name, dy in variants:
        shoot(LIVE)
        before = probe(LIVE)
        if not before["pawns"]:
            print("  поле пусто")
            return 1
        drag(x1, y1, x2, y2 + dy)
        time.sleep(1.5)
        shoot(LIVE)
        after = probe(LIVE)
        source_alive = nearest(after["pawns"], (x1, y1)) is not None
        print(f"  {name}: пешек было {len(before['pawns'])}, стало {len(after['pawns'])}, "
              f"источник на месте: {source_alive}")
        if not source_alive:
            print("  -> слияние вышло")
            x = x2
            y = y2 + dy
            img = cv2.imread(str(LIVE))
            crop = img[max(0, y - 60):y + 40, max(0, x - 60):x + 60]
            cv2.imwrite("/tmp/opencode/drop_after.png", crop)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
