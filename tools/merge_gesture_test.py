#!/usr/bin/env python3
"""Проверка жеста: перетаскивание пешки на пустую клетку.

Если пешка переезжает — жест работает, и неудачи слияний объясняются
правилами игры (разный ранг). Если не переезжает — жест не подхватывается,
и надо менять тайминги или способ перетаскивания.

    .venv/bin/python tools/merge_gesture_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/lennistyle/rushroyale-lab")
CPP = ROOT / "cpp"
FRAMES = Path("/tmp/opencode/gesture_exp")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as f:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=f)


def probe(frame: Path) -> dict:
    out = sh([str(CPP / "build" / "merge_probe"), str(frame)])
    arena = re.search(r"арена: своя (\d+),(\d+) (\d+)x(\d+)", out)
    pawns = {}
    for x, y, radius in re.findall(r"  своя (\d+),(\d+) радиус (\d+)", out):
        pawns[(int(x), int(y))] = int(radius)
    origin = (int(arena.group(1)), int(arena.group(2))) if arena else (0, 0)
    return {"pawns": pawns, "origin": origin}


def waydroid_window() -> tuple[int, int]:
    out = subprocess.run(["hyprctl", "clients"], capture_output=True, text=True).stdout
    for block in out.split("Window "):
        if "waydroid.com.my.defense" not in block:
            continue
        at = re.search(r"at:\s*(-?\d+),(-?\d+)", block)
        if at:
            return int(at.group(1)), int(at.group(2))
    return 0, 0


def mouse_drag(x1: int, y1: int, x2: int, y2: int, hold: int, step_ms: int,
               steps: int, hold_end: int) -> None:
    sys.path.insert(0, str(ROOT))
    from rrbot.input import UinputMouseBackend

    ox, oy = waydroid_window()
    mouse = UinputMouseBackend()
    mouse.drag(ox + x1, oy + y1, ox + x2, oy + y2, steps=steps, press_ms=hold,
               step_ms=step_ms, hold_ms=hold_end)
    mouse.close()


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[r]rbot_merge --backend"], capture_output=True)
    time.sleep(1)
    FRAMES.mkdir(parents=True, exist_ok=True)
    log = (FRAMES / "log.txt").open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot_merge"), "--backend", "emulator", "--steps", "60",
         "--rounds", "1", "--no-merge", "--record", str(FRAMES)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = Path("/tmp/opencode/gesture_live.png")
    print("ждём пешки на поле...")
    pawn = None
    deadline = time.time() + 180
    while time.time() < deadline and pawn is None:
        time.sleep(2)
        shoot(live)
        info = probe(live)
        if len(info["pawns"]) >= 2:
            pawn = sorted(info["pawns"])[0]
    if pawn is None:
        print("пешек нет")
        proc.kill()
        return 1
    os_pid = proc.pid
    subprocess.run(["kill", "-STOP", str(os_pid)], capture_output=True)
    print(f"бот остановлен, пешка {pawn}")

    # Пустая соседняя клетка: сдвиг на 78 px вправо (шаг сетки).
    step_x = 78
    for index, (name, hold, step_ms, steps, hold_end) in enumerate(
            [("мышь как в боте", 150, 40, 12, 200),
             ("мышь медленно", 400, 80, 20, 400)]):
        before = Path(f"/tmp/opencode/gesture_before{index}.png")
        after = Path(f"/tmp/opencode/gesture_after{index}.png")
        shoot(before)
        info_before = probe(before)
        if not info_before["pawns"]:
            continue
        # Ищем пешку, справа от которой (через клетку) пусто.
        source = None
        target = None
        for (px, py) in sorted(info_before["pawns"]):
            for dx, dy in ((78, 0), (0, 78), (-78, 0), (0, -78)):
                candidate = (px + dx, py + dy)
                if any(max(abs(k[0] - candidate[0]), abs(k[1] - candidate[1])) <= 45
                       for k in info_before["pawns"]):
                    continue
                source, target = (px, py), candidate
                break
            if source:
                break
        if source is None:
            print("  нет пустой соседней клетки")
            continue
        print(f"попытка {index + 1} ({name}): ({source[0]},{source[1]}) -> "
              f"({target[0]},{target[1]})")
        mouse_drag(source[0], source[1], target[0], target[1], hold, step_ms, steps, hold_end)
        time.sleep(2)
        shoot(after)
        info_after = probe(after)
        shift = (info_after["origin"][0] - info_before["origin"][0],
                 info_after["origin"][1] - info_before["origin"][1])
        at_source = any(max(abs(k[0] - source[0] - shift[0]), abs(k[1] - source[1] - shift[1])) <= 30
                        for k in info_after["pawns"])
        at_target = any(max(abs(k[0] - target[0] - shift[0]), abs(k[1] - target[1] - shift[1])) <= 30
                        for k in info_after["pawns"])
        print("  пешки до:  ", sorted(info_before['pawns']))
        print("  пешки после:", sorted(info_after['pawns']))
        print(f"  пешек было {len(info_before['pawns'])}, стало {len(info_after['pawns'])}; "
              f"на месте: {at_source}, на цели: {at_target}")
        if at_target and not at_source:
            print("  -> пешка переехала: жест работает")
            break
    subprocess.run(["kill", "-CONT", str(os_pid)], capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
