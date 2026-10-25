#!/usr/bin/env python3
"""Проверка гипотезы «слияние не выходит из-за ранга».

Сценарий:
1. Бот играет с ключом --no-merge: он зовёт юнитов (карточка одна, так что
   все юниты на поле одного типа).
2. Ждём двух одинаковых соседних пешек, сливаем их — получается юнит
   ранга 2.
3. Пробуем слить РЕЗУЛЬТАТ с ещё одной такой же пешкой. Если игра
   отказывает — значит, дело в ранге, и бот такие пары должен обходить.

    .venv/bin/python tools/merge_rank_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/lennistyle/rushroyale-lab")
CPP = ROOT / "cpp"
FRAMES = Path("/tmp/opencode/rank_exp")
LOG = Path("/tmp/opencode/rank_exp.log")

HOLD, STEP_MS, STEPS, HOLD_END = 150, 40, 12, 200


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
    pairs = []
    for a, b, c, d, score in re.findall(
            r"\((\d+),(\d+)\) -> \((\d+),(\d+)\): похожесть ([\d.]+)", out):
        pairs.append((int(a), int(b), int(c), int(d), float(score)))
    origin = (int(arena.group(1)), int(arena.group(2))) if arena else (0, 0)
    return {"pawns": pawns, "pairs": pairs, "origin": origin}


def waydroid_window() -> tuple[int, int]:
    out = subprocess.run(["hyprctl", "clients"], capture_output=True, text=True).stdout
    for block in out.split("Window "):
        if "waydroid.com.my.defense" not in block:
            continue
        at = re.search(r"at:\s*(-?\d+),(-?\d+)", block)
        if at:
            return int(at.group(1)), int(at.group(2))
    return 0, 0


def mouse_drag(x1: int, y1: int, x2: int, y2: int) -> None:
    """Перетаскивание: как в боте — виртуальной мышью (uinput)."""
    sys.path.insert(0, str(ROOT))
    from rrbot.input import UinputMouseBackend

    ox, oy = waydroid_window()
    mouse = UinputMouseBackend()
    mouse.drag(ox + x1, oy + y1, ox + x2, oy + y2, steps=STEPS, press_ms=HOLD,
               step_ms=STEP_MS, hold_ms=HOLD_END)
    mouse.close()


def adb_drag(x1: int, y1: int, x2: int, y2: int, duration: int = 900) -> None:
    """Перетаскивание через adb: input draganddrop — это настоящее
    «зажать, подождать, провести, отпустить» (Android 10+)."""
    subprocess.run(["adb", "shell", "input", "draganddrop", str(x1), str(y1),
                    str(x2), str(y2), str(duration)], capture_output=True)


def main() -> int:
    subprocess.run(["pkill", "-f", "[b]attle_loop.sh"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "[r]rbot_merge --backend"], capture_output=True)
    time.sleep(1)
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    log = LOG.open("w")
    proc = subprocess.Popen(
        [str(CPP / "build" / "rrbot_merge"), "--backend", "emulator", "--steps", "60",
         "--rounds", "1", "--no-merge", "--record", str(FRAMES)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(CPP), start_new_session=True)
    live = Path("/tmp/opencode/rank_live.png")
    print("ждём три одинаковые пешки (нужны на слияние и на проверку)...")
    pair = None
    deadline = time.time() + 240
    while time.time() < deadline and pair is None:
        time.sleep(2)
        shoot(live)
        info = probe(live)
        for x1, y1, x2, y2, score in info["pairs"]:
            if score < 0.95:
                continue
            if abs(x1 - x2) <= 120 and abs(y1 - y2) <= 120:
                pair = (x1, y1, x2, y2)
                break
    if not pair:
        print("пара не появилась")
        proc.kill()
        return 1
    os_pid = proc.pid
    subprocess.run(["kill", "-STOP", str(os_pid)], capture_output=True)
    print(f"бот остановлен, сливаем {pair}")
    x1, y1, x2, y2 = pair
    before = Path("/tmp/opencode/rank_before.png")
    after = Path("/tmp/opencode/rank_after.png")
    shoot(before)
    info_before = probe(before)
    adb_drag(x1, y1, x2, y2)
    time.sleep(2)
    shoot(after)
    info_after = probe(after)
    shift = (info_after["origin"][0] - info_before["origin"][0],
             info_after["origin"][1] - info_before["origin"][1])
    source = (x1 + shift[0], y1 + shift[1])
    target = (x2 + shift[0], y2 + shift[1])
    source_gone = all(max(abs(k[0] - source[0]), abs(k[1] - source[1])) > 30
                      for k in info_after["pawns"])
    target_alive = any(max(abs(k[0] - target[0]), abs(k[1] - target[1])) <= 30
                       for k in info_after["pawns"])
    print(f"первое слияние: пешек было {len(info_before['pawns'])}, стало "
          f"{len(info_after['pawns'])}; источник исчез: {source_gone}, "
          f"цель на месте: {target_alive}")
    if not source_gone:
        print("не слилось — эксперимент не удался")
        subprocess.run(["kill", "-CONT", str(os_pid)], capture_output=True)
        return 1

    # Теперь результат (ранг 2) на месте цели. Ищем рядом ещё одну такую же
    # пешку и пробуем слить её с результатом.
    print("пешки после слияния:", sorted(info_after["pawns"]))
    third = None
    for (px, py) in sorted(info_after["pawns"]):
        if max(abs(px - target[0]), abs(py - target[1])) <= 30:
            continue
        if max(abs(px - target[0]), abs(py - target[1])) <= 120:
            third = (px, py)
            break
    if third is None:
        print("третьей пешки рядом нет")
        subprocess.run(["kill", "-CONT", str(os_pid)], capture_output=True)
        return 1
    print(f"пробуем слить результат ранга 2 с пешкой {third}")
    before2 = Path("/tmp/opencode/rank_before2.png")
    after2 = Path("/tmp/opencode/rank_after2.png")
    shoot(before2)
    adb_drag(third[0], third[1], target[0], target[1])
    time.sleep(2)
    shoot(after2)
    info_after2 = probe(after2)
    shift2 = (info_after2["origin"][0] - info_after["origin"][0],
              info_after2["origin"][1] - info_after["origin"][1])
    src2 = (third[0] + shift2[0], third[1] + shift2[1])
    tgt2 = (target[0] + shift2[0], target[1] + shift2[1])
    gone2 = all(max(abs(k[0] - src2[0]), abs(k[1] - src2[1])) > 30 for k in info_after2["pawns"])
    alive2 = any(max(abs(k[0] - tgt2[0]), abs(k[1] - tgt2[1])) <= 30 for k in info_after2["pawns"])
    print(f"второе слияние (ранг 2 + ранг 1): пешек стало {len(info_after2['pawns'])}; "
          f"источник исчез: {gone2}, цель на месте: {alive2}")
    print("  пешки после:", sorted(info_after2['pawns']))
    print("кадры: /tmp/opencode/rank_before2.png, /tmp/opencode/rank_after2.png")
    subprocess.run(["kill", "-CONT", str(os_pid)], capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
