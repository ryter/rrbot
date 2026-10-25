#!/usr/bin/env python3
"""Живая проверка слияния: бот играет без слияний, слияние делаем руками.

Сценарий:
1. Запускаем бота с ключом --no-merge: он только зовёт юнитов.
2. Ждём, пока на поле появятся две одинаковые соседние пешки.
3. Останавливаем бота (SIGSTOP) — он больше не кликает.
4. Делаем перетаскивание (мышью, как это делает бот) и смотрим, что получилось:
   пешка-источник исчезла (слияние) или осталась на месте (не вышло).

Перетаскивание пробуем с разными задержками: если игра не успевает
«взять» юнита, жест нужно вести медленнее.

    .venv/bin/python tools/merge_experiment.py
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
FRAMES = Path("/tmp/opencode/merge_exp")
LOG = Path("/tmp/opencode/merge_exp.log")

# Тайминги жеста: (название, зажать до движения, шаг, шагов, ждать перед отпусканием)
VARIANTS = [
    ("как в боте", 150, 40, 12, 200),
    ("медленнее", 400, 60, 20, 400),
    ("ещё медленнее", 700, 90, 25, 600),
]


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def shoot(path: Path) -> None:
    with path.open("wb") as f:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=f)


def probe(frame: Path) -> dict:
    """Пешки, пары и положение своей половины арены."""
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
    """Положение окна эмулятора на экране: (x, y)."""
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
    """Перетаскивание той же виртуальной мышью, что и в боте."""
    sys.path.insert(0, str(ROOT))
    from rrbot.input import UinputMouseBackend

    ox, oy = waydroid_window()
    mouse = UinputMouseBackend()
    mouse.drag(ox + x1, oy + y1, ox + x2, oy + y2, steps=steps, press_ms=hold,
               step_ms=step_ms, hold_ms=hold_end)
    mouse.close()


def nearest(pawns: dict, point: tuple[int, int], limit: int = 30) -> tuple[int, int] | None:
    best = None
    best_distance = limit
    for key in pawns:
        distance = max(abs(key[0] - point[0]), abs(key[1] - point[1]))
        if distance < best_distance:
            best_distance = distance
            best = key
    return best


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
    print("бот запущен, ждём пару одинаковых соседних пешек...")
    live = Path("/tmp/opencode/exp_live.png")
    chosen = None
    deadline = time.time() + 240
    while time.time() < deadline and chosen is None:
        time.sleep(2)
        shoot(live)
        info = probe(live)
        for x1, y1, x2, y2, score in info["pairs"]:
            if score < 0.95:
                continue
            if abs(x1 - x2) <= 120 and abs(y1 - y2) <= 120:
                chosen = (x1, y1, x2, y2)
                break
    if not chosen:
        print("пара не появилась")
        proc.kill()
        return 1
    os_pid = proc.pid
    subprocess.run(["kill", "-STOP", str(os_pid)], capture_output=True)
    print(f"бот остановлен (pid {os_pid}), пара {chosen}")
    x1, y1, x2, y2 = chosen
    for index, (name, hold, step_ms, steps, hold_end) in enumerate(VARIANTS):
        before = Path(f"/tmp/opencode/exp_before{index}.png")
        after = Path(f"/tmp/opencode/exp_after{index}.png")
        shoot(before)
        info = probe(before)
        print(f"попытка {index + 1} ({name}): ({x1},{y1}) -> ({x2},{y2})")
        mouse_drag(x1, y1, x2, y2, hold, step_ms, steps, hold_end)
        time.sleep(2)
        shoot(after)
        after_info = probe(after)
        # Камера могла сдвинуться — сравниваем относительно арены.
        shift = (after_info["origin"][0] - info["origin"][0],
                 after_info["origin"][1] - info["origin"][1])
        source = (x1 + shift[0], y1 + shift[1])
        target = (x2 + shift[0], y2 + shift[1])
        source_alive = nearest(after_info["pawns"], source) is not None
        target_alive = nearest(after_info["pawns"], target) is not None
        print(f"  пешек было {len(info['pawns'])}, стало {len(after_info['pawns'])}; "
              f"источник на месте: {source_alive}, цель на месте: {target_alive}")
        if not source_alive:
            print("  -> пешка-источник исчезла")
            # Крупный вид цели до и после: сравниваем, что изменилось у
            # юнита (знак ранга, форма основания).
            big = Path("/tmp/opencode/exp_cell.png")
            tiles = []
            for name, path in (("ДО", before), ("ПОСЛЕ", after)):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                x = x2 + (shift[0] if name == "ПОСЛЕ" else 0)
                y = y2 + (shift[1] if name == "ПОСЛЕ" else 0)
                crop = img[max(0, y - 50):y + 50, max(0, x - 50):x + 50]
                if crop.shape[0] != 100 or crop.shape[1] != 100:
                    continue
                crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_LANCZOS4)
                cv2.putText(crop, name, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tiles.append(crop)
            if tiles:
                cv2.imwrite(str(big), np.hstack(tiles))
                print("  крупно:", big)
            # Сохраняем крупные портреты всех своих пешек: после слияния
            # можно сравнить слитого юнита (ранг 2) с обычными (ранг 1).
            zoom = Path("/tmp/opencode/exp_zoom.png")
            tiles = []
            for (px, py) in sorted(after_info["pawns"]):
                crop = cv2.imread(str(after))[
                    max(0, py - 42):py + 42, max(0, px - 42):px + 42]
                if crop.shape[0] != 84 or crop.shape[1] != 84:
                    continue
                crop = cv2.resize(crop, (168, 168), interpolation=cv2.INTER_CUBIC)
                cv2.putText(crop, f"{px},{py}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)
                tiles.append(crop)
            if tiles:
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
                cv2.imwrite(str(zoom), np.vstack(rows))
                print("  портреты:", zoom)
            break
    subprocess.run(["kill", "-CONT", str(os_pid)], capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
