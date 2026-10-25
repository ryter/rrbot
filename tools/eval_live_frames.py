#!/usr/bin/env python3
"""Проверка детектора на живых кадрах боя (без эталонной разметки).

Разметки у живых кадров нет, поэтому качество оценивается косвенно:

* **сколько находок на кадре** — на поле должно быть столько, сколько
  видно юнитов; ноль или десятки «находок» в пустом поле — мусор;
* **повторяемость между соседними кадрами** — юниты стоят на месте
  (камера ходит медленно), поэтому находка должна повторяться на следующем
  кадре; доля повторов показывает, что детектор не дрожит;
* **сколько находок попало в пустые места** — находки вне половин арены и
  в чёрных (не арена) местах — мусор.

Кадры боя лежат в ``/tmp/opencode/battle_loop`` — это полные снимки экрана
эмулятора (941x1030). Каноническая доска (400x522, две половины арены одна
под другой) собирается тем же кодом, что и у бота: ``rr::canonical_board``
из ``cpp/src/arena.cpp`` — проверочная программа ``cpp/eval/arena_probe.cpp``
собирается на лету (как в tests/test_cpp_arena.py).

Запуск:

    .venv/bin/python tools/eval_live_frames.py --frames /tmp/opencode/battle_loop
    .venv/bin/python tools/eval_live_frames.py --frames ... --templates assets/real_emulator2
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "cpp" / "build" / "rrvision"
PROBE_SRC = ROOT / "cpp" / "eval" / "arena_probe.cpp"
ARENA_SRC = ROOT / "cpp" / "src" / "arena.cpp"

DET_RE = re.compile(r"^НАХОДКИ (\S+)(.*)$")
ITEM_RE = re.compile(r"(\S+) ([\d.]+) \((-?\d+),(-?\d+)\)")
# Половины арены в каноническом кадре (см. cpp/src/arena.hpp).
HALF_H = 240
GAP = 42
BOARD_H = 522


def build_probe(work_dir: Path) -> Path | None:
    """Собрать проверочную программу, которая собирает канонические кадры."""
    binary = work_dir / "arena_probe"
    build = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-I/usr/include/opencv5",
         f"-I{ROOT / 'cpp' / 'src'}", str(PROBE_SRC), str(ARENA_SRC),
         "-o", str(binary), "-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs"],
        capture_output=True, text=True, check=False,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr)
        return None
    return binary


def canonical_boards(frames: list[Path], work_dir: Path) -> tuple[Path, list[str]]:
    """Собрать канонические кадры из полных снимков экрана."""
    probe = build_probe(work_dir)
    if probe is None:
        raise SystemExit("не собралась проверочная программа arena_probe")
    boards_dir = work_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(probe), "--board", str(boards_dir), *[str(p) for p in frames]],
        capture_output=True, text=True, check=False,
    )
    names = []
    for line in result.stdout.splitlines():
        if "канонический кадр" in line:
            names.append(line.split(":", 1)[1].strip().split()[0])
    return boards_dir, names


def run_detector(boards: Path, templates: Path, extra: list[str],
                 by_base: bool = True) -> dict[str, list[tuple[str, float, int, int]]]:
    cmd = [str(TOOL), "--frames", str(boards), "--truth", "/dev/null",
           "--templates", str(templates), "--all-templates",
           "--team-by-position", "--split", "0.5", "--print-detections"]
    if by_base:
        cmd.append("--by-base")
    cmd += extra
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit("утилита замера упала")
    detections: dict[str, list[tuple[str, float, int, int]]] = defaultdict(list)
    for line in result.stdout.splitlines():
        match = DET_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1)
        for item in match.group(2).split("|"):
            found = ITEM_RE.search(item)
            if found:
                detections[name].append((found.group(1), float(found.group(2)),
                                         int(found.group(3)), int(found.group(4))))
    return detections


def save_preview(path: Path, boards: Path, names: list[str],
                 detections: dict[str, list[tuple[str, float, int, int]]]) -> None:
    """Сохранить кадр с отмеченными находками — чтобы глазами проверить мусор."""
    import cv2
    import numpy as np

    if not names:
        return
    name = names[len(names) // 2]
    board = cv2.imread(str(boards / name))
    if board is None:
        return
    for (_name, _score, x, y) in detections.get(name, []):
        cv2.circle(board, (x, y), 18, (0, 0, 255), 2)
    cv2.imwrite(str(path), board)
    print(f"кадр с отметками: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка детектора на живых кадрах")
    parser.add_argument("--frames", default="/tmp/opencode/battle_loop")
    parser.add_argument("--templates", default=str(ROOT / "assets" / "real_emulator2"))
    parser.add_argument("--count", type=int, default=40, help="сколько кадров взять")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--pair-radius", type=float, default=25.0,
                        help="на каком расстоянии находка считается той же самой")
    parser.add_argument("--extra", default="", help="дополнительные ключи утилиты замера")
    parser.add_argument("--no-base", action="store_true",
                        help="старая проверка: только сопоставление с эталонами")
    parser.add_argument("--preview", default="",
                        help="куда сохранить кадр с отметками находок")
    args = parser.parse_args()

    if not TOOL.exists():
        print(f"не собрана утилита {TOOL} (cd cpp && make build/rrvision)", file=sys.stderr)
        return 1
    source = Path(args.frames)
    files = sorted(source.glob("*.png"))
    if not files:
        print(f"в каталоге {source} нет кадров", file=sys.stderr)
        return 1
    files = files[args.start::args.step][:args.count]

    work_dir = Path(tempfile.mkdtemp(prefix="rrlive-"))
    try:
        boards_dir, names = canonical_boards(files, work_dir)
        if not names:
            print("не собралось ни одного канонического кадра", file=sys.stderr)
            return 1
        extra = args.extra.split() if args.extra else []
        detections = run_detector(boards_dir, Path(args.templates), extra, not args.no_base)

        if args.preview:
            save_preview(Path(args.preview), boards_dir, names, detections)

        counts = []
        repeats = []
        outside = 0
        total = 0
        for index, name in enumerate(names):
            found = detections.get(name, [])
            counts.append(len(found))
            for (_name, _score, x, y) in found:
                total += 1
                if not (y < HALF_H or y >= HALF_H + GAP):
                    outside += 1
            if index == 0:
                continue
            previous = detections.get(names[index - 1], [])
            if not found or not previous:
                repeats.append(0.0)
                continue
            matched = 0
            used: set[int] = set()
            for (_name, _score, x, y) in found:
                best, best_index = 1e9, -1
                for j, (_n, _s, px, py) in enumerate(previous):
                    if j in used:
                        continue
                    d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                    if d < best:
                        best, best_index = d, j
                if best_index >= 0 and best <= args.pair_radius:
                    used.add(best_index)
                    matched += 1
            repeats.append(matched / len(found))

        print(f"кадров: {len(names)}")
        print(f"находок на кадр: минимум {min(counts)}, максимум {max(counts)}, "
              f"в среднем {sum(counts) / len(counts):.2f}")
        histogram: dict[int, int] = defaultdict(int)
        for value in counts:
            histogram[value] += 1
        print("распределение числа находок: " +
              ", ".join(f"{value}->{count}" for value, count in sorted(histogram.items())))
        if repeats:
            print(f"повторяемость на соседнем кадре: в среднем "
                  f"{sum(repeats) / len(repeats):.3f} (минимум {min(repeats):.3f})")
        print(f"находок вне половин арены: {outside} из {total}")
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
