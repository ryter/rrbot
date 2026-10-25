#!/usr/bin/env python3
"""Точность определения ТИПА юнита (не только нахождения).

Манифест ``assets/real_emulator2/samples.tsv`` говорит, какой тип у каждого
юнита в эталонных кадрах: для каждого найденного юнита — кадр, позиция и
эталон. Детектор на тех же кадрах выдаёт свой эталон для каждой находки.

Сравниваются описания вида юнита (часть имени после ``unit_em_NN_``):
``teal_on_gray`` и так далее. Так сравнение работает и для первого набора
эталонов (``assets/real_emulator``), и для второго.

Запуск:

    .venv/bin/python tools/eval_types.py --templates assets/real_emulator2
    .venv/bin/python tools/eval_types.py --templates assets/real_emulator
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "cpp" / "build" / "rrvision"
FRAMES = ROOT / "assets" / "eval_real" / "frames"
TRUTH = ROOT / "assets" / "eval_real" / "truth.tsv"
SAMPLES = ROOT / "assets" / "real_emulator2" / "samples.tsv"

DET_RE = re.compile(r"^НАХОДКИ (\S+)(.*)$")
ITEM_RE = re.compile(r"(\S+) ([\d.]+) \((-?\d+),(-?\d+)\)")


def descriptor(name: str) -> str:
    """Описание вида юнита из имени эталона: unit_em_03_teal_on_teal -> teal_on_teal.

    Номер варианта (_1, _2) отбрасывается: это тот же тип юнита, просто
    другой кадр анимации.
    """
    parts = name.split("_")
    desc = "_".join(parts[3:]) if len(parts) > 3 else name
    if "_" in desc:
        head, _, tail = desc.rpartition("_")
        if tail.isdigit():
            desc = head
    return desc


def load_samples(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        out[parts[0]].append((float(parts[1]), float(parts[2]), parts[5]))
    return out


def run_tool(templates: Path, extra: list[str]) -> dict[str, list[tuple[str, float, int, int]]]:
    cmd = [str(TOOL), "--frames", str(FRAMES), "--truth", str(TRUTH),
           "--templates", str(templates), "--all-templates",
           "--team-by-position", "--split", "0.5", "--radius", "30",
           "--print-detections", "--by-base"] + extra
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Точность определения типа юнита")
    parser.add_argument("--templates", default=str(ROOT / "assets" / "real_emulator2"))
    parser.add_argument("--samples", default=str(SAMPLES))
    parser.add_argument("--radius", type=float, default=15.0)
    args = parser.parse_args()

    samples = load_samples(Path(args.samples))
    if not samples:
        print(f"нет манифеста {args.samples}", file=sys.stderr)
        return 1
    detections = run_tool(Path(args.templates), [])

    total = 0
    matched = 0
    correct = 0
    confusion: Counter[str] = Counter()
    for frame, items in samples.items():
        found = detections.get(frame, [])
        used: set[int] = set()
        for (x, y, kind) in items:
            total += 1
            best, best_index = 1e9, -1
            for index, (_name, _score, dx, dy) in enumerate(found):
                if index in used:
                    continue
                d = ((dx - x) ** 2 + (dy - y) ** 2) ** 0.5
                if d < best:
                    best, best_index = d, index
            if best_index < 0 or best > args.radius:
                continue
            used.add(best_index)
            matched += 1
            predicted = descriptor(found[best_index][0])
            expected = descriptor(kind)
            if predicted == expected:
                correct += 1
            else:
                confusion[f"{expected} -> {predicted}"] += 1

    print(f"юнитов в манифесте: {total}")
    print(f"найдено детектором рядом: {matched} ({matched / max(1, total):.3f})")
    print(f"тип определён верно: {correct} из {matched} ({correct / max(1, matched):.3f})")
    if confusion:
        print("путает:")
        for key, value in confusion.most_common(12):
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
