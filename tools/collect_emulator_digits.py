#!/usr/bin/env python3
"""Сбор эталонов цифр из боевых кадров эмулятора.

Эталоны цифр в ``assets/digits_emulator`` описывают то начертание, которое
было в момент их сбора. Игра обновилась — и цифры в HUD стали другими:
например, «6» перестала совпадать со своим эталоном (0.55 вместо 0.9),
и распознаватель отдавал «?».

Инструмент проходит по кадрам боя, берёт те, где число прочиталось
уверенно, и сохраняет глифы как новые эталоны. Значение для подписи берётся
из самого прочитанного текста, поэтому подтверждённые кадры служат и
проверкой: если после сбора распознавание ухудшилось — эталонов мало или они
подписаны неверно.

Запуск:

    .venv/bin/python tools/collect_emulator_digits.py /tmp/opencode/frames2
    .venv/bin/python tools/collect_emulator_digits.py /tmp/opencode/frames2 --min-score 0.8
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "cpp" / "build" / "rrbot"
DIGIT_DIR = ROOT / "assets" / "digits_emulator"

# «по цифрам: 1(0.92) 0(0.88)» -> [(цифра, уверенность), ...]
GLYPH_RE = re.compile(r"([0-9?])\(([\d.]+)\)")
TEXT_RE = re.compile(r'цифры энергии: "([0-9?]*)"')


def probe(frame: Path) -> tuple[str, list[tuple[str, float]]]:
    """Что бот видит в кадре: текст числа и оценка по каждой цифре."""
    out = subprocess.run(
        [str(BOT), "--backend", "adb", "--frame", str(frame)],
        capture_output=True, text=True, check=False,
    ).stdout
    text_match = TEXT_RE.search(out)
    glyphs = [(m.group(1), float(m.group(2))) for m in GLYPH_RE.finditer(out)]
    return (text_match.group(1) if text_match else ""), glyphs


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор эталонов цифр из кадров боя")
    parser.add_argument("frames", nargs="+", help="каталог с кадрами (PNG)")
    parser.add_argument("--out", default=str(ROOT / "assets" / "digits_emulator_new"),
                        help="куда складывать новые эталоны")
    parser.add_argument("--min-score", type=float, default=0.8,
                        help="минимальная уверенность по каждой цифре")
    parser.add_argument("--max-per-digit", type=int, default=6,
                        help="сколько образцов на цифру сохранять")
    parser.add_argument("--apply", action="store_true",
                        help="заменить текущие эталоны собранными")
    args = parser.parse_args()

    if not BOT.exists():
        print("нет собранного бота: сначала make в cpp/", file=sys.stderr)
        return 1

    frames: list[Path] = []
    for entry in args.frames:
        path = Path(entry)
        frames.extend(sorted(path.glob("*.png")) if path.is_dir() else [path])
    if not frames:
        print("кадров не найдено", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    counts: collections.Counter[str] = collections.Counter()
    used = 0
    for frame in frames:
        text, glyphs = probe(frame)
        if not text or "?" in text or len(glyphs) != len(text):
            continue
        if any(score < args.min_score for _, score in glyphs):
            continue
        saved = subprocess.run(
            [str(BOT), "--backend", "adb", "--frame", str(frame), "--learn", text,
             "--learn-dir", str(out_dir), "--learn-max", str(args.max_per_digit)],
            capture_output=True, text=True, check=False,
        )
        if "сохранено" not in saved.stdout:
            continue
        used += 1

    # Считаем по файлам: так видно, сколько образцов на цифру реально есть.
    counts: collections.Counter[str] = collections.Counter()
    for path in out_dir.glob("*.png"):
        counts[path.stem.split("_")[0]] += 1

    print(f"кадров разобрано: {len(frames)}, использовано: {used}")
    print("образцов на цифру:", dict(sorted(counts.items())))
    missing = [d for d in "0123456789" if counts[d] == 0]
    if missing:
        print("нет образцов для цифр:", " ".join(missing))

    if args.apply:
        if missing:
            print("эталоны не заменены: не все цифры собраны", file=sys.stderr)
            return 1
        shutil.rmtree(DIGIT_DIR, ignore_errors=True)
        shutil.copytree(out_dir, DIGIT_DIR)
        print(f"эталоны заменены: {DIGIT_DIR}")
    else:
        print(f"новые эталоны в {out_dir} (применить: --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
