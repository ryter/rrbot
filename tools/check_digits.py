#!/usr/bin/env python3
"""Проверка чтения числа энергии по набору вырезов.

Набор лежит в ``assets/test_energy``: картинка с цифрами и файл
``labels.txt`` с ожидаемым значением. Скрипт прогоняет бот по каждому
вырезу и считает, сколько чисел прочитано верно.

    .venv/bin/python tools/check_digits.py
    .venv/bin/python tools/check_digits.py --digits-dir assets/digits_emulator_new
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "cpp" / "build" / "rrbot"
DEFAULT_DIR = ROOT / "assets" / "test_energy"

TEXT_RE = re.compile(r'цифры: "([0-9?]*)"')


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка чтения энергии")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="каталог с вырезами")
    parser.add_argument("--digits-dir", default=None,
                        help="каталог эталонов цифр (по умолчанию — рабочий)")
    parser.add_argument("--verbose", action="store_true", help="показать все вырезы")
    args = parser.parse_args()

    if not BOT.exists():
        print("нет собранного бота: сначала make в cpp/", file=sys.stderr)
        return 1

    directory = Path(args.dir)
    labels_path = directory / "labels.txt"
    if not labels_path.exists():
        print(f"нет файла {labels_path}", file=sys.stderr)
        return 1

    total = 0
    correct = 0
    wrong: list[tuple[str, str, str]] = []
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split()
        total += 1
        command = [str(BOT), "--backend", "adb", "--digits-image", str(directory / name)]
        if args.digits_dir:
            command += ["--digits-dir", args.digits_dir]
        out = subprocess.run(command, capture_output=True, text=True, check=False).stdout
        match = TEXT_RE.search(out)
        got = match.group(1) if match else ""
        if got == expected:
            correct += 1
        else:
            wrong.append((name, expected, got))
        if args.verbose:
            print(f"  {name}: ожидалось {expected}, получено {got or '—'}")

    print(f"прочитано верно: {correct}/{total}")
    for name, expected, got in wrong:
        print(f"  ОШИБКА {name}: ожидалось {expected}, получено {got or '—'}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
