#!/usr/bin/env python3
"""Клик настоящей мышью по экранным координатам.

Используется то же устройство uinput, что и в боте: координаты попадают
пиксель в пиксель, курсор виден, кликает любое приложение.

Примеры:
    python tools/mouse_click.py 1400 900          # клик
    python tools/mouse_click.py --move 1400 900   # только навести курсор
    python tools/mouse_click.py --where           # где сейчас курсор
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cursor_pos() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["hyprctl", "cursorpos"], capture_output=True, timeout=5, check=False
        ).stdout.decode()
        x, y = out.strip().split(",")
        return int(x), int(y)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Клик виртуальной мышью")
    parser.add_argument("x", nargs="?", type=int)
    parser.add_argument("y", nargs="?", type=int)
    parser.add_argument("--move", action="store_true", help="только навести курсор")
    parser.add_argument("--where", action="store_true", help="показать положение курсора")
    args = parser.parse_args()

    if args.where:
        print("курсор:", cursor_pos())
        return 0
    if args.x is None or args.y is None:
        parser.error("нужны координаты X Y")

    from rrbot.input import UinputMouseBackend

    mouse = UinputMouseBackend()
    if args.move:
        mouse.move(args.x, args.y)
    else:
        mouse.click(args.x, args.y)
    print("курсор:", cursor_pos())
    mouse.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
