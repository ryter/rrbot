#!/usr/bin/env python3
"""Генерация эталонных шаблонов юнитов.

Шаблоны лежат в ``assets/templates``. В режиме симулятора они генерируются
из тех же примитивов, что и отрисовка, поэтому гарантированно совпадают.

Для реальной игры шаблоны нужно снять с экрана (см. ``tools/calibrate.py``):
на скриншоте вырезается область юнита, сохраняется как ``<тип>.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rrbot.assets import write_templates  # noqa: E402
from rrbot.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерация шаблонов юнитов")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--out", default=None, help="каталог для шаблонов")
    args = parser.parse_args()

    config = load_config(args.config)
    kinds = list(config.vision.units.keys())
    if not kinds:
        print("В конфиге нет юнитов (секция vision.units)", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else ROOT / config.vision.templates_dir
    written = write_templates(out, kinds)
    for path in written:
        print(f"  {path.relative_to(ROOT)}")
    print(f"Готово: {len(written)} шаблонов в {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
