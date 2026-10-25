#!/usr/bin/env python3
"""Точка входа исследовательского бота Rush Royale.

Запуск:
    python main.py                # графический интерфейс, симулятор
    python main.py --source screen # захват экрана
    python main.py --headless 200  # прогон без GUI (для замеров)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rrbot.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Исследовательский бот Rush Royale")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--source", choices=["sim", "screen"], default=None)
    parser.add_argument("--headless", type=int, default=0, metavar="N", help="прогнать N кадров без GUI")
    parser.add_argument("--allow-input", action="store_true", help="разрешить реальные клики")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.source:
        config.capture.source = args.source
    if args.allow_input:
        config.allow_input = True

    if args.headless:
        from rrbot.headless import run_headless

        return run_headless(config, args.headless)

    from PySide6.QtWidgets import QApplication

    from rrbot.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Rush Royale research bot")
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
