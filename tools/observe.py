#!/usr/bin/env python3
"""Наблюдение за игрой: запись полевых кадров и решений.

Записывает состояние поля (кадры холста, энергию, позиции юнитов) в
каталог. Нужен, чтобы собрать обучающую выборку и сравнить решения
бота с действиями человека.

Запуск:
    python tools/observe.py --seconds 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402

from cdp import Session, pick_page  # noqa: E402
from rrbot.capture import GrimSource  # noqa: E402
from rrbot.config import load_config  # noqa: E402
from rrbot.live import LiveBot, LiveGame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Наблюдение за игрой")
    parser.add_argument("--config", default=str(ROOT / "config" / "live.yaml"))
    parser.add_argument("--out", default="/tmp/opencode/observe")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=0.6)
    parser.add_argument("--play", action="store_true", help="бот играет сам, а не только смотрит")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    game = LiveGame()
    grabber = GrimSource(game.window.region)
    session = Session(pick_page())
    bot = LiveBot(config, game, clicker=None)

    # Дойти до боя.
    for step in range(25):
        game.sync_geometry(session)
        canvas = game.grab_canvas(grabber)
        if canvas is None:
            time.sleep(1.0)
            continue
        if game.in_battle(canvas):
            print(f"== бой начался (шаг {step}) ==")
            break
        button = game.navigation_button(canvas)
        if button is None:
            time.sleep(1.5)
            continue
        time.sleep(14.0 if button.kind == "pvp" else 2.5)
    else:
        print("не удалось дойти до боя", file=sys.stderr)
        return 1

    log = []
    deadline = time.time() + args.seconds
    index = 0
    while time.time() < deadline:
        game.sync_geometry(session)
        result = bot.step()
        if result.state is None:
            break
        frame_path = out / "frames" / f"f{index:04d}.png"
        cv2.imwrite(str(frame_path), result.canvas)
        entry = {
            "t": round(result.state.time, 2),
            "mana": result.mana,
            "own": [[u.kind, round(u.x), round(u.y)] for u in result.state.ally_units],
            "enemy": [[u.kind, round(u.x), round(u.y)] for u in result.state.enemy_units],
            "action": result.action.kind if result.action else None,
        }
        log.append(entry)
        print(
            f"[{index:04d}] t={entry['t']:6.1f} энергия={entry['mana']:5.0f} "
            f"своих={len(entry['own']):2d} врагов={len(entry['enemy']):2d} "
            f"действие={entry['action']}"
        )
        index += 1
        if not game.in_battle(result.canvas):
            print("== бой закончился ==")
            break
        time.sleep(args.interval)

    (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано {index} кадров в {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
