#!/usr/bin/env python3
"""Автобой на живой игре: полный цикл бота.

Бот сам проходит по меню, дожидается начала матча, анализирует поле,
принимает решение и ставит юнитов. Каждое действие логируется, кадры
сохраняются.

Главное отличие от «просто прокликать»: после нажатия «PvP» бот **ждёт
матч** и проверяет, что бой действительно начался. Экран подбора
соперника анимируется, поэтому судить по нему «изменился ли экран» нельзя
— признак начала боя только один: сам бой.

Запуск:
    python tools/autoplay.py --steps 40
    python tools/autoplay.py --steps 40 --input cdp   # клики через CDP
    python tools/autoplay.py --steps 40 --mode coop   # сразу Co-Op
    python tools/autoplay.py --steps 40 --dry-run     # без кликов
"""

from __future__ import annotations

import argparse
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
from rrbot.live import (  # noqa: E402
    MATCH_MODES,
    LiveBot,
    LiveGame,
    Matchmaking,
    screen_changed,
    wait_for_battle,
)

# Сколько шагов меню пройти, прежде чем сдаться.
MENU_STEPS = 60


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="autoplay.py",
        description="Автобой на живой игре Rush Royale (веб-клиент в браузере).",
        epilog=(
            "Примеры:\n"
            "  autoplay.py --steps 40                     обычный забег\n"
            "  autoplay.py --steps 40 --mode coop         сразу искать Co-Op\n"
            "  autoplay.py --steps 40 --timeout-match 45  ждать матч дольше\n"
            "  autoplay.py --steps 40 --dry-run           считать, но не кликать\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "live.yaml"),
        help="файл конфигурации бота (по умолчанию config/live.yaml)",
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="сколько шагов боя (кадров) отыграть"
    )
    parser.add_argument(
        "--out", default="/tmp/opencode/autoplay", help="куда сохранять кадры боя"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="не кликать: только смотреть и считать (проверка разбора экрана)",
    )
    parser.add_argument(
        "--input",
        choices=["mouse", "cdp"],
        default="mouse",
        help="mouse — настоящая мышь (uinput); cdp — клики внутри вкладки Chrome",
    )
    parser.add_argument(
        "--timeout-match",
        type=float,
        default=30.0,
        help=(
            "сколько секунд ждать матч после нажатия «PvP»/«Co-Op» "
            "(по умолчанию 30)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=list(MATCH_MODES),
        default="auto",
        help=(
            "auto — сначала PvP, при неудаче Co-Op; pvp — только PvP; "
            "coop — только Co-Op (по умолчанию auto)"
        ),
    )
    return parser


def choose_candidates(candidates, matchmaking: Matchmaking):
    """Отбор кандидатов по режиму подбора.

    В режиме «pvp» кнопку «Co-Op» жать нельзя (и наоборот) — иначе бот
    сделает не то, что попросили. В режиме «auto» сначала идёт PvP, а
    после неудачных подборов — Co-Op.
    """
    if matchmaking.mode == "pvp":
        return [b for b in candidates if b.kind != "coop"]
    if matchmaking.mode == "coop":
        return [b for b in candidates if b.kind != "pvp"]
    if not matchmaking.use_coop:
        return [b for b in candidates if b.kind != "coop"]
    coop = [b for b in candidates if b.kind == "coop"]
    rest = [b for b in candidates if b.kind != "coop"]
    return coop + rest


def go_to_battle(
    game: LiveGame,
    grab,
    click_canvas,
    matchmaking: Matchmaking,
    timeout_match: float,
    sync=None,
) -> bool:
    """Пройти по меню до начала боя.

    Возвращает ``True``, если бой начался. После нажатия «PvP» или «Co-Op»
    ждём матч (проверяя, что бой начался, а не что «изменился экран»).
    """
    dead: dict[str, int] = {}
    tries: dict[str, int] = {}
    awaiting_match = False

    for step in range(MENU_STEPS):
        if sync is not None:
            sync()
        canvas = grab()
        if canvas is None:
            time.sleep(1.0)
            continue
        if game.in_battle(canvas):
            print(f"== матч начался (шаг {step}) ==")
            matchmaking.on_match_started()
            return True

        candidates = choose_candidates(game.navigation_candidates(canvas), matchmaking)
        for button in candidates:
            if button.kind in ("pvp", "coop"):
                sx, sy = click_canvas(*button.center)
                print(f"[меню {step}] «{button.kind}» -> ({sx}, {sy})")
                # Ждём матч: подбор соперника идёт по сети и может занять
                # полминуты. Пока идёт — экран анимируется, поэтому по кадру
                # «изменилось» судить бесполезно: смотрим, начался ли бой.
                awaiting_match = True
                matchmaking.begin_attempt()
                print(f"    жду матч ({matchmaking.mode_name}, до {timeout_match:.0f} с)")
                started = wait_for_battle(
                    grab,
                    game.in_battle,
                    timeout=timeout_match,
                )
                if started:
                    print(f"== матч начался после «{button.kind}» ==")
                    matchmaking.on_match_started()
                    return True
                # Ждали и не дождались: сервер не находит соперника. Дальше
                # по меню идти нельзя — экран подбора анимируется, и слепые
                # клики по нему бесполезны. Следующий шаг цикла разберётся с
                # экраном заново.
                if matchmaking.on_search_failed():
                    print(
                        f"    соперник не нашёлся, переключаюсь на Co-Op "
                        f"(неудачных подборов: {matchmaking.pvp_fails})"
                    )
                else:
                    print("    соперник не нашёлся, пробую ещё раз")
                break

            if dead.get(button.kind, 0) > step:
                continue
            # Экран с анимацией «меняется» после любого клика, поэтому
            # одну и ту же кнопку не жмём подряд больше трёх раз.
            if tries.get(button.kind, 0) >= 3:
                dead[button.kind] = step + 12
                tries[button.kind] = 0
                continue
            tries[button.kind] = tries.get(button.kind, 0) + 1
            before = canvas.copy()
            sx, sy = click_canvas(*button.center)
            print(f"[меню {step}] «{button.kind}» -> ({sx}, {sy})")

            time.sleep(2.5)
            after = grab()
            changed = screen_changed(before, after)
            if button.kind == "cancel" or (awaiting_match and button.kind == "ok"):
                # Отменили подбор или закрыли окно «Противник не найден» —
                # значит PvP на этом сервере соперников не находит.
                awaiting_match = False
                if matchmaking.on_search_cancelled():
                    print(
                        f"    PvP не работает, переключаюсь на Co-Op "
                        f"(неудачных подборов: {matchmaking.pvp_fails})"
                    )
            print(f"    {'ок' if changed else 'без эффекта'}")
            if changed:
                break
            dead[button.kind] = step + 4
        else:
            time.sleep(1.5)
    return False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    game = LiveGame()
    grabber = GrimSource(game.window.region)

    # CDP нужен всегда: через него бот узнаёт точные координаты холста.
    session = None if args.dry_run else Session(pick_page())

    mouse = None
    if args.input == "mouse" and not args.dry_run:
        from rrbot.input import UinputMouseBackend

        mouse = UinputMouseBackend()

    def click(x: int, y: int) -> None:
        # Бот передаёт экранные координаты — мыши они нужны как есть.
        if args.dry_run:
            return
        if mouse is not None:
            mouse.click(x, y)
        elif session is not None:
            game.click_screen(session, x, y)

    bot = LiveBot(config, game, clicker=click, source=grabber)

    def click_canvas(cx: int, cy: int) -> tuple[int, int]:
        """Клик по точке холста."""
        point = game.to_screen(cx, cy)
        if args.dry_run:
            return point
        if mouse is not None:
            mouse.click(*point)
        elif session is not None:
            game.click_screen(session, *point)
        return point

    def grab():
        """Свежий кадр холста (или None, если кадр не получился)."""
        return game.grab_canvas(grabber)

    matchmaking = Matchmaking(mode=args.mode)
    print(
        f"режим подбора: {matchmaking.mode_name} "
        f"(переключение на Co-Op после {matchmaking.waits_before_coop} неудач), "
        f"ожидание матча: {args.timeout_match:.0f} с"
        + (", без кликов (--dry-run)" if args.dry_run else "")
    )

    # 1) Дойти до боя.
    if not go_to_battle(
        game,
        grab,
        click_canvas,
        matchmaking,
        args.timeout_match,
        sync=lambda: game.sync_geometry(session),
    ):
        print("не удалось дойти до боя", file=sys.stderr)
        return 1

    # 2) Основной цикл боя.
    for step in range(args.steps):
        game.sync_geometry(session)
        result = bot.step()
        if result.canvas is None:
            print("кадр не получен")
            continue
        if not result.in_battle:
            # Экран итогов: в области энергии там лежит совсем другое,
            # читать её и кликать призыва нельзя.
            print("== матч закончился ==")
            break
        action = result.action
        energy = result.mana
        units = len(result.state.enemy_units)
        own = len(result.state.ally_units)
        print(
            f"[{step:02d}] энергия={energy:5.0f} своих={own:2d} врагов={units:2d} "
            f"призыв={'да' if result.affordable else 'нет'} "
            f"действие={action.kind:6s} {action.unit_kind or '-':8s} "
            f"причина: {action.reason[:50]}"
        )
        if result.clicked:
            print(f"     клики: {result.clicked}")
            cv2.imwrite(str(out / f"step_{step:02d}.png"), result.canvas)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
