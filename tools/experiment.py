#!/usr/bin/env python3
"""Сравнение конфигураций планировщика на партиях симулятора.

Прогоняет серию матчей с заданной стратегией и собирает метрики:
победы, средняя длительность партии, потраченная мана, убийства, смерти и
число вызванных юнитов по типам.

Про честность сравнения (важно!):

* партии всех пресетов идут на одних и тех же сидах, поэтому волны врага
  у всех одинаковые — сравнение парное;
* в симуляторе у карточек есть **перезарядка**: после призыва карточка
  горит ``UnitSpec.recharge`` секунд, и звать этот тип юнита снова нельзя.
  Поэтому «доступные юниты» — это карточки, которые не горят, а не просто
  «хватает маны». Пресеты, которые меняют веса, теперь действительно могут
  изменить выбор типа юнита;
* печатается доля решений, где кандидат был один: по этой цифре видно,
  насколько сравнение вообще информативно;
* базовая линия — случайный выбор карточки. Её два варианта:
  ``random`` (случайная карточка из колоды, если она горит или не хватает
  маны — ждём) и ``random_ready`` (случайная из готовых и оплачиваемых —
  это верхняя граница для случайной игры).

Пример:
    python tools/experiment.py --seeds 20
    python tools/experiment.py --compare counter_heavy,balanced --out /tmp/opencode/exp.md
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rrbot.config import AppConfig, load_config  # noqa: E402
from rrbot.models import Team  # noqa: E402
from rrbot.planner import Planner  # noqa: E402
from rrbot.sim import SimWorld  # noqa: E402

# Пресеты: каждая запись меняет настройки планировщика.
# Накопление маны включаем там, где меняются веса: без него в симуляторе
# доступен ровно один юнит (самый дешёвый) и веса не могут изменить выбор.
PRESETS: dict[str, dict[str, float]] = {
    "balanced": {},
    "no_save": {"save_mana_enabled": False},
    "save_mana": {"save_mana_enabled": True},
    "counter_heavy": {"save_mana_enabled": True, "weight_counter": 6.0, "weight_cost_efficiency": 0.4},
    "cost_focused": {"save_mana_enabled": True, "weight_cost_efficiency": 3.0, "weight_counter": 1.0},
    "pressure_focused": {"save_mana_enabled": True, "weight_pressure": 6.0, "weight_counter": 1.0},
    "always_tank": {"save_mana_enabled": True, "weight_cost_efficiency": 2.0, "weight_synergy": 0.0},
}

# Базовая линия: случайный доступный юнит.
RANDOM_BASELINE = "random"
# Верхняя граница случайной игры: случайный из готовых и оплачиваемых карточек.
RANDOM_READY_BASELINE = "random_ready"
BASELINES = (RANDOM_BASELINE, RANDOM_READY_BASELINE)


@dataclass
class MatchResult:
    """Итог одной партии."""

    winner: str | None
    our_base: float
    enemy_base: float
    time: float
    kills: int
    deaths: int
    spent_mana: float
    spawns: int = 0
    spawn_kinds: dict[str, int] = field(default_factory=dict)
    waits: int = 0  # решений «подождать»
    decisions: int = 0  # решений планировщика всего
    single_candidate: int = 0  # решений, где кандидат был один
    no_card_ready: int = 0  # тактов, когда не горит ни одна карточка
    card_waits: int = 0  # тактов, когда ждали из-за горящей карточки
    ready_cards_total: int = 0  # сумма числа горящих карточек по тактам
    trace: list[tuple] = field(default_factory=list)  # (время, тип, линия)

    @property
    def won(self) -> bool:
        return self.winner == "ally"


def play_match(config: AppConfig, seed: int, strategy: str = "planner") -> MatchResult:
    """Сыграть одну партию ботом против штатных волн симулятора."""
    specs = config.vision.specs()
    world = SimWorld(config.sim, specs, seed=seed)
    planner = Planner(config)
    rng = random.Random(seed * 7919 + 13)
    result = MatchResult(
        winner=None, our_base=0.0, enemy_base=0.0, time=0.0,
        kills=0, deaths=0, spent_mana=0.0,
    )
    deck = sorted(planner.specs.keys())

    while not world.game_over and world.time < config.sim.max_duration:
        world.step(config.sim.dt)
        # Доступные карточки: те, что не горят (перезарядка закончилась).
        # Мана — отдельное ограничение, планировщик решает сам.
        ready = world.ready_kinds()
        result.ready_cards_total += len(ready)
        if not ready:
            result.no_card_ready += 1
        if strategy == RANDOM_BASELINE:
            # Случайный юнит из колоды (а не «самый дешёвый из доступных»:
            # так базовая линия действительно случайная, а не копия стратегии
            # планировщика, который тоже берёт самое дешёвое).
            kind = rng.choice(deck)
            if not world.can_summon(kind):
                result.waits += 1
                if not world.card_ready(kind):
                    result.card_waits += 1
                continue
            lane = rng.choice(["left", "right"])
        elif strategy == RANDOM_READY_BASELINE:
            # Случайный из тех, кого можно позвать прямо сейчас: верхняя
            # граница для случайной игры (никогда не ждём зря).
            usable = [k for k in ready if world.can_afford(k)]
            if not usable:
                result.waits += 1
                continue
            kind = rng.choice(usable)
            lane = rng.choice(["left", "right"])
        else:
            state = world.to_state(available=ready)
            result.decisions += 1
            # Сколько юнитов вообще можно позвать: если кандидат один, веса
            # не могли изменить решение — это видно в отчёте.
            if len(planner.candidates(state)) <= 1:
                result.single_candidate += 1
            action = planner.decide(state)
            if action.kind != "spawn" or not action.unit_kind:
                result.waits += 1
                continue
            save = planner.should_save_mana(state, action.unit_kind)
            if save is not None:
                result.waits += 1
                continue
            kind = action.unit_kind
            lane = action.lane or "left"

        if world.pay(kind):
            deploy = 0 if lane == "left" else 1
            world.spawn(kind, Team.ALLY, deploy=deploy)
            result.spawns += 1
            result.spawn_kinds[kind] = result.spawn_kinds.get(kind, 0) + 1
            result.trace.append((round(world.time, 2), kind, lane))

    result.winner = world.winner
    result.our_base = world.our_base_hp
    result.enemy_base = world.enemy_base_hp
    result.time = world.time
    result.kills = world.kills
    result.deaths = world.deaths
    result.spent_mana = world.spent_mana
    return result


def run_preset(base: AppConfig, name: str, seeds: int) -> dict:
    """Прогнать пресет по нескольким сидам и собрать сводку."""
    config = copy.deepcopy(base)
    for key, value in PRESETS.get(name, {}).items():
        setattr(config.planner, key, value)
    strategy = name if name in BASELINES else "planner"

    results = [play_match(config, seed, strategy) for seed in range(seeds)]
    wins = sum(1 for r in results if r.won)
    n = max(1, len(results))
    spawn_kinds: dict[str, int] = {}
    for r in results:
        for kind, count in r.spawn_kinds.items():
            spawn_kinds[kind] = spawn_kinds.get(kind, 0) + count
    decisions = sum(r.decisions for r in results)
    single = sum(r.single_candidate for r in results)
    ticks = sum(r.decisions for r in results) if strategy == "planner" else \
        sum(int(r.time / max(base.sim.dt, 1e-6)) for r in results)
    ready_total = sum(r.ready_cards_total for r in results)
    return {
        "preset": name,
        "settings": dict(PRESETS.get(name, {})) if name not in BASELINES else {},
        "matches": len(results),
        "wins": wins,
        "win_rate": wins / n,
        "avg_our_base": sum(r.our_base for r in results) / n,
        "avg_enemy_base": sum(r.enemy_base for r in results) / n,
        "avg_time": sum(r.time for r in results) / n,
        "avg_kills": sum(r.kills for r in results) / n,
        "avg_deaths": sum(r.deaths for r in results) / n,
        "avg_spent": sum(r.spent_mana for r in results) / n,
        "avg_spawns": sum(r.spawns for r in results) / n,
        "avg_waits": sum(r.waits for r in results) / n,
        "avg_no_card_ready": sum(r.no_card_ready for r in results) / n,
        "avg_ready_cards": ready_total / max(1, ticks),
        "spawn_kinds": spawn_kinds,
        # У случайных базовых линий решений планировщика нет — доля не считается.
        "single_candidate_share": (single / decisions) if decisions else None,
        "traces": [r.trace for r in results],
    }


def find_duplicates(rows: list[dict]) -> list[tuple[str, str]]:
    """Найти пресеты, которые дали полностью одинаковые решения."""
    duplicates: list[tuple[str, str]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["traces"] == b["traces"]:
                duplicates.append((a["preset"], b["preset"]))
    return duplicates


def format_table(rows: list[dict]) -> str:
    """Сводная таблица по пресетам."""
    header = (
        f"{'пресет':<17} {'партий':>6} {'побед':>6} {'доля':>7} {'время':>7} "
        f"{'наша база':>10} {'враг база':>10} {'убийств':>8} {'смертей':>8} "
        f"{'мана':>7} {'призывов':>9} {'ждали':>7} {'карточек':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['preset']:<17} {r['matches']:>6} {r['wins']:>6} {r['win_rate']:>6.1%} "
            f"{r['avg_time']:>7.0f} {r['avg_our_base']:>10.0f} {r['avg_enemy_base']:>10.0f} "
            f"{r['avg_kills']:>8.1f} {r['avg_deaths']:>8.1f} {r['avg_spent']:>7.0f} "
            f"{r['avg_spawns']:>9.1f} {r['avg_waits']:>7.0f} {r['avg_ready_cards']:>9.2f}"
        )
    return "\n".join(lines)


def format_spawn_table(rows: list[dict], kinds: list[str]) -> str:
    """Сколько юнитов каждого типа вызвали пресеты."""
    lines = ["| пресет | " + " | ".join(kinds) + " | всего |", "|---" * (len(kinds) + 2) + "|"]
    for r in rows:
        cells = [str(r["spawn_kinds"].get(k, 0)) for k in kinds]
        lines.append(f"| {r['preset']} | " + " | ".join(cells) + f" | {sum(r['spawn_kinds'].values())} |")
    return "\n".join(lines)


def build_report(rows: list[dict], args, duplicates: list[tuple[str, str]]) -> str:
    """Отчёт для файла."""
    kinds = sorted({k for r in rows for k in r["spawn_kinds"]})
    out: list[str] = []
    out.append("# Сравнение конфигураций планировщика")
    out.append("")
    out.append(f"- конфиг: `{args.config}`")
    out.append(f"- партий на пресет: {args.seeds}")
    if args.wave_size:
        out.append(f"- размер волны: {args.wave_size}")
    if args.base_hp:
        out.append(f"- здоровье баз: {args.base_hp}")
    if args.wave_interval:
        out.append(f"- интервал волн: {args.wave_interval}")
    out.append("")
    out.append("## Сводка")
    out.append("")
    out.append("| пресет | партий | побед | доля побед | время | наша база | враг база | "
               "убийств | смертей | мана | призывов | ждали | горящих карточек | кандидат один |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        share = r["single_candidate_share"]
        share_text = "—" if share is None else f"{share:.0%}"
        out.append(
            f"| {r['preset']} | {r['matches']} | {r['wins']} | {r['win_rate']:.1%} | "
            f"{r['avg_time']:.0f} | {r['avg_our_base']:.0f} | {r['avg_enemy_base']:.0f} | "
            f"{r['avg_kills']:.1f} | {r['avg_deaths']:.1f} | {r['avg_spent']:.0f} | "
            f"{r['avg_spawns']:.1f} | {r['avg_waits']:.0f} | {r['avg_ready_cards']:.2f} | {share_text} |"
        )
    out.append("")
    out.append("## Вызванные юниты по типам")
    out.append("")
    out.append(format_spawn_table(rows, kinds))
    out.append("")
    out.append("## Настройки пресетов")
    out.append("")
    out.append("| пресет | настройки |")
    out.append("|---|---|")
    for r in rows:
        if r["preset"] == RANDOM_BASELINE:
            settings = "случайный юнит из колоды (если карточка горит или не хватает маны — ждём)"
        elif r["preset"] == RANDOM_READY_BASELINE:
            settings = "случайный из готовых и оплачиваемых карточек"
        else:
            settings = ", ".join(f"{k}={v}" for k, v in r["settings"].items()) or "как в конфиге"
        out.append(f"| {r['preset']} | {settings} |")
    out.append("")
    if duplicates:
        out.append("## Одинаковые результаты")
        out.append("")
        for a, b in duplicates:
            out.append(f"- **{a}** и **{b}** дали полностью одинаковые решения: "
                       "разница настроек не изменила поведение.")
        out.append("")
    out.append("## Как это читать")
    out.append("")
    out.append("- «горящих карточек» — сколько карточек в среднем не горят (готовы "
               "к призыву) в каждый такт. Это и есть пространство выбора: планировщик "
               "выбирает среди них, а мана ограничивает отдельно.")
    out.append("- «кандидат один» — доля решений, где звать было можно ровно одного "
               "юнита (остальные карточки горят или не хватает маны). Чем она выше, "
               "тем слабее различаются пресеты.")
    out.append("- «ждали» — сколько раз планировщик решил не звать: не хватило маны, "
               "не горит ни одна карточка или ждём более сильного юнита.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение конфигураций планировщика")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--seeds", type=int, default=10, help="сколько партий на пресет")
    parser.add_argument("--compare", default="", help="пресеты через запятую")
    parser.add_argument("--wave-size", type=int, default=0, help="размер волны (0 — из конфига)")
    parser.add_argument("--base-hp", type=int, default=0, help="здоровье баз (0 — из конфига)")
    parser.add_argument("--wave-interval", type=float, default=0.0, help="интервал волн")
    parser.add_argument("--out", default="", help="куда сохранить отчёт (пусто — не сохранять)")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.wave_size:
        base.sim.wave_size = args.wave_size
    if args.base_hp:
        base.sim.base_hp = float(args.base_hp)
    if args.wave_interval:
        base.sim.wave_interval = args.wave_interval

    names = [n.strip() for n in args.compare.split(",") if n.strip()]
    if not names:
        names = list(PRESETS) + list(BASELINES)

    known = list(PRESETS) + list(BASELINES)
    rows = []
    for name in names:
        if name not in known:
            print(f"неизвестный пресет: {name} (доступны: {', '.join(known)})")
            continue
        rows.append(run_preset(base, name, args.seeds))

    if not rows:
        return 1

    print(format_table(rows))
    print()
    kinds = sorted({k for r in rows for k in r["spawn_kinds"]})
    print(format_spawn_table(rows, kinds))
    print()
    for r in rows:
        share = r["single_candidate_share"]
        share_text = "решений планировщика нет" if share is None else \
            f"кандидат был один в {share:.0%} решений"
        print(f"{r['preset']}: {share_text}, ждали {r['avg_waits']:.0f} раз(а) за партию, "
              f"горящих карточек в среднем {r['avg_ready_cards']:.2f}")

    duplicates = find_duplicates(rows)
    if duplicates:
        print("\nодинаковые результаты (разница настроек не изменила поведение):")
        for a, b in duplicates:
            print(f"  {a} == {b}")

    best = max(rows, key=lambda r: (r["win_rate"], r["avg_enemy_base"]))
    print(f"\nлучший пресет по доле побед: {best['preset']} ({best['win_rate']:.1%})")
    if len(rows) > 1 and len({r["win_rate"] for r in rows}) == 1:
        print("внимание: доля побед у всех пресетов одинакова — по ней не выбрать, "
              "смотрите длительность, базы и число призывов")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(build_report(rows, args, duplicates), encoding="utf-8")
        print(f"отчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
