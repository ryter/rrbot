"""Регрессионный тест: планировщик должен выигрывать у случайного выбора.

Главный вопрос к планировщику: даёт ли он выигрыш или просто тратит энергию
хуже, чем случайный выбор карточки. Долгое время было наоборот — и причина
оказалась в выборе линии: волны врага идут по одной линии, юниты, отправленные
в ту же линию, гибнут в стычке, не доходя до базы врага.

Тест держит правило выбора линии (контр-пуш + защита прорвавшегося) и
сравнивает планировщик со случайным выбором карточки на частых волнах — там,
где результат различается (на обычных волнах выигрывают все).

Замер медленный (десятки партий), но это единственный тест, который
ловит именно «выигрыш», а не отдельные слагаемые оценки.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rrbot.config import load_config  # noqa: E402
from rrbot.models import Team  # noqa: E402
from rrbot.planner import Planner  # noqa: E402
from rrbot.sim import SimWorld  # noqa: E402

# Партий на стратегию. 30 партий — достаточно, чтобы разница в 20 и более
# процентов была видна уверенно, и недолго (около 20 секунд).
SEEDS = 30
# Волны: частые и тяжёлые — на обычных выигрывают все стратегии.
WAVE_SIZE = 4
WAVE_INTERVAL = 13


def _config():
    config = load_config(ROOT / "config" / "default.yaml")
    config.sim.wave_size = WAVE_SIZE
    config.sim.wave_interval = WAVE_INTERVAL
    return config


def _play(config, seed: int, strategy: str) -> bool:
    """Одна партия: "planner" или "random" (случайная карточка)."""
    specs = config.vision.specs()
    world = SimWorld(config.sim, specs, seed=seed)
    planner = Planner(config)
    rng = random.Random(seed * 7919 + 13)
    while not world.game_over and world.time < config.sim.max_duration:
        world.step(config.sim.dt)
        ready = world.ready_kinds()
        if not ready:
            continue
        if strategy == "random":
            # Случайная карточка из колоды: если она не готова или не хватает
            # маны — ждём. Это верхняя граница для «случайной игры».
            kind = rng.choice(sorted(specs.keys()))
            if not world.can_summon(kind):
                continue
            lane = rng.choice(["left", "right"])
        else:
            state = world.to_state(available=ready)
            action = planner.decide(state)
            if action.kind != "spawn" or not action.unit_kind:
                continue
            if planner.should_save_mana(state, action.unit_kind) is not None:
                continue
            kind = action.unit_kind
            lane = action.lane or "left"
        if world.pay(kind):
            world.spawn(kind, Team.ALLY, deploy=0 if lane == "left" else 1)
    return world.winner == "ally"


def test_планировщик_выигрывает_у_случайного_выбора():
    """Планировщик должен выигрывать заметно чаще случайного выбора."""
    config = _config()
    planner_wins = sum(_play(config, seed, "planner") for seed in range(SEEDS))
    random_wins = sum(_play(config, seed, "random") for seed in range(SEEDS))
    assert planner_wins > random_wins, (
        f"планировщик {planner_wins}/{SEEDS}, случайный выбор {random_wins}/{SEEDS} — "
        "планировщик должен выигрывать чаще (проверь правило выбора линии)"
    )
    # Запас сверх «просто не хуже»: разница должна быть заметной.
    assert planner_wins - random_wins >= 4, (
        f"планировщик {planner_wins}/{SEEDS}, случайный выбор {random_wins}/{SEEDS} — "
        "разница слишком мала, чтобы считать это выигрышем"
    )


def test_правило_линии_даёт_выигрыш():
    """То же сравнение, но с отключённым правилом линии (старое поведение).

    Если выигрыш держится и без правила — значит, правило перестало работать
    (или перестало быть нужным).
    """
    config = _config()
    config.planner.defend_line_height = 0.0  # середина поля не ищется
    planner_wins = sum(_play(config, seed, "planner") for seed in range(SEEDS))
    random_wins = sum(_play(config, seed, "random") for seed in range(SEEDS))
    assert planner_wins <= random_wins, (
        f"без правила линии планировщик выигрывает чаще случайного выбора "
        f"({planner_wins}/{SEEDS} против {random_wins}/{SEEDS}) — значит, "
        "правило линии перестало быть причиной выигрыша"
    )
