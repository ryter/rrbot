"""Тесты паритета планировщика C++ и Python-прототипа.

Планировщик на C++ — порт ``rrbot/planner.py``. Порт должен давать те же
числа на том же состоянии: иначе замеры симулятора (сделаны на Python) не
переносятся на живого бота.

Сценарии — те же, что в ``tests/test_core.py`` (Python-версия), поэтому
ожидания проверяются с обеих сторон.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rrbot.config import load_config  # noqa: E402
from rrbot.models import GameState, Lane, Team, Unit  # noqa: E402
from rrbot.planner import Planner  # noqa: E402

PLAN_TOOL = ROOT / "cpp" / "build" / "rrplan"

SCORES_RE = re.compile(r"^ОЦЕНКИ(?: D)? (?P<body>.*)$", re.MULTILINE)
DECISION_RE = re.compile(r"^СЦЕНАРИЙ (?P<name>[A-Z]) \| (?P<body>.*)$", re.MULTILINE)
SAVING_RE = re.compile(r"^НАКОПЛЕНИЕ (?P<yes>да|нет) \| (?P<reason>.*)$", re.MULTILINE)


def _specs(config):
    return config.vision.specs()


def _state(config, mana: float, units: list[tuple[str, Team, float, float]],
           available: tuple[str, ...]) -> GameState:
    """Состояние поля: мана, юниты и список готовых карточек."""
    specs = _specs(config)
    state = GameState(
        mana=mana,
        spawn_points={"left": Lane("left", 90, 560), "right": Lane("right", 270, 560)},
    )
    state.available_units = available
    for index, (kind, team, x, y) in enumerate(units):
        spec = specs[kind]
        unit = Unit(uid=index + 1, kind=kind, team=team, x=x, y=y, hp=spec.hp, max_hp=spec.hp,
                    spec=spec)
        if team is Team.ALLY:
            state.ally_units.append(unit)
        else:
            state.enemy_units.append(unit)
    return state


def _planner(config, **weights) -> Planner:
    planner = Planner(copy.deepcopy(config))
    for key, value in weights.items():
        setattr(planner.weights, key, value)
    return planner


def _cpp_output(scenario: str) -> str:
    result = subprocess.run([str(PLAN_TOOL), "--scenario", scenario], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, f"утилита упала:\n{result.stderr}"
    return result.stdout


# Сценарии: как в tests/test_core.py.
#   A — враг-warden, своего knight рядом: ждём контр-юнита.
#   B — маны нет: ждём.
#   D — готов только стрелок, мана на 90: копим под того, кто выгоднее.
#   E — враг прорвался в нашу половину: защищаем его линию.
SCENARIOS = {
    "A": dict(mana=500.0, units=[("warden", Team.ENEMY, 100.0, 300.0),
                                 ("knight", Team.ALLY, 120.0, 500.0)],
              available=("knight", "archer", "golem", "samurai", "warden"), weights={}),
    "B": dict(mana=0.0, units=[("warden", Team.ENEMY, 100.0, 300.0)],
              available=("knight", "archer", "golem", "samurai", "warden"), weights={}),
    "D": dict(mana=90.0, units=[],
              available=("archer",),
              weights={"save_mana_enabled": True, "weight_cost_efficiency": 3.0}),
    "E": dict(mana=500.0, units=[("samurai", Team.ENEMY, 100.0, 400.0)],
              available=("knight", "archer", "golem", "samurai", "warden"), weights={}),
}


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_оценки_совпадают(scenario):
    """Оценка каждого типа юнита — одна и та же в Python и в C++."""
    config = load_config(ROOT / "config" / "default.yaml")
    spec = SCENARIOS[scenario]
    planner = _planner(config, **spec["weights"])
    state = _state(config, spec["mana"], spec["units"], spec["available"])
    lane = state.spawn_points["left"]
    expected = {kind: planner.score_unit(kind, state, lane)[0] for kind in _specs(config)}

    match = SCORES_RE.search(_cpp_output(scenario))
    assert match, "в выводе нет оценок"
    for item in match.group("body").split():
        name, _, value = item.partition("=")
        assert name in expected, f"C++ считает неизвестный тип {name}"
        assert abs(float(value) - expected[name]) < 1e-4, (
            f"{scenario}/{name}: C++ {value} против Python {expected[name]:.5f}"
        )


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_решение_совпадает(scenario):
    """Решение, оценка и формулировка — одни и те же."""
    config = load_config(ROOT / "config" / "default.yaml")
    spec = SCENARIOS[scenario]
    planner = _planner(config, **spec["weights"])
    state = _state(config, spec["mana"], spec["units"], spec["available"])
    action = planner.decide(state)

    match = DECISION_RE.search(_cpp_output(scenario))
    assert match, "в выводе нет решения"
    body = match.group("body")
    assert f"решение={action.kind}" in body
    if action.unit_kind is not None:
        assert f"юнит={action.unit_kind}" in body
    cpp_score = float(re.search(r"оценка=([-\d.]+)", body).group(1))
    assert abs(cpp_score - action.score) < 1e-4, f"C++ {cpp_score} против Python {action.score}"
    # Формулировка причины совпадает до знака.
    reason = re.search(r'причина="([^"]*)"', body)
    assert reason, "в выводе нет причины"
    assert reason.group(1) == action.reason, (
        f"C++ «{reason.group(1)}» против Python «{action.reason}»"
    )


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
def test_накопление_маны_совпадает():
    """Решение «копим ману» — то же самое и с той же формулировкой."""
    config = load_config(ROOT / "config" / "default.yaml")
    spec = SCENARIOS["D"]
    planner = _planner(config, **spec["weights"])
    state = _state(config, spec["mana"], spec["units"], spec["available"])
    action = planner.decide(state)
    save = planner.should_save_mana(state, action.unit_kind)
    assert save is not None

    match = SAVING_RE.search(_cpp_output("D"))
    assert match, "в выводе нет решения о накоплении"
    assert match.group("yes") == "да"
    assert match.group("reason") == save.reason, (
        f"C++ «{match.group('reason')}» против Python «{save.reason}»"
    )
