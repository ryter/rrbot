"""Модульные тесты ядра бота."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from rrbot.assets import ALLY_BGR, ENEMY_BGR, GRASS_BGR, draw_unit  # noqa: E402
from rrbot.config import load_config  # noqa: E402
from rrbot.geometry import Calibration, homography, screen_to_board, warp_to_board  # noqa: E402
from rrbot.models import GameState, Lane, Team, Unit  # noqa: E402
from rrbot.planner import Planner  # noqa: E402
from rrbot.sim import SimWorld  # noqa: E402
from rrbot.vision import VisionEngine  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config" / "default.yaml")


@pytest.fixture(scope="module")
def engine(config):
    eng = VisionEngine(config)
    eng.load_templates()
    return eng


def _planner(config) -> Planner:
    """Планировщик на копии конфига.

    Планировщик делит с конфигом один объект весов, поэтому тест, который
    меняет вес, испортил бы все следующие тесты. Работаем на копии.
    """
    return Planner(copy.deepcopy(config))


def _board_with(units: list[tuple[str, tuple[int, int], tuple[int, int, int]]]) -> np.ndarray:
    board = np.full((640, 360, 3), GRASS_BGR, dtype=np.uint8)
    for kind, (x, y), color in units:
        draw_unit(board, kind, x, y, color)
    return board


class TestVision:
    def test_шаблон_найден_в_своей_позиции(self, engine):
        board = _board_with([("knight", (100, 100), ALLY_BGR)])
        dets = engine.match_template(board, "knight", threshold=0.5)
        assert dets, "юнит не найден"
        best = max(dets, key=lambda d: d.score)
        assert abs(best.cx - 100) < 4
        assert abs(best.cy - 100) < 4
        assert best.score > 0.8

    def test_тип_юнита_определён_верно(self, engine):
        board = _board_with([("golem", (120, 200), ENEMY_BGR)])
        dets = engine.detect_units(board)
        assert dets
        assert dets[0].name == "golem"

    def test_сторона_определяется_по_цвету(self, engine):
        ally = engine.detect_units(_board_with([("knight", (100, 100), ALLY_BGR)]))
        enemy = engine.detect_units(_board_with([("knight", (100, 100), ENEMY_BGR)]))
        assert ally and ally[0].team is Team.ALLY
        assert enemy and enemy[0].team is Team.ENEMY

    def test_пустое_поле_не_даёт_ложных_срабатываний(self, engine):
        board = np.full((640, 360, 3), GRASS_BGR, dtype=np.uint8)
        dets = engine.detect_units(board)
        assert len(dets) == 0

    def test_юниты_только_одной_стороны(self, engine):
        """На поле только свои — все находки должны быть своими."""
        board = _board_with([
            ("knight", (100, 100), ALLY_BGR),
            ("golem", (200, 200), ALLY_BGR),
            ("archer", (300, 300), ALLY_BGR),
        ])
        dets = engine.detect_units(board)
        assert len(dets) == 3
        assert all(d.team is Team.ALLY for d in dets)

    def test_на_низком_пороге_находок_не_миллион(self, engine):
        """Низкий порог даёт тысячи кандидатов — они должны быть ограничены."""
        board = _board_with([("knight", (180, 320), ALLY_BGR)])
        dets = engine.match_template(board, "knight", threshold=0.5)
        assert len(dets) <= engine.vision.max_per_template
        # Настоящий юнит при этом всё равно найден.
        assert any(abs(d.cx - 180) < 4 and abs(d.cy - 320) < 4 for d in dets)

    def test_uid_юнита_стабилен_между_запусками(self, engine):
        """Идентификатор юнита не должен зависеть от рандомизации хэшей Python.

        ``hash()`` строки в Python меняется от запуска к запуску, поэтому
        один и тот же юнит получал бы разные uid и нельзя было бы связать
        наблюдения между сессиями.
        """
        board = _board_with([("knight", (100, 100), ALLY_BGR)])
        det = engine.detect_units(board)[0]
        expected = engine._uid_for(det)

        script = (
            "import sys, numpy as np;"
            f"sys.path.insert(0, {str(ROOT)!r});"
            "from rrbot.assets import ALLY_BGR, GRASS_BGR, draw_unit;"
            "from rrbot.config import load_config;"
            "from rrbot.vision import VisionEngine;"
            f"cfg = load_config({str(ROOT / 'config' / 'default.yaml')!r});"
            "eng = VisionEngine(cfg); eng.load_templates();"
            "b = np.full((640, 360, 3), GRASS_BGR, dtype=np.uint8);"
            "draw_unit(b, 'knight', 100, 100, ALLY_BGR);"
            "d = eng.detect_units(b)[0];"
            "print(eng._uid_for(d))"
        )
        uids = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True,
                env=env, check=True,
            )
            uids.add(int(out.stdout.strip()))
        assert len(uids) == 1, f"uid зависит от PYTHONHASHSEED: {uids}"
        assert uids.pop() == expected

    def test_энергия_в_пустой_полосе_нулевая(self, engine):
        """Полоски маны нет — читаем ноль, а не падаем."""
        frame = np.zeros((860, 360, 3), dtype=np.uint8)
        value, confidence = engine.read_mana(frame)
        assert value == 0.0
        assert confidence == 0.0

    def test_энергия_читается_по_доле(self, config, engine):
        from rrbot.sim import SimRenderer

        specs = config.vision.specs()
        world = SimWorld(config.sim, specs)
        renderer = SimRenderer(config.sim, specs)
        for expected in (100, 250, 500):
            world.mana = float(expected)
            frame = renderer.render(world, world.affordable_kinds())
            value, confidence = engine.read_mana(frame)
            assert abs(value - expected) < 8, f"ожидалось {expected}, получено {value}"
            assert confidence > 0.05


class TestGeometry:
    def test_гомография_переводит_углы(self):
        cal = Calibration(
            quad=((10.0, 20.0), (210.0, 20.0), (210.0, 420.0), (10.0, 420.0)),
            board_width=100,
            board_height=200,
        )
        matrix = homography(cal)
        assert matrix is not None
        points = screen_to_board([(10.0, 20.0), (210.0, 420.0)], cal)
        assert abs(points[0][0]) < 1.0 and abs(points[0][1]) < 1.0
        assert abs(points[1][0] - 99.0) < 1.5
        assert abs(points[1][1] - 199.0) < 1.5

    def test_варп_даёт_нужный_размер(self):
        cal = Calibration(
            quad=((0.0, 0.0), (360.0, 0.0), (360.0, 640.0), (0.0, 640.0)),
            board_width=180,
            board_height=320,
        )
        frame = np.zeros((640, 360, 3), dtype=np.uint8)
        out = warp_to_board(frame, cal)
        assert out.shape[:2] == (320, 180)


class TestPlanner:
    def _state(self, config, mana: float = 500.0) -> GameState:
        state = GameState(
            mana=mana,
            mana_max=500.0,
            spawn_points={"left": Lane("left", 90, 560), "right": Lane("right", 270, 560)},
        )
        state.available_units = tuple(config.vision.units.keys())
        return state

    def _unit(self, config, kind: str, team: Team, x: float = 100.0, y: float = 300.0,
              hp_fraction: float = 1.0) -> Unit:
        spec = config.vision.specs()[kind]
        return Unit(
            uid=id(spec) + len(kind) + int(x),
            kind=kind,
            team=team,
            x=x,
            y=y,
            hp=spec.hp * hp_fraction,
            max_hp=spec.hp,
            spec=spec,
        )

    def test_при_нулевой_мане_ждём(self, config):
        planner = _planner(config)
        action = planner.decide(self._state(config, mana=0.0))
        assert action.kind == "wait"
        assert "мана" in action.reason

    def test_при_достаточной_мане_выбираем_юнита(self, config):
        planner = _planner(config)
        action = planner.decide(self._state(config))
        assert action.kind == "spawn"
        assert action.unit_kind in config.vision.units
        assert action.lane in ("left", "right")

    def test_выбирается_более_эффективный_тип(self, config):
        planner = _planner(config)
        state = self._state(config)
        scores = {
            kind: planner.score_unit(kind, state, state.spawn_points["left"])[0]
            for kind in config.vision.units
        }
        best = max(scores, key=scores.get)
        action = planner.decide(state)
        assert action.unit_kind == best

    def test_накопление_маны_уступает_сильному_юниту(self, config):
        planner = _planner(config)
        # Механизм включаем явно: по умолчанию он выключен — см. README
        # (замеры показали, что копить ману в симуляторе невыгодно).
        planner.weights.save_mana_enabled = True
        planner.weights.weight_cost_efficiency = 2.0
        state = self._state(config, mana=90.0)
        # Готов только стрелок: по отдаче на единицу маны он слабее рыцаря.
        state.available_units = ("archer",)
        action = planner.decide(state)
        assert action.kind == "spawn"
        save = planner.should_save_mana(state, "archer")
        assert save is not None, "ожидалось решение копить ману"
        assert save.kind == "wait"

    def test_без_накопления_решаем_сразу(self, config):
        planner = _planner(config)
        planner.weights.save_mana_enabled = False
        state = self._state(config, mana=70.0)
        state.available_units = ("knight",)
        assert planner.should_save_mana(state, "knight") is None

    def test_без_доступных_юнитов_ждём(self, config):
        """Карточки известны и не горят ни одна — звать некого."""
        planner = _planner(config)
        state = self._state(config, mana=500.0)
        state.available_units = ()
        action = planner.decide(state)
        assert action.kind == "wait"
        assert "нет доступных" in action.reason

    def test_неизвестная_доступность_считается_за_все_типы(self, config):
        """Зрение ничего не сказало о карточках — считаем доступными все."""
        planner = _planner(config)
        state = self._state(config, mana=500.0)
        state.available_units = None
        assert planner.roster(state) == list(config.vision.units.keys())
        action = planner.decide(state)
        assert action.kind == "spawn"

    # ---------------------------------------------- сценарии выбора юнита
    def test_против_врага_другого_типа_берём_контр_юнита(self, config):
        """Враг-warden: его перебивает knight — его и зовём."""
        planner = _planner(config)
        state = self._state(config)
        state.enemy_units.append(self._unit(config, "warden", Team.ENEMY, x=90, y=300))
        action = planner.decide(state)
        assert action.unit_kind == "knight", f"ждали knight, получили {action.unit_kind}"
        # Пункт «counter» в разборе решения ненулевой и действительно
        # отвечает за выбор: без него победил бы другой тип.
        assert action.scores["counter"] > 0
        planner.weights.weight_counter = 0.0
        assert planner.decide(state).unit_kind != "knight"

    def test_контр_юнит_выигрывает_у_контр_юнита_по_эффективности(self, config):
        """Archer перебивает samurai — при прочих равных берём его."""
        planner = _planner(config)
        planner.weights.weight_cost_efficiency = 0.0
        state = self._state(config)
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=90, y=300))
        action = planner.decide(state)
        assert action.unit_kind == "archer"

    def test_нет_подходящего_юнита_копим_ману(self, config):
        """Дешёвый юнит доступен, но ждать выгоднее: копим ману."""
        planner = _planner(config)
        planner.weights.save_mana_enabled = True
        planner.weights.weight_cost_efficiency = 3.0
        state = self._state(config, mana=90.0)
        state.available_units = ("archer",)
        action = planner.decide(state)
        # Позвать можно только стрелка...
        assert action.kind == "spawn"
        assert action.unit_kind == "archer"
        # ...но ждать выгоднее: решение — ждать.
        save = planner.should_save_mana(state, "archer")
        assert save is not None, "ожидалось решение копить ману"
        assert save.kind == "wait"
        # Копим под того, у кого отдача на единицу маны лучше всего.
        best = max(config.vision.specs(), key=lambda k: config.vision.specs()[k].value_per_cost)
        assert best in save.reason
        assert "не хватает 50" in save.reason  # цена warden 140 - 90

    def test_копим_ману_на_самый_выгодный_юнит(self, config):
        """Из кандидатов на ожидание выбирается самый выгодный, а не первый."""
        planner = _planner(config)
        planner.weights.save_mana_enabled = True
        planner.weights.weight_cost_efficiency = 3.0
        state = self._state(config, mana=90.0)
        state.available_units = ("archer",)
        save = planner.should_save_mana(state, "archer")
        assert save is not None
        # Лучший по отдаче на единицу маны (warden) выигрывает у рыцаря
        # и самурая: выбираем его, а не первого попавшегося в списке.
        specs = config.vision.specs()
        best = max(specs, key=lambda k: specs[k].value_per_cost)
        assert best in save.reason
        assert specs[best].value_per_cost > specs["knight"].value_per_cost

    def test_синергия_не_даёт_забить_линию_одним_типом(self, config):
        """На поле уже два рыцаря — третьим рыцарем линию не забиваем."""
        planner = _planner(config)
        planner.weights.weight_cost_efficiency = 0.0
        state = self._state(config)
        state.ally_units.append(self._unit(config, "knight", Team.ALLY, x=90, y=500))
        state.ally_units.append(self._unit(config, "knight", Team.ALLY, x=110, y=500))

        knight_score = planner.score_unit("knight", state, None)[0]
        archer_score = planner.score_unit("archer", state, None)[0]
        assert knight_score < archer_score, "синергия не учтена"
        action = planner.decide(state)
        assert action.unit_kind != "knight", f"выбрали рыцаря: {action.unit_kind}"

    def test_синергия_работает_и_для_другого_типа(self, config):
        planner = _planner(config)
        planner.weights.weight_cost_efficiency = 0.0
        state = self._state(config)
        state.ally_units.append(self._unit(config, "archer", Team.ALLY, x=90, y=500))
        state.ally_units.append(self._unit(config, "archer", Team.ALLY, x=110, y=500))
        assert planner.score_unit("archer", state, None)[0] < planner.score_unit("knight", state, None)[0]
        assert planner.decide(state).unit_kind != "archer"

    def test_дешевле_выгоднее_по_эффективности(self, config):
        """При равных прочих выигрывает тот, у кого выгоднее отдача на ману."""
        planner = _planner(config)
        planner.weights.weight_counter = 0.0
        planner.weights.weight_pressure = 0.0
        planner.weights.weight_synergy = 0.0
        state = self._state(config)
        # Метрика берётся из конфига: у неё свой смысл и свой порядок
        # величины (tools/compare_cost_metric.py).
        efficiency = {k: planner.cost_efficiency(s) for k, s in config.vision.specs().items()}
        action = planner.decide(state)
        assert efficiency[action.unit_kind] == max(efficiency.values()), (
            f"выбрали {action.unit_kind} (эффективность {efficiency[action.unit_kind]:.2f}), "
            f"лучший — {max(efficiency, key=efficiency.get)}"
        )
        # В разборе решения «cost» ровно эффективность, умноженная на вес.
        assert action.scores["cost"] == pytest.approx(
            planner.weights.weight_cost_efficiency * efficiency[action.unit_kind]
        )

    def test_пустое_поле_даёт_выбор_без_давления_врага(self, config):
        """На пустом поле слагаемое давления равно нулю."""
        planner = _planner(config)
        state = self._state(config)
        _score, parts = planner.score_unit("knight", state, state.spawn_points["left"])
        assert parts["pressure"] == 0.0
        assert parts["counter"] == 0.0

    def test_только_свои_юниты_не_считаются_врагами(self, config):
        """Юниты одной стороны: давление считается по своим, врагов нет."""
        planner = _planner(config)
        state = self._state(config)
        state.ally_units.append(self._unit(config, "knight", Team.ALLY, x=90, y=500))
        pressures = planner.lane_pressure(state)
        assert pressures
        assert all(p.enemy_threat == 0.0 for p in pressures)
        assert all(p.ally_support > 0.0 for p in pressures)
        assert all(p.net < 0.0 for p in pressures)

    def test_на_пустом_поле_линия_выбирается_по_числу_своих(self, config):
        """Линии уравновешены — берём ту, где меньше своих юнитов."""
        planner = _planner(config)
        state = self._state(config)
        state.ally_units.append(self._unit(config, "knight", Team.ALLY, x=90, y=560))
        lane = planner._pick_lane(state, planner.lane_pressure(state))
        assert lane is not None
        assert lane.name == "right"

    def test_враг_прорвался_защищаем_его_линию(self, config):
        """Враг дошёл до нашей половины поля — зовём подкрепление в его линию."""
        planner = _planner(config)
        state = self._state(config)
        # Враг-самурай прорвался по левой линии и дошёл до середины поля.
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=100, y=400))
        lane = planner._pick_lane(state, planner.lane_pressure(state))
        assert lane is not None
        assert lane.name == "left", f"защищать надо его линию, а бот пошёл в «{lane.name}»"

    def test_враг_в_своей_половине_толкаем_в_другую(self, config):
        """Враг ещё не прорвался — не лезем в его линию, толкаем в другую."""
        planner = _planner(config)
        state = self._state(config)
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=100, y=200))
        lane = planner._pick_lane(state, planner.lane_pressure(state))
        assert lane is not None
        assert lane.name == "right", f"толкать надо в другую линию, а бот пошёл в «{lane.name}»"

    def test_без_линий_решение_всё_равно_принимается(self, config):
        """Точек высадки нет — не падаем, просто спавним без линии."""
        planner = _planner(config)
        state = self._state(config)
        state.spawn_points = {}
        action = planner.decide(state)
        assert action.kind == "spawn"
        assert action.lane is None
        assert action.position is None

    def test_нулевая_энергия_и_пустой_ростер_ждут(self, config):
        planner = _planner(config)
        state = self._state(config, mana=0.0)
        state.available_units = ()
        action = planner.decide(state)
        assert action.kind == "wait"

    # ------------------------------------------------- выбор при готовых карточках
    def test_при_нескольких_готовых_карточках_берём_контр_юнита(self, config):
        """Карточек горит несколько — берём не самый дешёвый, а контр-юнита."""
        planner = _planner(config)
        state = self._state(config, mana=500.0)
        # Готовы все карточки — выбирать есть из чего.
        state.available_units = tuple(config.vision.units.keys())
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=90, y=300))
        action = planner.decide(state)
        cheapest = min(config.vision.units, key=lambda k: config.vision.specs()[k].cost)
        assert action.unit_kind == "archer", f"ждали archer, получили {action.unit_kind}"
        assert action.unit_kind != cheapest, "взяли самый дешёвый юнит вместо контр-юнита"

    def test_контр_юнит_выбирается_по_горящим_карточкам(self, config):
        """То же самое, но на настоящем мире симулятора: карточки горят."""
        world = SimWorld(config.sim, config.vision.specs())
        world.mana = 500.0
        # На поле — враг-самурай, которого перебивает арчер.
        world.spawn("samurai", Team.ENEMY, deploy=0)
        state = world.to_state()
        assert len(state.available_units) == len(config.vision.units), "все карточки должны гореть"
        planner = _planner(config)
        action = planner.decide(state)
        cheapest = min(config.vision.units, key=lambda k: config.vision.specs()[k].cost)
        assert action.unit_kind == "archer"
        assert action.unit_kind != cheapest

    def test_горящая_карточка_исключается_из_выбора(self, config):
        """Единственный контр-юнит на горещей карточке — планировщик его не берёт."""
        planner = _planner(config)
        state = self._state(config, mana=500.0)
        state.available_units = ("knight", "bomber", "samurai")
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=90, y=300))
        action = planner.decide(state)
        assert action.unit_kind in ("knight", "bomber", "samurai")
        assert action.unit_kind != "archer", "взяли юнита с горящей карточки"

    def test_ждём_пока_восстановится_карточка(self, config):
        """Карточка горит — планировщик может подождать её восстановления."""
        planner = _planner(config)
        planner.weights.save_mana_enabled = True
        # Контр-счётство тут главное: арчер перебивает самурая, поэтому ждать
        # его восстановления выгоднее, чем звать рыцаря.
        planner.weights.weight_counter = 6.0
        state = self._state(config, mana=500.0)
        state.available_units = ("knight", "bomber")
        # Арчер горит ещё 3 секунды, но он сильнее рыцаря по контр-счёту.
        state.enemy_units.append(self._unit(config, "samurai", Team.ENEMY, x=90, y=300))
        state.card_cooldowns = {"archer": 3.0}
        action = planner.decide(state)
        assert action.kind == "spawn"
        save = planner.should_save_mana(state, action.unit_kind)
        assert save is not None, "ожидалось ожидание до восстановления карточки"
        assert "archer" in save.reason
        assert "горит" in save.reason


class TestSim:
    def test_мир_начинается_пустым(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        assert world.units == []
        assert world.mana == 0.0

    def test_спавн_списывает_ману(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        world.mana = 500.0
        assert world.pay("knight")
        assert world.mana < 500.0
        assert not world.pay("knight") or world.mana >= 0

    def test_энергия_восстанавливается(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        for _ in range(60):
            world.step(config.sim.dt)
        assert world.mana > 0

    def test_волны_врагов_появляются(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        for _ in range(int(config.sim.wave_interval / config.sim.dt) + 30):
            world.step(config.sim.dt)
        enemies = [u for u in world.units if u.team is Team.ENEMY]
        assert enemies, "волна врагов не появилась"

    def test_юниты_идут_к_базе(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        unit = world.spawn("knight", Team.ALLY, deploy=0)
        assert unit is not None
        start_y = unit.y
        for _ in range(120):
            world.step(config.sim.dt)
        assert unit.y < start_y, "юнит не движется к вражеской базе"

    def test_эталон_разметки_согласован_с_миром(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        world.spawn("knight", Team.ALLY, deploy=0)
        world.spawn("golem", Team.ENEMY, deploy=1)
        truth = world.ground_truth()
        assert len(truth) == 2
        assert {t[0] for t in truth} == {"knight", "golem"}


class TestПерезарядкаКарточек:
    """Перезарядка карточек: карточка горит и снова становится доступной."""

    def test_время_перезарядки_берётся_из_конфига(self, config):
        specs = config.vision.specs()
        assert specs["knight"].recharge > 0, "у карточки рыцаря должно быть время восстановления"
        assert specs["warden"].recharge > specs["knight"].recharge, (
            "у дорогого юнита карточка должна восстанавливаться дольше"
        )
        # Время восстановления — часть описания юнита, а не магическое число.
        assert config.vision.units["knight"].recharge == specs["knight"].recharge

    def test_карточка_становится_доступна_через_заданное_время(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        recharge = config.vision.specs()["knight"].recharge
        world.mana = 500.0
        assert world.pay("knight")
        # Сразу после призыва карточка горит.
        assert not world.card_ready("knight")
        assert "knight" not in world.ready_kinds()
        # За миг до истечения времени — всё ещё горит.
        world.step(recharge - config.sim.dt * 0.5)
        assert not world.card_ready("knight")
        # После истечения времени карточка снова доступна.
        world.step(config.sim.dt)
        assert world.card_ready("knight")
        assert "knight" in world.ready_kinds()

    def test_две_карточки_одного_типа_не_восстанавливаются_вдвое_быстрее(self, config):
        """Повторный призыв откладывает готовность карточки, а не ускоряет её."""
        world = SimWorld(config.sim, config.vision.specs())
        recharge = config.vision.specs()["knight"].recharge
        world.mana = 500.0
        assert world.pay("knight")
        world.step(recharge - config.sim.dt * 0.5)
        assert not world.card_ready("knight")
        # Карточка горит — второй раз этого типа позвать нельзя.
        assert not world.can_summon("knight")
        world.step(config.sim.dt)
        assert world.card_ready("knight")
        # Второй призыв: карточка снова горит на полный срок.
        assert world.pay("knight")
        assert not world.card_ready("knight")
        world.step(recharge * 0.5)
        assert not world.card_ready("knight"), "карточка восстановилась вдвое быстрее"
        world.step(recharge * 0.5 + config.sim.dt)
        assert world.card_ready("knight")

    def test_нельзя_позвать_юнита_с_горящей_карточкой(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        world.mana = 500.0
        assert world.pay("knight")
        assert not world.can_summon("knight")
        assert not world.pay("knight"), "призыв прошёл, хотя карточка горит"
        assert world.mana == 500.0 - config.vision.specs()["knight"].cost

    def test_доступность_не_зависит_от_маны(self, config):
        """Карточка горит — юнит недоступен, даже когда маны хватает."""
        world = SimWorld(config.sim, config.vision.specs())
        world.mana = 500.0
        assert world.pay("knight")
        assert world.can_afford("knight")
        assert not world.card_ready("knight")
        assert "knight" not in world.ready_kinds()
        # Готовые карточки и «можно позвать прямо сейчас» — разные вещи.
        assert "archer" in world.ready_kinds()
        assert "knight" not in world.summonable_kinds()

    def test_в_состоянии_симулятора_доступны_только_готовые_карточки(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        world.mana = 500.0
        world.pay("knight")
        state = world.to_state()
        assert state.available_units is not None
        assert "knight" not in state.available_units
        assert set(state.available_units) == set(world.ready_kinds())
        # Планировщик видит именно этот набор.
        planner = _planner(config)
        assert planner.roster(state) == list(world.ready_kinds())
        # И время до готовности горящей карточки известно.
        assert state.card_cooldowns["knight"] > 0


class TestСемантикаДоступности:
    """available_units=None — «информации нет», () — «не горит ни одна карточка»."""

    def test_none_означает_информации_нет(self, config):
        planner = _planner(config)
        state = GameState(mana=500.0, spawn_points={})
        state.available_units = None
        assert planner.roster(state) == list(config.vision.units.keys())
        assert planner.decide(state).kind == "spawn"

    def test_пустой_список_означает_не_горит_ни_одна(self, config):
        planner = _planner(config)
        state = GameState(mana=500.0, spawn_points={})
        state.available_units = ()
        assert planner.roster(state) == []
        action = planner.decide(state)
        assert action.kind == "wait"
        assert "нет доступных" in action.reason

    def test_в_состоянии_симулятора_можно_сообщить_об_очевидном(self, config):
        world = SimWorld(config.sim, config.vision.specs())
        # По умолчанию состояние берётся из мира.
        assert world.to_state().available_units == tuple(world.ready_kinds())
        # Явно «информации нет».
        assert world.to_state(None).available_units is None
        # Явно «не горит ни одна карточка».
        assert world.to_state([]).available_units == ()
