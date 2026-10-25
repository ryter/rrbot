"""Планировщик: оценка полезности спавна и выбор цели.

Логика прозрачна и легко тюнится: каждая составляющая полезности —
отдельный вес в конфиге. Это позволяет проводить эксперименты
(см. ``tools/experiment.py``), сравнивая разные наборы весов.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from .config import AppConfig, PlannerConfig
from .models import Action, GameState, Lane, Team, Unit, UnitSpec

log = logging.getLogger(__name__)

# Шкала слагаемого «эффективность по мане»: метрика "hp_dps" (живучесть ×
# урон на единицу цены) делится на это число, чтобы её значения были
# сопоставимы с метрикой "hp" (живучесть на единицу цены) — иначе вес
# weight_cost_efficiency значил бы разное в зависимости от метрики.
VALUE_SCALE = 25.0


@dataclass
class LanePressure:
    """Оценка давления противника на конкретной линии."""

    lane: str
    enemy_threat: float = 0.0
    ally_support: float = 0.0

    @property
    def net(self) -> float:
        return self.enemy_threat - self.ally_support


class Planner:
    """Превращает наблюдаемое состояние в одно решение."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.weights: PlannerConfig = config.planner
        self.specs: dict[str, UnitSpec] = config.vision.specs()

    # ------------------------------------------------------------------ lanes
    def lane_pressure(self, state: GameState) -> list[LanePressure]:
        """Оценить давление по каждой линии спавна."""
        result: list[LanePressure] = []
        lanes = list(state.spawn_points.values())
        if not lanes:
            return result
        for lane in lanes:
            pressure = LanePressure(lane=lane.name)
            for enemy in state.enemy_units:
                dist = math.hypot(enemy.x - lane.x, enemy.y - lane.y)
                # Вклад падает с расстоянием: чем ближе к нашей базе, тем опаснее.
                weight = 1.0 / (1.0 + dist / 90.0)
                pressure.enemy_threat += enemy.spec.dps * weight * enemy.hp_fraction
            for ally in state.ally_units:
                dist = math.hypot(ally.x - lane.x, ally.y - lane.y)
                weight = 1.0 / (1.0 + dist / 90.0)
                pressure.ally_support += ally.spec.dps * weight * ally.hp_fraction
            result.append(pressure)
        return result

    def roster(self, state: GameState) -> list[str]:
        """Типы юнитов, которые вообще можно рассматривать.

        ``available_units is None`` — зрение не сообщило о карточках (информации
        нет, считаем доступными все типы). Пустой кортеж — карточки известны и
        не горит ни одна: значит, сейчас звать некого и надо ждать.
        """
        available = state.available_units
        if available is None:
            return list(self.specs.keys())
        return [k for k in self.specs if k in available]

    def card_cooldown(self, state: GameState, kind: str) -> float:
        """Сколько секунд ещё горит карточка этого типа (0 — карточка готова).

        Источники: ``GameState.card_cooldowns`` (время до готовности) и
        ``GameState.available_units`` (список горящих карточек, если зрение
        его сообщило). Если информации нет — считаем карточку готовой.
        """
        left = max(0.0, float(state.card_cooldowns.get(kind, 0.0)))
        if left > 0:
            return left
        available = state.available_units
        if available is not None and kind not in available:
            # Карточка не горит по списку, но времени ожидания не знаем:
            # считаем, что восстановится сразу (иначе ждать бессмысленно).
            return 0.0
        return 0.0

    def is_ready(self, state: GameState, kind: str) -> bool:
        """Готова ли карточка этого типа к призыву."""
        if self.card_cooldown(state, kind) > 0:
            return False
        available = state.available_units
        if available is not None and kind not in available:
            return False
        return True

    def candidates(self, state: GameState) -> list[str]:
        """Юниты, которых можно позвать прямо сейчас (хватает маны)."""
        usable = state.mana - self.weights.reserve_mana
        return [k for k in self.roster(state) if self.specs[k].cost <= usable]

    # -------------------------------------------------------------- scoring
    def cost_efficiency(self, spec: UnitSpec) -> float:
        """Отдача юнита на единицу маны — по метрике из конфига.

        "hp" — живучесть на единицу цены: танк с большой живучестью за те же
        деньги считается выгоднее стрелка.

        "hp_dps" — живучесть × урон на единицу цены: юнит без урона ничего
        не делает, сколько бы прочности у него ни было. Значения нормированы
        на VALUE_SCALE, чтобы порядок величины был как у метрики "hp".
        """
        if self.weights.cost_metric == "hp_dps":
            return spec.value_per_cost / VALUE_SCALE
        return spec.hp_per_cost

    def score_unit(
        self,
        kind: str,
        state: GameState,
        lane: Lane | None,
        assume_available: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Полезность конкретного типа юнита в текущей ситуации.

        ``assume_available`` нужен для оценки «будущего» спавна: когда юнит
        станет доступен, он получит бонус доступности, иначе сравнение
        с текущим спавном было бы некорректным.
        """
        spec = self.specs.get(kind)
        if spec is None:
            return 0.0, {}
        w = self.weights
        parts: dict[str, float] = {}

        # 1. Контр-счётство: какую долю вражеского урона мы перебиваем.
        # Считаем долю, а не сумму: иначе слагаемое зависит от числа врагов
        # на поле и перебивает все остальные слагаемые (сумма dps врага —
        # десятки, а остальные слагаемые — единицы).
        countered = [e for e in state.enemy_units if e.kind in spec.counters]
        counter_value = sum(e.hp_fraction * e.spec.dps for e in countered)
        total_threat = sum(e.hp_fraction * e.spec.dps for e in state.enemy_units)
        share = (counter_value / total_threat) if total_threat > 0 else 0.0
        parts["counter"] = w.weight_counter * share

        # 2. Давление на линии: спавн туда, где враг сильнее.
        # Слагаемое получается одинаковым для всех типов, поэтому на выбор
        # юнита оно не влияет — только показывает, насколько важна линия
        # (и не даёт оценке уйти в минус, когда врагов нет).
        pressure = 0.0
        if lane is not None:
            for enemy in state.enemy_units:
                dist = math.hypot(enemy.x - lane.x, enemy.y - lane.y)
                pressure += enemy.spec.dps * enemy.hp_fraction / (1.0 + dist / 120.0)
        parts["pressure"] = w.weight_pressure * pressure

        # 3. Синергия: не забивать линию одним типом.
        same = sum(1 for a in state.ally_units if a.kind == kind)
        total_ally = max(1, len(state.ally_units))
        parts["synergy"] = w.weight_synergy * (1.0 - 2.0 * same / total_ally)

        # 4. Эффективность по мане: отдача юнита на единицу маны (по метрике
        # из конфига). Метрики нормированы к одному порядку величины, поэтому
        # вес weight_cost_efficiency значит одно и то же при любой из них.
        parts["cost"] = w.weight_cost_efficiency * self.cost_efficiency(spec)

        # 5. Бонус за то, что юнит вообще доступен для спавна.
        # Если зрение ничего не сказало о карточках (None) — считаем, что
        # доступны все: иначе планировщик откажется звать в пустоту.
        available = state.available_units
        is_available = assume_available or available is None or kind in available
        parts["available"] = w.weight_spawn_value if is_available else 0.0

        total = sum(parts.values())
        return total, parts

    # -------------------------------------------------------------- decision
    def decide(self, state: GameState) -> Action:
        """Выбрать действие для текущего состояния."""
        # Ростер: юниты, о которых зрение сообщило как о доступных. Если
        # информации нет, считаем доступными всех.
        roster = self.roster(state)
        if not roster:
            # Карточки известны и не горят ни одна — звать некого.
            return Action(kind="wait", reason="нет доступных юнитов", score=0.0)

        # Кого можно заспавнить прямо сейчас: хватает маны.
        usable = state.mana - self.weights.reserve_mana
        spawnable = [k for k in roster if self.specs[k].cost <= usable]
        if not spawnable:
            need = min(self.specs[k].cost for k in roster)
            reserve = self.weights.reserve_mana
            reason = f"мана {state.mana:.0f} < стоимости {need:.0f}"
            if reserve > 0:
                reason += f" + резерва {reserve:.0f}"
            return Action(
                kind="wait",
                reason=reason,
                score=0.0,
            )

        lanes = self.lane_pressure(state)
        lane = self._pick_lane(state, lanes)

        best_kind = None
        best_score = -math.inf
        best_parts: dict[str, float] = {}
        for kind in spawnable:
            score, parts = self.score_unit(kind, state, lane)
            if score > best_score:
                best_score = score
                best_kind = kind
                best_parts = parts

        if best_kind is None:
            return Action(kind="wait", reason="нет кандидатов", score=0.0)

        if best_score < 0:
            return Action(
                kind="wait",
                reason=f"нет выгодного спавна (оценка {best_score:.2f})",
                score=best_score,
                scores=best_parts,
            )

        pos = (lane.x, lane.y) if lane else None
        reason = self._explain(best_kind, best_parts, lane)
        return Action(
            kind="spawn",
            unit_kind=best_kind,
            lane=lane.name if lane else None,
            position=pos,
            reason=reason,
            score=best_score,
            scores=best_parts,
        )

    def should_save_mana(self, state: GameState, best_kind: str) -> Action | None:
        """Решить, стоит ли подождать и накопить ману под более сильного юнита.

        Сравниваем лучший доступный сейчас юнит со всеми остальными: если
        какой-то юнит дал бы заметно больше пользы, а ожидание недолгое —
        выгодно повременить. Это воспроизводит поведение «копим ману под танк».

        Ждать бывает по двум причинам: карточка юнита ещё горит (не закончилась
        перезарядка) или маны пока не хватает. Время ожидания считается в
        секундах: ``max(время до готовности карточки, время накопления
        маны)`` и сравнивается с ``save_mana_max_wait``.

        Из кандидатов выбираем тот, у которого выигрыш больше, а не первый
        попавшийся: иначе порядок типов в конфиге определял бы решение.
        """
        if not self.weights.save_mana_enabled:
            return None
        if best_kind not in self.specs:
            return None
        lanes = self.lane_pressure(state)
        lane = self._pick_lane(state, lanes)
        current_score, _ = self.score_unit(best_kind, state, lane)

        usable = state.mana - self.weights.reserve_mana
        regen = state.mana_regen if state.mana_regen > 0 else 0.0
        best_gain: float | None = None
        best_kind_wait: str | None = None
        best_parts: dict[str, float] = {}
        best_wait = 0.0
        best_deficit = 0.0
        best_card = 0.0
        for kind, spec in self.specs.items():
            ready = self.is_ready(state, kind)
            deficit = spec.cost - usable
            if ready and deficit <= 0:
                continue  # доступен прямо сейчас — это текущий выбор
            card_left = self.card_cooldown(state, kind)
            mana_wait = (deficit / regen) if (deficit > 0 and regen > 0) else 0.0
            wait = max(card_left, mana_wait)
            if wait > self.weights.save_mana_max_wait:
                continue
            future_score, parts = self.score_unit(kind, state, lane, assume_available=True)
            gain = future_score - current_score
            if gain < self.weights.save_mana_min_gain:
                continue
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_kind_wait = kind
                best_parts = parts
                best_wait = wait
                best_deficit = max(0.0, deficit)
                best_card = card_left
        if best_kind_wait is None:
            return None
        if best_deficit > 0:
            reason = (
                f"копим ману на {best_kind_wait} (выигрыш {best_gain:+.2f}, "
                f"не хватает {best_deficit:.0f})"
            )
        else:
            reason = f"ждём карточку {best_kind_wait} (выигрыш {best_gain:+.2f})"
        if best_card > 0:
            reason += f", карточка горит ещё {best_card:.0f} с"
        return Action(
            kind="wait",
            reason=reason,
            score=current_score,
            scores=best_parts,
        )

    def _lane_threat(self, state: GameState, lane: Lane) -> float:
        """Угроза врага у линии (без нашей поддержки)."""
        threat = 0.0
        for enemy in state.enemy_units:
            dist = math.hypot(enemy.x - lane.x, enemy.y - lane.y)
            threat += enemy.spec.dps * enemy.hp_fraction / (1.0 + dist / 90.0)
        return threat

    def _guard_lane(self, state: GameState) -> Lane | None:
        """Линия, которую надо защищать: враг прорвался в нашу половину.

        Волны врага идут по одной линии и числом превосходят наши
        подкрепления: юниты, отправленные в ту же линию, гибнут в стычке и
        не доходят до базы врага. Поэтому, пока враг держится в своей
        половине, бот не лезет в его линию, а толкает в другую (контр-пуш);
        как только враг перешёл середину поля — защищаем его линию.

        Замерено в симуляторе (tools/experiment.py, волны 4/13): контр-пуш
        с защитой даёт 97% побед, выбор линии по давлению — 35%.
        """
        half = self.weights.defend_line_height * 0.5
        if half <= 0:
            return None
        # Из прорвавшихся берём того, кто ближе к нашей базе.
        intruders = [e for e in state.enemy_units if e.y > half]
        if not intruders:
            return None
        closest = max(intruders, key=lambda e: e.y)
        lanes = list(state.spawn_points.values())
        if not lanes:
            return None
        # Линии спавна лежат на одной высоте, поэтому линию врага узнаём по
        # горизонтали.
        return min(lanes, key=lambda lane: abs(closest.x - lane.x))

    def _pick_lane(self, state: GameState, pressures: list[LanePressure]) -> Lane | None:
        """Выбрать линию для призыва.

        Порядок:
        1. враг прорвался в нашу половину — защищаем его линию;
        2. иначе толкаем в линию, где врага меньше (контр-пуш);
        3. при равенстве — ту, где меньше наших юнитов.
        """
        if not pressures:
            return None
        lanes = list(state.spawn_points.values())
        if not lanes:
            return None

        guard = self._guard_lane(state)
        if guard is not None:
            return guard

        threats = {lane.name: self._lane_threat(state, lane) for lane in lanes}
        least = min(threats.values())
        # При равной угрозе берём линию, где меньше наших юнитов: не забиваем
        # одну линию. Считаем по горизонтали — линии спавна лежат на одной
        # высоте, и по вертикали каждый юнит попадал бы сразу в обе линии.
        candidates = [name for name, threat in threats.items() if threat <= least + 1e-9]
        best_name = candidates[0]
        if len(candidates) > 1:
            counts = {name: 0 for name in state.spawn_points}
            for ally in state.ally_units:
                nearest = min(lanes, key=lambda lane: abs(ally.x - lane.x))
                counts[nearest.name] += 1
            best_name = min(candidates, key=lambda name: counts[name])
        return state.spawn_points.get(best_name)

    def _explain(self, kind: str, parts: dict[str, float], lane: Lane | None) -> str:
        top = sorted(parts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        contrib = ", ".join(f"{k}={v:+.2f}" for k, v in top)
        where = f" на линию «{lane.name}»" if lane else ""
        return f"{kind}{where} ({contrib})"

    # ------------------------------------------------------------------ focus
    def suggest_focus(self, state: GameState) -> Action:
        """Подсказать юнита, на который стоит сосредоточить урон."""
        if not state.enemy_units:
            return Action(kind="wait", reason="нет целей")
        our_centre = self._centroid(state.ally_units)
        target = min(
            state.enemy_units,
            key=lambda e: math.hypot(e.x - our_centre[0], e.y - our_centre[1]) - 40.0 * e.hp_fraction,
        )
        return Action(
            kind="focus",
            target_uid=target.uid,
            reason=f"близкая цель {target.kind} (hp {target.hp_fraction:.0%})",
        )

    @staticmethod
    def _centroid(units: Sequence[Unit]) -> tuple[float, float]:
        if not units:
            return 0.0, 0.0
        return (
            sum(u.x for u in units) / len(units),
            sum(u.y for u in units) / len(units),
        )
