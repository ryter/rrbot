"""Симулятор Rush Royale для исследований.

Зачем он нужен:
* даёт эталонную разметку (ground truth) — можно измерить точность
  детекции юнитов и чтения маны;
* позволяет прогонять планировщик в тысячах партий и сравнивать
  конфигурации весов без участия человека;
* служит источником синтетических кадров для отладки пайплайна.

Геометрия (как в оригинале, в портретной ориентации):
свой баз�� внизу, база противника вверху, поперёк поля проходит река,
которую пересекают два моста — по левой и правой линии. Юниты выходят
из своей базы по двум точкам высадки (левая и правая линия) и идут
к вражеской базе, переходя реку по своему мосту.

Все координаты — в пикселях канонического поля, то есть масштаб
совпадает с масштабом шаблонов (см. ``rrbot.assets``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .assets import ALLY_BGR, ENEMY_BGR, GRASS_ALT_BGR, GRASS_BGR, draw_unit
from .config import SimConfig
from .models import GameState, Lane, Team, Unit, UnitSpec

RIVER_COLOR = (140, 120, 90)
BRIDGE_COLOR = (120, 90, 60)
BASE_ALLY = (180, 110, 40)
BASE_ENEMY = (40, 40, 180)
HUD_BG = (38, 38, 38)
MANA_FILL = (240, 200, 60)
CARD_LIT = (78, 78, 78)
CARD_DARK = (52, 52, 52)
CARD_EDGE = (150, 150, 150)

AGGRO_MARGIN = 16.0  # запас к дальности атаки при захвате цели
BASE_REACH_FRACTION = 0.03  # доля высоты поля, считающаяся зоной базы


class _FromWorld:
    """Маркер «взять значение из мира симулятора»."""

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return "FROM_WORLD"


FROM_WORLD = _FromWorld()


@dataclass
class SimUnit:
    """Юнит внутри симулятора."""

    uid: int
    kind: str
    team: Team
    x: float
    y: float
    hp: float
    max_hp: float
    spec: UnitSpec
    cooldown: float = 0.0
    target: int | None = None
    deploy: int = 0  # 0 — левая точка высадки, 1 — правая
    alive: bool = True


class SimWorld:
    """Состояние игрового мира симулятора."""

    def __init__(self, config: SimConfig, specs: dict[str, UnitSpec], seed: int | None = None):
        self.config = config
        self.specs = specs
        self.rng = random.Random(config.seed if seed is None else seed)
        self.width = config.width
        self.height = config.height
        self.time = 0.0
        self.mana = 0.0
        self.tick = 0
        self.units: list[SimUnit] = []
        self.our_base_hp = config.base_hp
        self.enemy_base_hp = config.base_hp
        self.next_uid = 1
        self.wave_timer = config.wave_interval * 0.6
        self.wave_index = 0
        self.game_over = False
        self.winner: str | None = None
        self.kills = 0
        self.deaths = 0
        self.spent_mana = 0.0
        # Перезарядка карточек: сколько секунд ещё горит карточка типа.
        # Карточка горит после призыва юнита этого типа (время берётся из
        # ``UnitSpec.recharge``). Пусто в начале партии — все карточки готовы.
        self.card_cooldowns: dict[str, float] = {}
        self.cards_used: dict[str, int] = {}

        # Ключевые точки.
        # Река идёт поперёк поля, мосты — вертикальные полосы на линиях.
        self.river_y = self.height * 0.5
        self.river_half = self.height * 0.045
        self.lane_x = [self.width * 0.25, self.width * 0.75]
        self.bridge_half = self.width * 0.06
        self.our_base = (self.width * 0.5, self.height * 0.94)
        self.enemy_base = (self.width * 0.5, self.height * 0.06)
        self.deploy_points = [
            (self.lane_x[0], self.height * 0.86),
            (self.lane_x[1], self.height * 0.86),
        ]
        self.enemy_deploy_points = [
            (self.lane_x[0], self.height * 0.14),
            (self.lane_x[1], self.height * 0.14),
        ]
        self.base_reach = self.height * BASE_REACH_FRACTION
        self.spawn_points = {
            "left": Lane("left", *self.deploy_points[0]),
            "right": Lane("right", *self.deploy_points[1]),
        }

    # ------------------------------------------------------------- геометрия
    def on_bridge(self, x: float) -> bool:
        """Находится ли точка на мосту (то есть над рекой)."""
        return any(abs(x - lx) <= self.bridge_half for lx in self.lane_x)

    def near_river(self, y: float) -> bool:
        return abs(y - self.river_y) <= self.river_half

    def blocked_by_river(self, x: float, y: float) -> bool:
        """Река мешает проходу: точка над водой и не на мосту."""
        return self.near_river(y) and not self.on_bridge(x)

    def waypoints(self, team: Team) -> list[tuple[float, float]]:
        """Маршрут от своей базы к чужой: мост своей линии, затем база."""
        return []

    # ------------------------------------------------------------- сущности
    def spawn(self, kind: str, team: Team, deploy: int | None = None) -> SimUnit | None:
        spec = self.specs.get(kind)
        if spec is None:
            return None
        if deploy is None:
            deploy = self.rng.randint(0, 1)
        x, y = self.deploy_points[deploy] if team is Team.ALLY else self.enemy_deploy_points[deploy]
        unit = SimUnit(
            uid=self.next_uid,
            kind=kind,
            team=team,
            x=x,
            y=y,
            hp=spec.hp,
            max_hp=spec.hp,
            spec=spec,
            deploy=deploy,
        )
        self.next_uid += 1
        self.units.append(unit)
        return unit

    def can_afford(self, kind: str) -> bool:
        """Хватает ли маны на юнита этого типа."""
        spec = self.specs.get(kind)
        return spec is not None and self.mana >= spec.cost

    def card_ready(self, kind: str) -> bool:
        """Горит ли ещё карточка этого типа."""
        return self.card_time_left(kind) <= 0.0

    def card_time_left(self, kind: str) -> float:
        """Сколько секунд осталось до готовности карточки."""
        return max(0.0, self.card_cooldowns.get(kind, 0.0))

    def can_summon(self, kind: str) -> bool:
        """Можно ли позвать юнита прямо сейчас: карточка горит и не хватает маны."""
        return self.card_ready(kind) and self.can_afford(kind)

    def pay(self, kind: str) -> bool:
        """Оплатить призыв юнита и «поджечь» его карточку.

        Карточка начинает гореть на ``UnitSpec.recharge`` секунд: второй раз
        вызвать юнита этого типа раньше нельзя. Перезарядка не складывается:
        повторный призыв откладывает готовность карточки с нуля, а не
        ускоряет её.
        """
        spec = self.specs.get(kind)
        if spec is None or not self.can_summon(kind):
            return False
        self.mana -= spec.cost
        self.spent_mana += spec.cost
        self.cards_used[kind] = self.cards_used.get(kind, 0) + 1
        if spec.recharge > 0:
            self.card_cooldowns[kind] = spec.recharge
        else:
            self.card_cooldowns.pop(kind, None)
        return True

    def ready_kinds(self) -> list[str]:
        """Типы юнитов, карточки которых сейчас не горят (их можно звать).

        Это то, что в настоящей игре видно по подсвеченным карточкам:
        «доступность» не зависит от маны — горит карточка или нет.
        """
        return [k for k in self.specs if self.card_ready(k)]

    def summonable_kinds(self) -> list[str]:
        """Типы юнитов, которых можно позять прямо сейчас: карточка готова и хватает маны."""
        return [k for k in self.specs if self.can_summon(k)]

    def affordable_kinds(self) -> list[str]:
        """Историческое имя: карточки, доступные к призыву (не горят).

        Раньше здесь было «хватает маны». Теперь доступность карточек
        определяется перезарядкой, а хватает ли маны — проверяется отдельно
        (``can_summon`` / ``summonable_kinds``). Старое имя оставлено, чтобы
        не ломать вызовы из pipeline и утилит оценки зрения.
        """
        return self.ready_kinds()

    # ------------------------------------------------------------------ шаг
    def step(self, dt: float) -> None:
        if self.game_over:
            return
        self.time += dt
        self.tick += 1
        self.mana = min(self.config.mana_max, self.mana + self.config.mana_regen * dt)
        self._tick_cards(dt)

        self._spawn_wave(dt)
        self._act(dt)
        self._check_end()

    def _tick_cards(self, dt: float) -> None:
        """Горящие карточки остывают."""
        for kind in list(self.card_cooldowns):
            left = self.card_cooldowns[kind] - dt
            if left <= 0:
                self.card_cooldowns.pop(kind, None)
            else:
                self.card_cooldowns[kind] = left

    def _spawn_wave(self, dt: float) -> None:
        self.wave_timer -= dt
        if self.wave_timer > 0:
            return
        self.wave_timer = self.config.wave_interval
        self.wave_index += 1
        kinds = list(self.specs.keys())
        for _ in range(self.config.wave_size):
            self.spawn(self.rng.choice(kinds), Team.ENEMY)

    def _act(self, dt: float) -> None:
        for unit in self.units:
            if not unit.alive:
                continue
            unit.cooldown = max(0.0, unit.cooldown - dt)
            target = self._find_target(unit)
            unit.target = target.uid if target else None
            if target is not None:
                reach = unit.spec.attack_range + unit.spec.radius + target.spec.radius
                if math.hypot(target.x - unit.x, target.y - unit.y) <= reach:
                    if unit.cooldown <= 0:
                        target.hp -= unit.spec.damage
                        unit.cooldown = unit.spec.attack_interval
                        if target.hp <= 0:
                            target.alive = False
                            if target.team is Team.ENEMY:
                                self.kills += 1
                            else:
                                self.deaths += 1
                    continue
            self._advance(unit, dt)

        self.units = [u for u in self.units if u.alive]

        # Урон по базам: дошёл до вражеской базы — бьёт её.
        for unit in self.units:
            if unit.team is Team.ALLY and unit.y <= self.enemy_base[1] + self.base_reach:
                self.enemy_base_hp -= unit.spec.damage * dt
            elif unit.team is Team.ENEMY and unit.y >= self.our_base[1] - self.base_reach:
                self.our_base_hp -= unit.spec.damage * dt

    def _find_target(self, unit: SimUnit) -> SimUnit | None:
        """Ближайший враг в зоне захвата.

        Зона захвата зависит от дальности атаки: юнит цепляется за противника
        рядом, но не бежит за ним через всю карту. Иначе дальнобойные юниты
        уводят весь отряд и игра зависает.
        """
        best: SimUnit | None = None
        best_d = unit.spec.attack_range + unit.spec.radius + AGGRO_MARGIN
        for other in self.units:
            if other is unit or not other.alive or other.team is unit.team:
                continue
            d = math.hypot(other.x - unit.x, other.y - unit.y)
            if d < best_d:
                best_d = d
                best = other
        return best

    def _advance(self, unit: SimUnit, dt: float) -> None:
        """Движение к вражеской базе с переходом через мост своей линии.

        Логика:
        * по своей линии идём к реке;
        * над рекой выравниваемся по мосту своей линии;
        * перейдя реку, идём к базе противника;
        * если рядом враг на другой линии — можно сместиться на неё.
        """
        direction = -1.0 if unit.team is Team.ALLY else 1.0
        speed = unit.spec.speed
        lane_x = self.lane_x[unit.deploy]

        # Куда идём по вертикали.
        step_y = direction * speed * dt

        # Смещение по горизонтали: держим линию, кроме случаев у реки
        # (нужно попасть на мост) или когда рядом враг на другой линии.
        target_x = lane_x
        if self.near_river(unit.y) and not self.on_bridge(unit.x):
            target_x = lane_x  # мост своей линии
        else:
            rival = self._rival_in_other_lane(unit)
            if rival is not None:
                target_x = self.lane_x[rival.deploy]

        step_x = target_x - unit.x
        abs_x = abs(step_x)
        if abs_x > 1e-6:
            # Не двигаемся по x быстрее, чем по y, иначе юнит «плывёт» вбок.
            max_dx = speed * dt
            if abs_x > max_dx:
                step_x = math.copysign(max_dx, step_x)
            unit.x += step_x

        # Вертикальное движение останавливается, если мы над рекой не на мосту.
        if not (self.near_river(unit.y + step_y) and not self.on_bridge(unit.x)):
            unit.y += step_y
        else:
            # Подходим к мосту: ждём, пока выровняемся по нему.
            if self.on_bridge(unit.x):
                unit.y += step_y

        # Не заходим за базу противника (но и не отходим от неё дальше
        # зоны поражения).
        if unit.team is Team.ALLY:
            unit.y = max(unit.y, self.enemy_base[1] + self.base_reach)
        else:
            unit.y = min(unit.y, self.our_base[1] - self.base_reach)

    def _rival_in_other_lane(self, unit: SimUnit) -> SimUnit | None:
        """Ближайший враг на соседней линии, если он совсем рядом."""
        best: SimUnit | None = None
        best_d = 70.0
        for other in self.units:
            if other is unit or not other.alive or other.team is unit.team:
                continue
            if other.deploy == unit.deploy:
                continue
            d = math.hypot(other.x - unit.x, other.y - unit.y)
            if d < best_d:
                best_d = d
                best = other
        return best

    def _check_end(self) -> None:
        if self.enemy_base_hp <= 0:
            self.game_over = True
            self.winner = "ally"
        elif self.our_base_hp <= 0:
            self.game_over = True
            self.winner = "enemy"

    # -------------------------------------------------------------- выгрузка
    def to_state(self, available: Sequence[str] | None | _FromWorld = FROM_WORLD) -> GameState:
        """Собрать GameState приложения из состояния симулятора.

        ``available`` — какие карточки горят (доступны к призыву).
        По умолчанию берётся из мира: ``ready_kinds()``. Значение ``None``
        означает «информации нет» (планировщик считает доступными все типы),
        пустой список — «не горит ни одна карточка».
        """
        if isinstance(available, _FromWorld):
            available = self.ready_kinds()
        allies = [u for u in self.units if u.alive and u.team is Team.ALLY]
        enemies = [u for u in self.units if u.alive and u.team is Team.ENEMY]
        return GameState(
            time=self.time,
            tick=self.tick,
            mana=self.mana,
            mana_max=self.config.mana_max,
            mana_regen=self.config.mana_regen,
            ally_units=[
                Unit(uid=u.uid, kind=u.kind, team=u.team, x=u.x, y=u.y, hp=u.hp, max_hp=u.max_hp, spec=u.spec)
                for u in allies
            ],
            enemy_units=[
                Unit(uid=u.uid, kind=u.kind, team=u.team, x=u.x, y=u.y, hp=u.hp, max_hp=u.max_hp, spec=u.spec)
                for u in enemies
            ],
            spawn_points=dict(self.spawn_points),
            wave_index=self.wave_index,
            wave_active=True,
            available_units=tuple(available) if available is not None else None,
            card_cooldowns=dict(self.card_cooldowns),
            our_base_hp=self.our_base_hp / self.config.base_hp,
            enemy_base_hp=self.enemy_base_hp / self.config.base_hp,
            source="sim",
        )

    def ground_truth(self) -> list[tuple[str, Team, float, float]]:
        """Эталонные позиции юнитов (тип, сторона, x, y) для оценки детектора."""
        return [(u.kind, u.team, u.x, u.y) for u in self.units if u.alive]


class SimRenderer:
    """Отрисовка кадра симулятора: арена сверху, HUD снизу."""

    def __init__(self, config: SimConfig, specs: dict[str, UnitSpec]):
        self.config = config
        self.specs = specs
        self.hud_height = 220
        self.card_size = 84
        self.card_gap = 14

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.config.width, self.config.height + self.hud_height

    def card_rects(self) -> dict[str, tuple[int, int, int, int]]:
        """Прямоугольники карточек юнитов в экранных координатах."""
        kinds = list(self.specs.keys())
        total = len(kinds) * self.card_size + (len(kinds) - 1) * self.card_gap
        x0 = (self.config.width - total) // 2
        y0 = self.config.height + 96
        return {
            kind: (x0 + i * (self.card_size + self.card_gap), y0, self.card_size, self.card_size)
            for i, kind in enumerate(kinds)
        }

    def mana_rect(self) -> tuple[int, int, int, int]:
        w = int(self.config.width * 0.5)
        return (int(self.config.width * 0.25), self.config.height + 30, w, 24)

    def render(self, world: SimWorld, available: Sequence[str] | None = None) -> np.ndarray:
        """Отрисовать текущее состояние мира в BGR-кадр."""
        width, height = self.screen_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        arena_h = self.config.height
        frame[:arena_h] = GRASS_BGR
        stripe = max(4, self.config.height // 26)
        for y in range(0, arena_h, stripe * 2):
            frame[y : y + stripe, :] = GRASS_ALT_BGR

        # Река (поперёк поля) и мосты (вертикальные полосы на линиях).
        ry = int(world.river_y)
        rh = int(world.river_half * 2)
        cv2.rectangle(frame, (0, ry - rh // 2), (width, ry + rh // 2), RIVER_COLOR, -1)
        for lx in world.lane_x:
            bx = int(lx)
            bw = int(world.bridge_half * 2)
            cv2.rectangle(frame, (bx - bw // 2, ry - rh // 2), (bx + bw // 2, ry + rh // 2), BRIDGE_COLOR, -1)

        # Базы.
        base_r = int(self.config.width * 0.1)
        cv2.circle(frame, (int(world.our_base[0]), int(world.our_base[1])), base_r, BASE_ALLY, -1)
        cv2.circle(frame, (int(world.enemy_base[0]), int(world.enemy_base[1])), base_r, BASE_ENEMY, -1)

        # Юниты.
        for unit in world.units:
            if not unit.alive:
                continue
            color = ALLY_BGR if unit.team is Team.ALLY else ENEMY_BGR
            draw_unit(frame, unit.kind, unit.x, unit.y, color)

        self._render_hud(frame, world, available)
        return frame

    def _render_hud(self, frame: np.ndarray, world: SimWorld, available: Sequence[str] | None) -> None:
        arena_h = self.config.height
        cv2.rectangle(frame, (0, arena_h), (frame.shape[1], frame.shape[0]), HUD_BG, -1)

        # Полоса маны.
        x, y, w, h = self.mana_rect()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), -1)
        frac = 0.0 if self.config.mana_max <= 0 else max(0.0, min(1.0, world.mana / self.config.mana_max))
        fill = int(w * frac)
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + h), MANA_FILL, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), CARD_EDGE, 1)

        # Карточки юнитов: подсвечены, если юнит доступен.
        lit_set = set(available) if available is not None else set(self.specs.keys())
        for kind, (cx, cy, cw, ch) in self.card_rects().items():
            lit = kind in lit_set
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), CARD_LIT if lit else CARD_DARK, -1)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), CARD_EDGE, 1)
            icon_scale = (cw * 0.7) / 36.0
            draw_unit(
                frame,
                kind,
                cx + cw // 2,
                cy + ch // 2,
                ALLY_BGR if lit else (90, 90, 90),
                scale=icon_scale,
            )
