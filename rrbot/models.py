"""Доменные модели: спецификации юнитов, экземпляры, состояние игры."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class Team(str, Enum):
    """Сторона на поле."""

    ALLY = "ally"
    ENEMY = "enemy"

    @property
    def opposite(self) -> "Team":
        return Team.ENEMY if self is Team.ALLY else Team.ALLY


class UnitKind(str, Enum):
    """Типы юнитов, поддерживаемые симулятором и планировщиком."""

    KNIGHT = "knight"
    ARCHER = "archer"
    GOLEM = "golem"
    BOMBER = "bomber"
    SAMURAI = "samurai"
    WARDEN = "warden"


@dataclass(frozen=True)
class UnitSpec:
    """Статические характеристики типа юнита."""

    name: str
    cost: int
    hp: float
    damage: float
    attack_interval: float
    speed: float
    attack_range: float
    radius: float
    family: str
    counters: tuple[str, ...] = ()
    color: tuple[int, int, int] = (200, 200, 200)
    # Время восстановления карточки (секунды). После призыва карточка
    # «горит» это время и звать этот тип юнита снова нельзя. Ноль — карточка
    # восстанавливается мгновенно (в симуляторе так означает «без
    # перезарядки»).
    recharge: float = 0.0

    @property
    def dps(self) -> float:
        return self.damage / max(self.attack_interval, 1e-6)

    @property
    def hp_per_cost(self) -> float:
        return self.hp / max(self.cost, 1)

    @property
    def value_per_cost(self) -> float:
        """Отдача юнита на единицу маны: живучесть × урон.

        Юнит, который живёт дольше, наносит больше урона, поэтому ценность
        складывается из урона и живучести. Одной живучести мало: юнит без
        урона ничего не делает, сколько бы прочности у него ни было.
        """
        return self.hp * self.dps / max(self.cost, 1)


@dataclass
class Unit:
    """Живой юнит на поле (экземпляр спецификации)."""

    uid: int
    kind: str
    team: Team
    x: float
    y: float
    hp: float
    max_hp: float
    spec: UnitSpec
    cooldown: float = 0.0
    target_uid: int | None = None
    alive: bool = True

    @property
    def is_base(self) -> bool:
        return self.kind == "base"

    @property
    def hp_fraction(self) -> float:
        return 0.0 if self.max_hp <= 0 else max(0.0, min(1.0, self.hp / self.max_hp))

    def distance_to(self, other: "Unit") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def take_damage(self, amount: float) -> None:
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0.0
            self.alive = False


@dataclass
class Detection:
    """Результат детектирования объекта в канонических координатах."""

    name: str
    score: float
    cx: float
    cy: float
    box: tuple[int, int, int, int]
    team: Team | None = None
    hue: float | None = None
    saturation: float | None = None

    @property
    def area(self) -> int:
        x, y, w, h = self.box
        return w * h


@dataclass
class Lane:
    """Спавн-точка в конкретной полосе поля."""

    name: str
    x: float
    y: float


@dataclass
class GameState:
    """Наблюдаемое состояние поля — результат работы зрения."""

    time: float = 0.0
    tick: int = 0
    mana: float = 0.0
    mana_max: float = 500.0
    mana_confidence: float = 0.0
    ally_units: list[Unit] = field(default_factory=list)
    enemy_units: list[Unit] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    spawn_points: dict[str, Lane] = field(default_factory=dict)
    wave_index: int = 0
    wave_active: bool = False
    # Доступные к спавну типы юнитов. None — зрение не сообщило о карточках
    # (информации нет, планировщик считает доступными все типы). Пустой
    # кортеж — карточки известны и не горят ни одна: звать некого.
    available_units: tuple[str, ...] | None = None
    # Сколько секунд ещё горит каждая карточка (сколько осталось до
    # готовности). Непустой словарь — карточки известны, и для горящих
    # известно время ожидания. Пустой словарь — информации нет.
    card_cooldowns: dict[str, float] = field(default_factory=dict)
    # Сколько маны набегает в секунду. Нужно планировщику, чтобы оценить,
    # сколько ждать до нужной суммы маны. В живой игре это оценка.
    mana_regen: float = 34.0
    our_base_hp: float = 1.0
    enemy_base_hp: float = 1.0
    source: str = "sim"

    @property
    def all_units(self) -> list[Unit]:
        return [*self.ally_units, *self.enemy_units]

    def by_team(self, team: Team) -> list[Unit]:
        return self.ally_units if team is Team.ALLY else self.enemy_units

    def unit(self, uid: int) -> Unit | None:
        for u in self.all_units:
            if u.uid == uid:
                return u
        return None

    def spawn_lane_by_name(self, name: str) -> Lane | None:
        return self.spawn_points.get(name)


@dataclass
class Action:
    """Решение планировщика."""

    kind: str  # "spawn" | "focus" | "wait"
    unit_kind: str | None = None
    lane: str | None = None
    position: tuple[float, float] | None = None
    target_uid: int | None = None
    reason: str = ""
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "spawn":
            return f"спавн {self.unit_kind} на линию «{self.lane}» ({self.reason})"
        if self.kind == "focus":
            return f"фокус на юните #{self.target_uid} ({self.reason})"
        return f"ожидание ({self.reason})"


def units_by_kind(units: Iterable[Unit], kind: str) -> list[Unit]:
    return [u for u in units if u.kind == kind]


def closest(unit: Unit, others: Sequence[Unit]) -> Unit | None:
    best: Unit | None = None
    best_d = math.inf
    for other in others:
        if other is unit or not other.alive:
            continue
        d = unit.distance_to(other)
        if d < best_d:
            best_d = d
            best = other
    return best
