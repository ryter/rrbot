"""Типизированная конфигурация бота (YAML + значения по умолчанию)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .models import UnitSpec


@dataclass
class UnitConfig:
    """Настройки конкретного типа юнита."""

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
    # Время восстановления карточки в секундах: после призыва карточка
    # «горит» это время. Ноль — без перезарядки.
    recharge: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "UnitConfig":
        return cls(
            name=str(data["name"]),
            cost=int(data["cost"]),
            hp=float(data["hp"]),
            damage=float(data["damage"]),
            attack_interval=float(data.get("attack_interval", 1.0)),
            speed=float(data.get("speed", 20.0)),
            attack_range=float(data.get("attack_range", 20.0)),
            radius=float(data.get("radius", 9.0)),
            family=str(data.get("family", "melee")),
            counters=tuple(data.get("counters", ())),
            color=tuple(data.get("color", (200, 200, 200))),  # type: ignore[arg-type]
            recharge=float(data.get("recharge", 0.0)),
        )

    def to_spec(self) -> UnitSpec:
        return UnitSpec(
            name=self.name,
            cost=self.cost,
            hp=self.hp,
            damage=self.damage,
            attack_interval=self.attack_interval,
            speed=self.speed,
            attack_range=self.attack_range,
            radius=self.radius,
            family=self.family,
            counters=self.counters,
            color=self.color,
            recharge=self.recharge,
        )


@dataclass
class FactionConfig:
    """HSV-диапазоны для определения стороны по цвету юнита."""

    ally_hue_ranges: tuple[tuple[float, float], ...] = ((95.0, 135.0),)
    enemy_hue_ranges: tuple[tuple[float, float], ...] = ((0.0, 12.0), (168.0, 180.0))
    min_saturation: int = 60
    min_value: int = 40

    @classmethod
    def from_dict(cls, data: dict) -> "FactionConfig":
        def rng(v: Any) -> tuple[tuple[float, float], ...]:
            return tuple((float(a), float(b)) for a, b in v)

        return cls(
            ally_hue_ranges=rng(data.get("ally_hue_ranges", ((95, 135),))),
            enemy_hue_ranges=rng(data.get("enemy_hue_ranges", ((0, 12), (168, 180)))),
            min_saturation=int(data.get("min_saturation", 60)),
            min_value=int(data.get("min_value", 40)),
        )


@dataclass
class VisionConfig:
    """Параметры детектора."""

    templates_dir: str = "assets/templates"
    match_threshold: float = 0.62
    nms_iou: float = 0.35
    max_per_template: int = 24
    # Сколько самых сильных кандидатов оставлять до подавления дубликатов.
    # На низком пороге шаблон совпадает с тысячами пикселей, и без этого
    # ограничения подавление дубликатов считает миллионы пар за кадр.
    max_candidates: int = 256
    grayscale: bool = True
    patch_inset: float = 0.28  # доля отступа при замере цвета внутри найденного бокса
    units: dict[str, UnitConfig] = field(default_factory=dict)
    faction: FactionConfig = field(default_factory=FactionConfig)
    mana_bar: dict[str, float] = field(default_factory=lambda: {"x": 8.0, "y": 6.0, "w": 120.0, "h": 14.0})
    mana_max: float = 500.0
    mana_min_saturation: int = 80
    spawn_cards: dict[str, dict[str, float]] = field(default_factory=dict)
    card_min_saturation: int = 90
    card_min_saturated_fraction: float = 0.06
    # Явное сопоставление «тип юнита -> файл шаблона». Нужно для реальной
    # игры: одноимённый файл может содержать совсем другой спрайт.
    template_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "VisionConfig":
        units = {k: UnitConfig.from_dict({**v, "name": k}) for k, v in data.get("units", {}).items()}
        return cls(
            templates_dir=str(data.get("templates_dir", "assets/templates")),
            match_threshold=float(data.get("match_threshold", 0.62)),
            nms_iou=float(data.get("nms_iou", 0.35)),
            max_per_template=int(data.get("max_per_template", 24)),
            max_candidates=int(data.get("max_candidates", 256)),
            grayscale=bool(data.get("grayscale", True)),
            patch_inset=float(data.get("patch_inset", 0.28)),
            units=units,
            faction=FactionConfig.from_dict(data.get("faction", {})),
            mana_bar={k: float(v) for k, v in data.get("mana_bar", {}).items()},
            mana_max=float(data.get("mana_max", 500.0)),
            mana_min_saturation=int(data.get("mana_min_saturation", 80)),
            spawn_cards={k: {kk: float(vv) for kk, vv in v.items()} for k, v in data.get("spawn_cards", {}).items()},
            template_map={str(k): str(v) for k, v in data.get("template_map", {}).items()},
        )

    def spec(self, name: str) -> UnitSpec | None:
        uc = self.units.get(name)
        return uc.to_spec() if uc else None

    def specs(self) -> dict[str, UnitSpec]:
        return {name: uc.to_spec() for name, uc in self.units.items()}


@dataclass
class PlannerConfig:
    """Веса полезности планировщика."""

    weight_counter: float = 3.0
    weight_pressure: float = 2.5
    weight_synergy: float = 1.2
    weight_cost_efficiency: float = 1.0
    # Метрика эффективности по мане:
    #   "hp"    — живучесть на единицу цены (hp / цена);
    #   "hp_dps" — живучесть × урон на единицу цены, нормированные так же,
    #              как "hp" (значения сопоставимы, вес имеет один смысл).
    # Выбирается замером в симуляторе: tools/compare_cost_metric.py.
    cost_metric: str = "hp"
    weight_spawn_value: float = 0.6
    reserve_mana: float = 0.0
    max_spawn_per_tick: int = 1
    save_mana_enabled: bool = True
    save_mana_min_gain: float = 1.5
    # Максимальное ожидание в секундах: сколько можно ждать, пока догорает
    # карточка или копится мана под более сильного юнита.
    save_mana_max_wait: float = 8.0
    # Высота поля в канонических координатах: по ней решается, где середина
    # поля. Если враг перешёл середину — защищаем его линию, иначе толкаем
    # в другую (контр-пуш). Ноль — правило выключено (линия выбирается по
    # давлению, как раньше).
    defend_line_height: float = 640.0

    @classmethod
    def from_dict(cls, data: dict) -> "PlannerConfig":
        defaults = cls()
        for key, value in data.items():
            if hasattr(defaults, key):
                setattr(defaults, key, type(getattr(defaults, key))(value))
        return defaults


@dataclass
class CaptureConfig:
    """Захват кадра."""

    source: str = "sim"  # sim | screen
    region: tuple[int, int, int, int] = (0, 0, 1920, 1080)
    fps: float = 8.0
    monitor: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "CaptureConfig":
        region = tuple(int(v) for v in data.get("region", (0, 0, 1920, 1080)))
        return cls(
            source=str(data.get("source", "sim")),
            region=region,  # type: ignore[arg-type]
            fps=float(data.get("fps", 8.0)),
            monitor=int(data.get("monitor", 1)),
        )


@dataclass
class SimConfig:
    """Параметры симулятора."""

    width: int = 360
    height: int = 640
    mana_max: float = 500.0
    mana_regen: float = 34.0
    base_hp: float = 1200.0
    wave_interval: float = 14.0
    wave_size: int = 3
    max_duration: float = 300.0
    dt: float = 1.0 / 30.0
    seed: int = 7

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        defaults = cls()
        for key, value in data.items():
            if hasattr(defaults, key):
                setattr(defaults, key, type(getattr(defaults, key))(value))
        return defaults


@dataclass
class AppConfig:
    """Корневая конфигурация."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    allow_input: bool = False  # реальные клики по окну игры выключены по умолчанию
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            capture=CaptureConfig.from_dict(data.get("capture", {})),
            vision=VisionConfig.from_dict(data.get("vision", {})),
            planner=PlannerConfig.from_dict(data.get("planner", {})),
            sim=SimConfig.from_dict(data.get("sim", {})),
            allow_input=bool(data.get("allow_input", False)),
            log_level=str(data.get("log_level", "INFO")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Загрузить конфиг из YAML. Отсутствующие файлы не являются ошибкой."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return AppConfig.from_dict({})
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return AppConfig.from_dict(data)
