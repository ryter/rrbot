"""Поток обработки: захват -> зрение -> планирование -> действие.

Работает вне GUI, чтобы его можно было тестировать и запускать без Qt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .capture import FrameSource
from .config import AppConfig
from .geometry import Calibration
from .input import ActionExecutor
from .models import Action, GameState, Team
from .planner import Planner
from .vision import VisionEngine

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Результат обработки одного кадра."""

    frame: np.ndarray | None
    board: np.ndarray | None
    state: GameState | None
    action: Action | None
    timestamp: float
    duration_ms: float = 0.0
    detections: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state is not None


class BotPipeline:
    """Полный конвейер обработки кадра."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        calibration: Calibration,
        executor: ActionExecutor | None = None,
    ):
        self.config = config
        self.source = source
        self.calibration = calibration
        self.vision = VisionEngine(config)
        self.vision.load_templates()
        self.planner = Planner(config)
        self.executor = executor
        self.last_result: FrameResult | None = None
        self.last_frame: np.ndarray | None = None
        self._frame_index = 0
        self.spawned: list[tuple[str, str]] = []  # (тип юнита, линия)

    def process_once(self) -> FrameResult:
        """Обработать один кадр."""
        started = time.perf_counter()
        frame = self.source.grab()
        self.last_frame = frame
        if frame is None:
            result = FrameResult(None, None, None, None, time.time())
            self.last_result = result
            return result

        available = self.available_units()
        state, board = self.vision.analyze(
            frame,
            self.calibration,
            spawn_points=self.spawn_points(),
            available=available,
        )
        # Время и номер тика берём из мира симулятора: зрение их не знает.
        world = getattr(self.source, "world", None)
        if world is not None:
            state.time = world.time
            state.tick = world.tick
        action = self.planner.decide(state)
        if action.kind == "spawn" and action.unit_kind:
            # Планировщик может решить, что выгоднее подождать и накопить ману.
            save = self.planner.should_save_mana(state, action.unit_kind)
            if save is not None:
                action = save

        if action.kind == "spawn" and action.unit_kind:
            self.apply_spawn(action)

        result = FrameResult(
            frame=frame,
            board=board,
            state=state,
            action=action,
            timestamp=time.time(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detections=state.detections,
        )
        self.last_result = result
        self._frame_index += 1
        return result

    def apply_spawn(self, action: Action) -> bool:
        """Применить спавн к симулятору (и/или отправить клики)."""
        if not action.unit_kind:
            return False
        applied = False
        world = getattr(self.source, "world", None)
        if world is not None:
            if world.pay(action.unit_kind):
                deploy = 0 if action.lane == "left" else 1
                world.spawn(action.unit_kind, Team.ALLY, deploy=deploy)
                self.spawned.append((action.unit_kind, action.lane or "left"))
                applied = True
        if self.executor is not None:
            self.executor.execute(action)
        return applied

    def available_units(self) -> tuple[str, ...]:
        """Список юнитов, доступных к спавну.

        Определяется по подсвеченным карточкам в HUD. Если карточки не
        настроены (нет координат), считаем доступными все юниты, чья
        стоимость не превышает текущую ману.
        """
        frame = self.last_frame
        if frame is not None and self.vision.vision.spawn_cards:
            detected = self.vision.detect_available(frame)
            if detected:
                return detected
        world = getattr(self.source, "world", None)
        if world is not None:
            return tuple(world.affordable_kinds())
        return tuple(self.config.vision.units.keys())

    def sync_card_boxes(self, renderer) -> None:
        """Подставить координаты карточек HUD от рендерера симулятора."""
        rects = renderer.card_rects()
        self.vision.vision.spawn_cards = {
            kind: {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
            for kind, (x, y, w, h) in rects.items()
        }

    def spawn_points(self):
        world = getattr(self.source, "world", None)
        if world is not None:
            return dict(world.spawn_points)
        return {}
