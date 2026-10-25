"""Прогон конвейера без графического интерфейса.

Нужен для замеров: быстро прогнать тысячи кадров и получить статистику
по решениям, задержке и качеству детекции.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

from .capture import SimSource, make_screen_source
from .config import AppConfig
from .geometry import Calibration
from .models import Action
from .pipeline import BotPipeline
from .sim import SimRenderer, SimWorld

log = logging.getLogger(__name__)


def make_calibration(config: AppConfig, source: str) -> Calibration:
    if source == "sim":
        w = float(config.sim.width)
        h = float(config.sim.height)
        return Calibration(
            quad=((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)),
            board_width=config.sim.width,
            board_height=config.sim.height,
        )
    x, y, w, h = config.capture.region
    return Calibration(
        quad=((float(x), float(y)), (float(x + w), float(y)),
              (float(x + w), float(y + h)), (float(x), float(y + h))),
        board_width=config.sim.width,
        board_height=config.sim.height,
    )


def run_headless(config: AppConfig, frames: int) -> int:
    """Прогнать N кадров и напечатать сводку."""
    source_name = config.capture.source
    if source_name == "sim":
        specs = config.vision.specs()
        world = SimWorld(config.sim, specs, seed=config.sim.seed)
        renderer = SimRenderer(config.sim, specs)
        source = SimSource(world, renderer, config.sim.dt, available_fn=world.affordable_kinds)
    else:
        source = make_screen_source(tuple(config.capture.region), config.capture.monitor)

    pipeline = BotPipeline(config, source, make_calibration(config, source_name))
    if source_name == "sim":
        pipeline.sync_card_boxes(renderer)

    actions: Counter[str] = Counter()
    spawn_kinds: Counter[str] = Counter()
    latencies: list[float] = []
    started = time.perf_counter()

    for i in range(frames):
        result = pipeline.process_once()
        if not result.ok:
            continue
        action = result.action
        if action is not None:
            actions[action.kind] += 1
            if action.kind == "spawn" and action.unit_kind:
                spawn_kinds[action.unit_kind] += 1
        latencies.append(result.duration_ms)
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            log.info("кадр %d, %.1f fps", i + 1, (i + 1) / max(elapsed, 1e-6))

    elapsed = time.perf_counter() - started
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    print()
    print("=== Итог headless-прогона ===")
    print(f"кадров обработано : {len(latencies)}")
    print(f"время             : {elapsed:.2f} с ({len(latencies)/max(elapsed,1e-6):.1f} fps)")
    print(f"кадр в среднем    : {avg:.1f} мс")
    print(f"действия          : {dict(actions)}")
    if spawn_kinds:
        print(f"спавны по типам   : {dict(spawn_kinds)}")
    return 0
