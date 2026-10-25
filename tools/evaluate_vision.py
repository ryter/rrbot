#!/usr/bin/env python3
"""Оценка точности детектора по эталонной разметке симулятора.

Прогоняет симулятор, каждый кадр сравнивает найденные детекции с истинными
позициями юнитов и печатает метрики: полнота (recall), точность (precision),
средняя ошибка позиции и точность определения стороны.

Это основной инструмент для подбора порогов и проверки шаблонов.

Как устроена выборка (важно для повторяемости):

* каждый кадр — **независимая** выборка: поле очищается и заново заполняется
  случайными юнитами (случайный тип, случайная сторона, случайная позиция);
* раньше юниты подселялись один раз и оставались на поле до своей смерти, то
  есть все кадры показывали одни и те же 8 юнитов. Метрики тогда зависели от
  одного случайного розыгрыша, а типы, не попавшие в него (например, bomber),
  вообще не измерялись. Теперь каждый кадр — новая выборка, все типы покрыты
  равномерно;
* ключ ``--motion`` возвращает старое поведение (юниты живут и двигаются), его
  имеет смысл включать, чтобы проверить, что детектор не теряет юниты в
  движении.

Отчёт печатается таблицей и сохраняется в файл (``--out``), чтобы результат
можно было приложить к изменению кода.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rrbot.config import load_config  # noqa: E402
from rrbot.geometry import Calibration  # noqa: E402
from rrbot.models import Team  # noqa: E402
from rrbot.sim import SimRenderer, SimWorld  # noqa: E402
from rrbot.vision import VisionEngine  # noqa: E402

MATCH_RADIUS = 26.0
DEFAULT_OUT = Path("/tmp/opencode/eval-vision.md")

TABLE_HEADER = (
    "| порог | recall | precision | тип | сторона | ошибка позиции |"
)
TABLE_RULE = "|---|---|---|---|---|---|"
DETAIL_HEADER = (
    "| порог | TP | FP | FN | истинных | находок | ошибка (средняя / медиана / p95) |"
)
DETAIL_RULE = "|---|---|---|---|---|---|---|"


def _blank_types() -> dict[str, Counter]:
    return defaultdict(Counter)


def evaluate(
    config,
    steps: int,
    threshold: float | None,
    seed: int,
    max_units: int = 8,
    motion: bool = False,
) -> dict:
    """Прогнать детектор по кадрам симулятора и посчитать метрики."""
    # Работаем на копии конфига: подстановка порога не должна менять исходный
    # конфиг (иначе следующий порог в переборе считался бы от предыдущего).
    config = copy.deepcopy(config)
    specs = config.vision.specs()
    world = SimWorld(config.sim, specs, seed=seed)
    renderer = SimRenderer(config.sim, specs)
    engine = VisionEngine(config)
    engine.vision.match_threshold = threshold if threshold is not None else engine.vision.match_threshold
    engine.load_templates()

    cal = Calibration(
        quad=((0.0, 0.0), (float(config.sim.width), 0.0),
              (float(config.sim.width), float(config.sim.height)), (0.0, float(config.sim.height))),
        board_width=config.sim.width,
        board_height=config.sim.height,
    )

    tp = fp = fn = 0
    type_errors = 0
    team_errors = 0
    pos_errors: list[float] = []
    matched_pairs: list[tuple] = []
    kinds = list(specs.keys())
    # Счётчики по типам: сколько настоящих юнитов нашли, сколько находок
    # оказалось лишними.
    per_type: dict[str, Counter] = _blank_types()
    team_confusion: Counter = Counter()

    for step in range(steps):
        world.step(config.sim.dt)
        if not motion:
            # Новая выборка на каждом кадре: иначе все кадры показывают одни
            # и те же юниты и метрики зависят от одного случайного розыгрыша.
            world.units.clear()
        while len(world.units) < max_units:
            kind = world.rng.choice(kinds)
            team = world.rng.choice([Team.ALLY, Team.ENEMY])
            unit = world.spawn(kind, team)
            if unit is not None:
                unit.x = float(world.rng.uniform(40, config.sim.width - 40))
                unit.y = float(world.rng.uniform(60, config.sim.height - 60))
                unit.hp = unit.max_hp * world.rng.uniform(0.3, 1.0)

        frame = renderer.render(world, world.affordable_kinds())
        state, _ = engine.analyze(frame, cal)

        truth = world.ground_truth()
        dets = state.detections

        used: set[int] = set()
        for kind, team, tx, ty in truth:
            best_i, best_d = -1, math.inf
            for i, det in enumerate(dets):
                if i in used:
                    continue
                d = math.hypot(det.cx - tx, det.cy - ty)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0 and best_d <= MATCH_RADIUS:
                used.add(best_i)
                det = dets[best_i]
                tp += 1
                pos_errors.append(best_d)
                per_type[kind]["tp"] += 1
                per_type[det.name]["tp"] += 1
                if det.name != kind:
                    type_errors += 1
                    per_type[kind]["wrong_type"] += 1
                    per_type[det.name]["wrong_type"] += 1
                if det.team is not team:
                    team_errors += 1
                    per_type[kind]["wrong_team"] += 1
                    team_confusion[(team.value, det.team.value if det.team else "?")] += 1
                matched_pairs.append((kind, det.name, det.team, team, best_d))
            else:
                fn += 1
                per_type[kind]["fn"] += 1
        for i, det in enumerate(dets):
            if i not in used:
                fp += 1
                per_type[det.name]["fp"] += 1

    total_truth = tp + fn
    total_det = tp + fp
    return {
        "frames": steps,
        "threshold": threshold,
        "truth": total_truth,
        "detected": total_det,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / total_truth if total_truth else 0.0,
        "precision": tp / total_det if total_det else 0.0,
        "type_accuracy": 1.0 - type_errors / tp if tp else 0.0,
        "team_accuracy": 1.0 - team_errors / tp if tp else 0.0,
        "mean_pos_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "median_pos_error": float(np.median(pos_errors)) if pos_errors else 0.0,
        "p95_pos_error": float(np.percentile(pos_errors, 95)) if pos_errors else 0.0,
        "confusions": Counter((t, d) for t, d, _, _, _ in matched_pairs),
        "team_confusions": team_confusion,
        "per_type": {k: dict(v) for k, v in per_type.items()},
    }


def format_table(results: list[dict]) -> str:
    """Собрать таблицу порогов в markdown (формат как в README)."""
    lines = [TABLE_HEADER, TABLE_RULE]
    for res in results:
        thr = res["threshold"]
        thr_text = "—" if thr is None else f"{thr:.2f}"
        lines.append(
            f"| {thr_text} | {res['recall']:.3f} | {res['precision']:.3f} | "
            f"{res['type_accuracy']:.3f} | {res['team_accuracy']:.3f} | "
            f"{res['mean_pos_error']:.2f} px |"
        )
    return "\n".join(lines)


def format_detail_table(results: list[dict]) -> str:
    """Подробная таблица: счётчики и распределение ошибки позиции."""
    lines = [DETAIL_HEADER, DETAIL_RULE]
    for res in results:
        thr = res["threshold"]
        thr_text = "—" if thr is None else f"{thr:.2f}"
        lines.append(
            f"| {thr_text} | {res['tp']} | {res['fp']} | {res['fn']} | "
            f"{res['truth']} | {res['detected']} | "
            f"{res['mean_pos_error']:.2f} / {res['median_pos_error']:.2f} / "
            f"{res['p95_pos_error']:.2f} px |"
        )
    return "\n".join(lines)


def format_per_type(res: dict) -> str:
    """Разбор по типам: что нашли, что перепутали, что потеряли."""
    rows = ["| тип | настоящих | найдено | recall | precision | ошибок типа | ошибок стороны |",
            "|---|---|---|---|---|---|---|"]
    for kind, counters in sorted(res["per_type"].items()):
        found = counters.get("tp", 0) + counters.get("fn", 0)
        predicted = counters.get("tp", 0) + counters.get("fp", 0)
        recall = counters.get("tp", 0) / found if found else 0.0
        precision = counters.get("tp", 0) / predicted if predicted else 0.0
        rows.append(
            f"| {kind} | {found} | {predicted} | {recall:.3f} | {precision:.3f} | "
            f"{counters.get('wrong_type', 0)} | {counters.get('wrong_team', 0)} |"
        )
    return "\n".join(rows)


def build_report(results: list[dict], args, config) -> str:
    """Текст отчёта для файла."""
    out: list[str] = []
    out.append("# Точность детекции юнитов (эталонная разметка симулятора)")
    out.append("")
    out.append(f"- дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.append(f"- конфиг: `{args.config}`")
    out.append(f"- шагов (кадров): {args.steps}, сид: {args.seed}")
    out.append(f"- порог в конфиге: {config.vision.match_threshold:.2f}, "
               f"NMS по IoU: {config.vision.nms_iou:.2f}")
    out.append(f"- выборка: {'юниты живут и двигаются' if args.motion else 'независимая выборка на каждом кадре'}")
    out.append(f"- юнитов в кадре: {args.units}")
    out.append("")
    out.append("## Пороги")
    out.append("")
    out.append(format_table(results))
    out.append("")
    out.append(format_detail_table(results))
    out.append("")
    for res in results:
        thr = res["threshold"]
        thr_text = "—" if thr is None else f"{thr:.2f}"
        out.append(f"### Порог {thr_text}")
        out.append("")
        out.append(f"- кадров: {res['frames']}, истинных юнитов: {res['truth']}, "
                   f"находок: {res['detected']}")
        out.append(f"- recall {res['recall']:.3f}, precision {res['precision']:.3f}, "
                   f"тип {res['type_accuracy']:.3f}, сторона {res['team_accuracy']:.3f}")
        out.append(f"- ошибка позиции: средняя {res['mean_pos_error']:.2f} px, "
                   f"медиана {res['median_pos_error']:.2f} px, p95 {res['p95_pos_error']:.2f} px")
        out.append("")
        out.append(format_per_type(res))
        out.append("")
        if res["confusions"]:
            pairs = {f"{t} -> {d}": n for (t, d), n in res["confusions"].items() if t != d}
            if pairs:
                out.append(f"- путаницы типа (истина -> предсказано): {pairs}")
        if res["team_confusions"]:
            out.append(f"- путаницы стороны (истина -> предсказано): {dict(res['team_confusions'])}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка точности детектора")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--units", type=int, default=8, help="сколько юнитов в кадре")
    parser.add_argument("--sweep", action="store_true", help="перебрать пороги")
    parser.add_argument("--motion", action="store_true",
                        help="юниты живут и двигаются (старая выборка, коррелированная)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="куда сохранить отчёт")
    args = parser.parse_args()

    config = load_config(args.config)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85] if args.sweep else [args.threshold]
    results = []
    for thr in thresholds:
        if thr is None:
            continue
        res = evaluate(config, args.steps, thr, args.seed, max_units=args.units, motion=args.motion)
        results.append(res)

    table = format_table(results)
    print(table)
    print()
    print(format_detail_table(results))
    for res in results:
        thr = res["threshold"]
        thr_text = "—" if thr is None else f"{thr:.2f}"
        conf = {f"{t}->{d}": n for (t, d), n in res["confusions"].items() if t != d}
        extra = f" | путаницы типа {conf}" if conf else ""
        print(f"порог {thr_text}: ошибка позиции — средняя {res['mean_pos_error']:.2f} px, "
              f"медиана {res['median_pos_error']:.2f} px, p95 {res['p95_pos_error']:.2f} px"
              f"{extra}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(results, args, config), encoding="utf-8")
    print(f"\nотчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
