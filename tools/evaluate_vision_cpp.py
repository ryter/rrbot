#!/usr/bin/env python3
"""Замер точности C++-детектора по эталонной разметке.

Аналог tools/evaluate_vision.py, но измеряет C++-детектор (cpp/src/vision.cpp).

Два режима:

* ``sim`` — эталонная разметка симулятора (rrbot/sim.py). Кадры и разметка
  генерируются тем же способом, что в tools/evaluate_vision.py, поэтому числа
  Python- и C++-детекторов можно сравнивать напрямую;
* ``real`` — готовые кадры и готовая разметка (файлы кадров + TSV). По умолчанию
  берётся набор из репозитория: assets/eval_real (12 кадров боя из
  Android-эмулятора с разметкой, подготовленной скриптом
  tools/prepare_real_frames.py).

Примеры:

    .venv/bin/python tools/evaluate_vision_cpp.py --mode sim --steps 60
    .venv/bin/python tools/evaluate_vision_cpp.py --mode sim --sweep --compare
    .venv/bin/python tools/evaluate_vision_cpp.py --mode real
    .venv/bin/python tools/evaluate_vision_cpp.py --mode real --scale 1.2 \\
        --threshold 0.60 --team-by-position --split 0.5

О наборе assets/eval_real: эталоны в assets/real сняты с веб-клиента, а в
мобильном клиенте юниты выглядят иначе, поэтому совпадения там единичные и
точность низкая (см. отчёт). Разметка описывает только положение юнитов: тип
по основанию не определяется, стоит «-».
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rrbot.config import load_config  # noqa: E402
from rrbot.geometry import Calibration  # noqa: E402
from rrbot.models import Team  # noqa: E402

TOOL = ROOT / "cpp" / "build" / "rrvision"
MATCH_RADIUS = 26.0
# Строки, которые печатает C++-утилита.
SUMMARY_RE = re.compile(r"^ИТОГО: (.*)$")
KIND_RE = re.compile(r"^ПО ТИПАМ: (\S+)\s*(.*)$")


def _split(text: str) -> dict[str, str]:
    """Разобрать «ключ=значение | ключ=значение | ...»."""
    out: dict[str, str] = {}
    for item in text.split("|"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_summary(output: str) -> dict[str, str]:
    """Разобрать сводную строку C++-утилиты."""
    for line in output.splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match:
            return _split(match.group(1))
    return {}


def parse_kinds(output: str) -> dict[str, dict[str, str]]:
    """Разобрать строки «ПО ТИПАМ»."""
    out: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        match = KIND_RE.match(line.strip())
        if match:
            out[match.group(1)] = _split(match.group(2))
    return out


def run_tool(args: list[str]) -> str:
    """Запустить C++-утилиту и вернуть её вывод."""
    result = subprocess.run([str(TOOL), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"C++-утилита завершилась с кодом {result.returncode}")
    return result.stdout


# ------------------------------------------------------------------ симулятор
def build_dataset(config, steps: int, seed: int, max_units: int, out_dir: Path,
                  motion: bool = False) -> Path:
    """Сгенерировать кадры и эталонную разметку симулятора.

    Выборка повторяет tools/evaluate_vision.py: по умолчанию каждый кадр —
    независимая выборка (поле очищается, юниты расставляются заново), иначе
    все кадры показывают одни и те же юниты и метрики зависят от одного
    розыгрыша. Ключ ``motion`` возвращает старое поведение (юниты живут).
    """
    from rrbot.sim import SimRenderer, SimWorld

    specs = config.vision.specs()
    world = SimWorld(config.sim, specs, seed=seed)
    renderer = SimRenderer(config.sim, specs)
    kinds = list(specs.keys())

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    truth_path = out_dir / "truth.tsv"
    with truth_path.open("w", encoding="utf-8") as truth:
        for step in range(steps):
            world.step(config.sim.dt)
            if not motion:
                # Новая выборка на каждом кадре.
                world.units.clear()
            # Подселяем юнитов: иначе эталон пуст и метрики бессмысленны.
            while len(world.units) < max_units:
                kind = world.rng.choice(kinds)
                team = world.rng.choice([Team.ALLY, Team.ENEMY])
                unit = world.spawn(kind, team)
                if unit is not None:
                    unit.x = float(world.rng.uniform(40, config.sim.width - 40))
                    unit.y = float(world.rng.uniform(60, config.sim.height - 60))
                    unit.hp = unit.max_hp * world.rng.uniform(0.3, 1.0)

            frame = renderer.render(world, world.affordable_kinds())
            name = f"f{step:04d}.png"
            cv2.imwrite(str(frames_dir / name), frame)
            for kind, team, tx, ty in world.ground_truth():
                truth.write(f"{name}\t{kind}\t{team.value}\t{tx:.2f}\t{ty:.2f}\n")
    return truth_path


def python_metrics(config, steps: int, seed: int, max_units: int, threshold: float | None,
                   motion: bool = False) -> dict:
    """Метрики Python-детектора на тех же кадрах (для сравнения с C++)."""
    from rrbot.sim import SimRenderer, SimWorld
    from rrbot.vision import VisionEngine

    specs = config.vision.specs()
    world = SimWorld(config.sim, specs, seed=seed)
    renderer = SimRenderer(config.sim, specs)
    engine = VisionEngine(config)
    if threshold is not None:
        engine.vision.match_threshold = threshold
    engine.load_templates()
    cal = Calibration(
        quad=((0.0, 0.0), (float(config.sim.width), 0.0),
              (float(config.sim.width), float(config.sim.height)), (0.0, float(config.sim.height))),
        board_width=config.sim.width,
        board_height=config.sim.height,
    )
    kinds = list(specs.keys())
    tp = fp = fn = 0
    type_errors = 0
    team_errors = 0
    pos_errors: list[float] = []
    for _ in range(steps):
        world.step(config.sim.dt)
        if not motion:
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
                if det.name != kind:
                    type_errors += 1
                if det.team is not team:
                    team_errors += 1
            else:
                fn += 1
        fp += len(dets) - len(used)
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "type": 1.0 - type_errors / tp if tp else 0.0,
        "team": 1.0 - team_errors / tp if tp else 0.0,
        "pos": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def print_result(label: str, summary: dict[str, str], kinds: dict[str, dict[str, str]]) -> None:
    print(
        f"{label} | recall {summary.get('recall', '?')} | precision {summary.get('precision', '?')} | "
        f"тип {summary.get('тип', '?')} | сторона {summary.get('сторона', '?')} | "
        f"ошибка позиции {summary.get('ошибка позиции', '?')} | "
        f"TP {summary.get('TP', '?')} FP {summary.get('FP', '?')} FN {summary.get('FN', '?')}"
    )
    for kind, stats in kinds.items():
        print(
            f"   {kind}: TP {stats.get('TP', '?')} FP {stats.get('FP', '?')} FN {stats.get('FN', '?')} "
            f"recall {stats.get('recall', '?')} precision {stats.get('precision', '?')}"
        )


# --------------------------------------------------------------------- запуск
def main() -> int:
    parser = argparse.ArgumentParser(description="Замер точности C++-детектора юнитов")
    parser.add_argument("--mode", choices=("sim", "real"), default="sim")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="перебрать пороги")
    parser.add_argument("--compare", action="store_true", help="сравнить с Python-детектором")
    parser.add_argument("--motion", action="store_true",
                        help="юниты живут и двигаются (старая выборка, коррелированная)")
    # Режим real: готовые кадры и разметка.
    parser.add_argument("--frames", default="")
    parser.add_argument("--truth", default="")
    parser.add_argument("--templates", default=str(ROOT / "assets" / "real"))
    parser.add_argument("--map", default="")
    parser.add_argument("--types", default="")
    parser.add_argument("--arena", default="")
    parser.add_argument("--board", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=MATCH_RADIUS)
    parser.add_argument("--nms", type=float, default=0.30)
    parser.add_argument("--max-per-template", type=int, default=24)
    parser.add_argument("--all-templates", action="store_true")
    parser.add_argument("--team-by-position", action="store_true")
    parser.add_argument("--split", type=float, default=0.5,
                        help="доля высоты поля, выше которой юнит считается чужим")
    parser.add_argument("--print-detections", action="store_true")
    # Поиск юнитов по круглому основанию (мобильный клиент).
    parser.add_argument("--by-base", action="store_true",
                        help="искать юниты по круглому основанию, а не по эталону")
    parser.add_argument("--base-radius", default="",
                        help="диапазон радиуса основания: «min,max»")
    parser.add_argument("--base-min-score", type=float, default=None,
                        help="минимальная доля окружности, на которой нашёлся край")
    parser.add_argument("--base-gradient", type=float, default=None,
                        help="порог градиента при поиске оснований, процент краёв")
    parser.add_argument("--classify-radius", type=int, default=None,
                        help="насколько искать положение эталона относительно основания")
    args = parser.parse_args()

    if not TOOL.exists():
        print(f"не собрана утилита {TOOL} (cd cpp && make)", file=sys.stderr)
        return 1

    config = load_config(args.config)
    work_dir = Path(tempfile.mkdtemp(prefix="rrvision-"))
    try:
        if args.mode == "sim":
            # Порог по умолчанию — из конфига симулятора.
            default_threshold = config.vision.match_threshold
            thresholds = [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.85] if args.sweep else [args.threshold]
            if thresholds == [None]:
                thresholds = [default_threshold]
            truth_path = build_dataset(config, args.steps, args.seed, args.max_units, work_dir,
                                       args.motion)
            frames_dir = work_dir / "frames"
            kinds = list(config.vision.specs().keys())
            ally = ",".join(f"{a:g}-{b:g}" for a, b in config.vision.faction.ally_hue_ranges)
            enemy = ",".join(f"{a:g}-{b:g}" for a, b in config.vision.faction.enemy_hue_ranges)
            for thr in thresholds:
                cmd = [
                    "--frames", str(frames_dir),
                    "--truth", str(truth_path),
                    "--templates", str(ROOT / config.vision.templates_dir),
                    "--types", ",".join(kinds),
                    "--arena", f"0,0,{config.sim.width},{config.sim.height}",
                    "--board", f"{config.sim.width}x{config.sim.height}",
                    "--threshold", f"{thr:.2f}",
                    "--nms", f"{config.vision.nms_iou:.2f}",
                    "--max-per-template", str(config.vision.max_per_template),
                    "--radius", f"{MATCH_RADIUS:.1f}",
                    "--ally-hues", ally,
                    "--enemy-hues", enemy,
                ]
                output = run_tool(cmd)
                summary = parse_summary(output)
                print_result(f"C++ порог {thr:.2f}", summary, parse_kinds(output))
                if args.compare:
                    ref = python_metrics(config, args.steps, args.seed, args.max_units, thr, args.motion)
                    print(
                        f"   Python: recall {ref['recall']:.3f} | precision {ref['precision']:.3f} | "
                        f"тип {ref['type']:.3f} | сторона {ref['team']:.3f} | "
                        f"ошибка позиции {ref['pos']:.2f}px | "
                        f"TP {ref['tp']} FP {ref['fp']} FN {ref['fn']}"
                    )
            return 0

        # Режим real: кадры и разметка заданы пользователем. Если не заданы —
        # берём готовый набор из репозитория (assets/eval_real) и сопоставление
        # типов с эталонами из config/live.yaml.
        if not args.frames:
            args.frames = str(ROOT / "assets" / "eval_real" / "frames")
        if not args.truth:
            args.truth = str(ROOT / "assets" / "eval_real" / "truth.tsv")
        if not args.map or not args.types:
            import yaml

            live = yaml.safe_load((ROOT / "config" / "live.yaml").read_text(encoding="utf-8"))
            mapping = live["vision"]["template_map"]
            if not args.map:
                args.map = ",".join(f"{kind}={file}" for kind, file in mapping.items())
            if not args.types:
                args.types = ",".join(mapping.keys())
        if not Path(args.frames).exists() or not Path(args.truth).exists():
            print("в режиме real нужны --frames и --truth (или готовый набор assets/eval_real)",
                  file=sys.stderr)
            return 2
        cmd = [
            "--frames", args.frames,
            "--truth", args.truth,
            "--templates", args.templates,
            "--threshold", f"{(args.threshold if args.threshold is not None else 0.60):.2f}",
            "--nms", f"{args.nms:.2f}",
            "--scale", f"{args.scale:.3f}",
            "--radius", f"{args.radius:.1f}",
            "--max-per-template", str(args.max_per_template),
        ]
        if args.map:
            cmd += ["--map", args.map]
        if args.types:
            cmd += ["--types", args.types]
        if args.arena:
            cmd += ["--arena", args.arena]
        if args.board:
            cmd += ["--board", args.board]
        if args.all_templates:
            cmd += ["--all-templates"]
        if args.team_by_position:
            cmd += ["--team-by-position", "--split", f"{args.split:.3f}"]
        if args.print_detections:
            cmd += ["--print-detections"]
        if args.by_base:
            cmd += ["--by-base"]
        if args.base_radius:
            cmd += ["--base-radius", args.base_radius]
        if args.base_min_score is not None:
            cmd += ["--base-min-score", f"{args.base_min_score:.3f}"]
        if args.base_gradient is not None:
            cmd += ["--base-gradient", f"{args.base_gradient:.1f}"]
        if args.classify_radius is not None:
            cmd += ["--classify-radius", str(args.classify_radius)]
        output = run_tool(cmd)
        print(output.rstrip())
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
