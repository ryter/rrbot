#!/usr/bin/env python3
"""Сравнение метрик эффективности по мане в симуляторе.

Планировщик оценивает юнита по нескольким слагаемым, и одно из них —
«эффективность по мане»: сколько юнит даёт на единицу потраченной маны.
Метрики две:

* ``hp`` — живучесть на единицу цены (``hp / цена``);
* ``hp_dps`` — живучесть × урон на единицу цены, нормированные так же,
  как ``hp`` (значения сопоставимы, вес имеет один смысл).

Какая метрика лучше — вопрос измеримый, а не вкусовой: обе гоняются на
одинаковых сидах в симуляторе, где волны врага у всех одинаковы, поэтому
сравнение парное.

Запуск:

    .venv/bin/python tools/compare_cost_metric.py --seeds 20
    .venv/bin/python tools/compare_cost_metric.py --seeds 40 --presets balanced,cost_focused
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from rrbot.config import load_config  # noqa: E402

import experiment  # noqa: E402

METRICS = ("hp", "hp_dps")


def run(config, metric: str, preset: str, seeds: int, overrides: dict | None = None) -> dict:
    """Прогнать пресет с заданной метрикой по нескольким сидам."""
    local = copy.deepcopy(config)
    local.planner.cost_metric = metric
    for key, value in experiment.PRESETS.get(preset, {}).items():
        setattr(local.planner, key, value)
    for key, value in (overrides or {}).items():
        setattr(local.planner, key, value)
    strategy = preset if preset in experiment.BASELINES else "planner"
    results = [experiment.play_match(local, seed, strategy) for seed in range(seeds)]
    wins = sum(1 for r in results if r.won)
    kinds: dict[str, int] = {}
    for r in results:
        for kind, count in r.spawn_kinds.items():
            kinds[kind] = kinds.get(kind, 0) + count
    n = max(1, len(results))
    return {
        "metric": metric,
        "preset": preset,
        "matches": len(results),
        "wins": wins,
        "win_rate": wins / n,
        "avg_spawns": sum(r.spawns for r in results) / n,
        "avg_waits": sum(r.waits for r in results) / n,
        "avg_time": sum(r.time for r in results) / n,
        "avg_spent": sum(r.spent_mana for r in results) / n,
        "kinds": kinds,
    }


def format_table(rows: list[dict], kinds: list[str]) -> str:
    header = (
        f"{'метрика':<8} {'пресет':<18} {'перебор':<28} {'партий':>6} {'побед':>6} {'доля':>7} "
        f"{'призывов':>9} {'ждали':>7} {'мана':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['metric']:<8} {row['preset']:<18} {row.get('sweep', ''):<28} {row['matches']:>6} "
            f"{row['wins']:>6} {row['win_rate']:>6.1%} {row['avg_spawns']:>9.1f} "
            f"{row['avg_waits']:>7.1f} {row['avg_spent']:>7.0f}"
        )
    lines.append("")
    lines.append("| метрика | пресет | " + " | ".join(kinds) + " |")
    lines.append("|---" * (len(kinds) + 2) + "|")
    for row in rows:
        cells = [str(row["kinds"].get(k, 0)) for k in kinds]
        lines.append(f"| {row['metric']} | {row['preset']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение метрик эффективности по мане")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--seeds", type=int, default=20, help="партий на пресет")
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--presets", default=",".join(experiment.PRESETS.keys()))
    parser.add_argument("--wave-size", type=int, default=0, help="размер волны (0 — из конфига)")
    parser.add_argument("--base-hp", type=int, default=0, help="здоровье баз (0 — из конфига)")
    parser.add_argument("--wave-interval", type=float, default=0.0, help="интервал волн")
    parser.add_argument("--out", default="", help="куда сохранить сводку (пусто — не сохранять)")
    parser.add_argument("--sweep", default="",
                        help="перебор значения веса: имя=значение1,значение2 (например "
                             "save_mana_min_gain=0,0.5,1.5)")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.wave_size:
        config.sim.wave_size = args.wave_size
    if args.base_hp:
        config.sim.base_hp = float(args.base_hp)
    if args.wave_interval:
        config.sim.wave_interval = args.wave_interval

    # Перебор значения одного веса: «имя=1,2,3».
    sweep_key = ""
    sweep_values: list[float] = []
    if args.sweep:
        name, _, values = args.sweep.partition("=")
        sweep_key = name.strip()
        sweep_values = [float(v) for v in values.split(",") if v.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    rows: list[dict] = []
    for metric in metrics:
        for preset in presets:
            # Либо перебираем значение одного веса, либо гоняем как есть.
            variants = sweep_values if sweep_values else [None]
            for value in variants:
                overrides = {sweep_key: value} if value is not None else None
                row = run(config, metric, preset, args.seeds, overrides)
                row["sweep"] = f"{sweep_key}={value}" if value is not None else ""
                rows.append(row)
                print(
                    f"{metric:<8} {preset:<18} {row['sweep']:<28} побед {row['wins']}/{row['matches']} "
                    f"({row['win_rate']:.1%}), призывов {row['avg_spawns']:.1f}",
                    flush=True,
                )

    kinds = sorted({kind for row in rows for kind in row["kinds"]})
    print()
    print(format_table(rows, kinds))

    # Итог: средняя доля побед по пресетам для каждой метрики и значения
    # перебираемого веса.
    print()
    summary: dict[str, float] = {}
    for metric in metrics:
        subset = [row for row in rows if row["metric"] == metric]
        if not subset:
            continue
        total = sum(row["win_rate"] for row in subset) / len(subset)
        summary[metric] = total
        print(f"{metric}: средняя доля побед по пресетам {total:.1%}")
    if sweep_values:
        print()
        for value in sweep_values:
            subset = [row for row in rows if row["sweep"] == f"{sweep_key}={value}"]
            if not subset:
                continue
            total = sum(row["win_rate"] for row in subset) / len(subset)
            print(f"{sweep_key}={value}: средняя доля побед {total:.1%}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Сравнение метрик эффективности по мане",
            "",
            f"- конфиг: `{args.config}`",
            f"- партий на пресет: {args.seeds}",
            f"- размер волны: {args.wave_size or 'из конфига'}",
            f"- интервал волн: {args.wave_interval or 'из конфига'}",
            "",
            format_table(rows, kinds),
            "",
        ]
        for metric, total in summary.items():
            lines.append(f"- {metric}: средняя доля побед по пресетам {total:.1%}")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"сводка сохранена: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
