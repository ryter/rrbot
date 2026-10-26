#!/usr/bin/env python3
"""Разбор попыток слияния из лога боя: доля успеха по дистанции и похожести.

Лог боя (`/tmp/opencode/battle_loop.log`) содержит строки:

    сливаю: (x1,y1) на (x2,y2), похожесть S, разница размеров D %, ...
    слияние вышло: ...
    слияние не вышло: ...

Скрипт считает долю успеха в разрезе дистанции перетаскивания (камера едет
— дальние слияния чаще промахиваются) и похожести портретов (главный
признак успеха).

Запуск:
    .venv/bin/python tools/merge_success_stats.py /tmp/opencode/battle_loop.log
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

RE_ATTEMPT = re.compile(
    r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\), "
    r"похожесть ([\d.]+), разница размеров (-?\d+) %")
RE_OK = re.compile(r"слияние вышло")
RE_FAIL = re.compile(r"слияние не вышло")


def parse(path: str) -> list[tuple[float, float, int, float]]:
    """Список попыток: (похожесть, разница размеров, успех, дистанция)."""
    rows: list[tuple[float, float, int, float]] = []
    pending: tuple[float, float, float] | None = None
    for line in open(path, encoding="utf-8"):
        match = RE_ATTEMPT.search(line)
        if match:
            x1, y1, x2, y2 = (int(match.group(i)) for i in range(1, 5))
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            pending = (float(match.group(5)), float(match.group(6)), distance)
            continue
        if pending is None:
            continue
        if RE_OK.search(line):
            rows.append((pending[0], pending[1], 1, pending[2]))
            pending = None
        elif RE_FAIL.search(line):
            rows.append((pending[0], pending[1], 0, pending[2]))
            pending = None
    return rows


def bucket(rows: list[tuple[float, float, int, float]], key, step: float) -> dict:
    buckets: dict = defaultdict(lambda: [0, 0])
    for row in rows:
        index = int(row[key] / step)
        buckets[index][0] += row[2]
        buckets[index][1] += 1
    return buckets


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = parse(sys.argv[1])
    if not rows:
        print("в логе нет попыток слияния")
        return 1
    ok = sum(row[2] for row in rows)
    print(f"попыток: {len(rows)}, успех: {ok} ({100 * ok // len(rows)} %)")
    print()
    print("по дистанции перетаскивания:")
    for index in sorted(bucket(rows, 3, 40.0)):
        good, total = bucket(rows, 3, 40.0)[index]
        print(f"  {index * 40:4d}-{index * 40 + 40:4d} px: {good}/{total} "
              f"({100 * good // total} %)")
    print()
    print("по похожести портретов:")
    for index in sorted(bucket(rows, 0, 0.05)):
        good, total = bucket(rows, 0, 0.05)[index]
        print(f"  {index * 0.05:.2f}-{(index + 1) * 0.05:.2f}: {good}/{total} "
              f"({100 * good // total} %)")
    print()
    print("по разнице размеров пешек:")
    for index in sorted(bucket(rows, 1, 4.0)):
        good, total = bucket(rows, 1, 4.0)[index]
        print(f"  {index * 4:2d}-{index * 4 + 4:2d} %: {good}/{total} "
              f"({100 * good // total} %)")
    print()
    failed = [row for row in rows if not row[2]]
    if failed:
        print("неудачные подробно:")
        for score, size_diff, _, distance in failed:
            print(f"  дистанция {distance:.0f} px, похожесть {score:.3f}, "
                  f"разница размеров {size_diff:.0f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
