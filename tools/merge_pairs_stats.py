#!/usr/bin/env python3
"""Какие пары сливаются, а какие нет: разбор лога по форме оснований.

Каждая попытка слияния в логе бота выглядит так:
    сливаю: (x1,y1) на (x2,y2), похожесть S, разница размеров D %,
             основания F1/F2 %, сдвиг (...)
После неё — «слияние вышло» или «слияние не вышло».

Форма основания кодирует ТИП карты (круг / овал / ромб / щит), а не ранг.
Пары разных типов игра не сливает никогда, пары одного типа — сливаются,
если ранги совпадают. Скрипт проверяет, как доля успеха зависит от:

  * одинаковой ли формы основания (по заполненности: круг ~0.9, овал ~0.8,
    ромб ~0.65);
  * похожести портретов;
  * разницы размеров пешек.

Запуск:
    .venv/bin/python tools/merge_pairs_stats.py /tmp/opencode/diag1.log
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass


@dataclass
class Attempt:
    score: float
    size_diff: float
    fill_src: float
    fill_dst: float
    shape_src: str = ""
    shape_dst: str = ""
    estimate: float | None = None
    ok: bool | None = None


RE_ATTEMPT = re.compile(
    r"сливаю: \(\d+,\d+\) на \(\d+,\d+\), "
    r"похожесть ([\d.]+), разница размеров (-?\d+) %, "
    r"основания (\d+)/(\d+) %(?: \(([^/]+)/([^)]+)\))?"
    r"(?:, оценка успеха (\d+) %)?"
)


def parse(path: str) -> list[Attempt]:
    attempts: list[Attempt] = []
    pending: Attempt | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_ATTEMPT.search(line)
            if m:
                pending = Attempt(
                    score=float(m.group(1)),
                    size_diff=float(m.group(2)) / 100.0,
                    fill_src=float(m.group(3)) / 100.0,
                    fill_dst=float(m.group(4)) / 100.0,
                    shape_src=(m.group(5) or "").strip(),
                    shape_dst=(m.group(6) or "").strip(),
                    estimate=(float(m.group(7)) / 100.0) if m.group(7) else None,
                )
                attempts.append(pending)
                continue
            if pending is not None and "слияние вышло" in line:
                pending.ok = True
                pending = None
            elif pending is not None and "слияние не вышло" in line:
                pending.ok = False
                pending = None
    return attempts


def bucket_fill(value: float) -> str:
    """Форма основания по заполненности: круг / овал / ромб."""
    if value >= 0.88:
        return "круг"
    if value >= 0.72:
        return "овал"
    return "ромб"


def report(attempts: list[Attempt]) -> None:
    done = [a for a in attempts if a.ok is not None]
    if not done:
        print("нет завершённых попыток")
        return
    ok = sum(1 for a in done if a.ok)
    print(f"попыток {len(done)}, успех {ok} ({ok / len(done) * 100:.1f}%)")

    print("\n-- по форме оснований (тип карты) --")
    groups: dict[tuple[str, str], list[bool]] = {}
    for a in done:
        key = (bucket_fill(a.fill_src), bucket_fill(a.fill_dst))
        groups.setdefault(key, []).append(a.ok)
    for key in sorted(groups):
        res = groups[key]
        good = sum(res)
        label = "одинаковые" if key[0] == key[1] else "разные  "
        print(f"  {label} основания {key[0]}+{key[1]}: {good}/{len(res)} "
              f"({good / len(res) * 100:.0f}%)")

    print("\n-- по похожести портретов --")
    for lo, hi in [(0.65, 0.78), (0.78, 0.86), (0.86, 0.92), (0.92, 1.01)]:
        res = [a.ok for a in done if lo <= a.score < hi]
        if res:
            good = sum(res)
            print(f"  похожесть {lo:.2f}..{hi:.2f}: {good}/{len(res)} "
                  f"({good / len(res) * 100:.0f}%)")

    print("\n-- по разнице размеров --")
    for lo, hi in [(0.0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 0.26)]:
        res = [a.ok for a in done if lo <= a.size_diff < hi]
        if res:
            good = sum(res)
            print(f"  разница {lo * 100:.0f}..{hi * 100:.0f}%: {good}/{len(res)} "
                  f"({good / len(res) * 100:.0f}%)")

    print("\n-- по разнице размеров внутри пар одинакового типа --")
    same = [a for a in done if bucket_fill(a.fill_src) == bucket_fill(a.fill_dst)]
    for lo, hi in [(0.0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 0.26)]:
        res = [a.ok for a in same if lo <= a.size_diff < hi]
        if res:
            good = sum(res)
            print(f"  разница {lo * 100:.0f}..{hi * 100:.0f}%: {good}/{len(res)} "
                  f"({good / len(res) * 100:.0f}%)")


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        print(f"=== {path} ===")
        report(parse(path))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
