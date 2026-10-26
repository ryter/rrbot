#!/usr/bin/env python3
"""Сравнение правил выбора пары для слияния на одних и тех же кадрах.

Живые бои дают шумные числа: доля успешных слияний зависит от сложности
боя, от того, сколько пар бот вообще пробовал, и от того, в каком порядке
шли бои. Здесь сравнение делается **офлайн на одних и тех же кадрах**: для
каждого кадра строится список пар-кандидатов (`cpp/build/merge_probe --csv`),
и для каждого правила считается, какую пару оно бы выбрало и какова оценка
успеха этой пары.

Сравниваются правила:
  * `круг`   — прежнее: «предпочитать пары с круглым основанием» (на деле —
               «одинаковый тип карты»), порог похожести 0.78;
  * `любая`  — без правила: первая пара с похожестью >= 0.78;
  * `оценка` — новое: пара с максимальной оценкой успеха
               (`rr::merge_success_estimate`).

Запуск:
    .venv/bin/python tools/merge_policy_offline.py /tmp/opencode/pvp
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "cpp" / "build" / "merge_probe"

# Пороги бота (см. cpp/src/merge.hpp).
TRY_MIN = 0.78
TRY_STRICT = 0.90
TRY_FLOOR = 0.65
SIZE_MAX = 0.12
SIZE_RELAXED = 0.16


class Pair:
    __slots__ = ("score", "size_diff", "rank_src", "rank_dst", "same_rank",
                 "adjacent", "estimate")

    def __init__(self, fields: list[str]) -> None:
        self.score = float(fields[6])
        self.size_diff = float(fields[8])
        self.rank_src = int(fields[9])
        self.rank_dst = int(fields[10])
        self.same_rank = fields[11] == "1"
        self.adjacent = fields[12] == "1"
        self.estimate = float(fields[13])


class Frame:
    def __init__(self, our_pawns: int, pairs: list[Pair]) -> None:
        self.our_pawns = our_pawns
        self.pairs = pairs


def probe(frame: Path) -> Frame | None:
    result = subprocess.run(
        [str(PROBE), str(frame), "--csv"], capture_output=True, text=True, check=False,
    )
    our_pawns = 0
    pairs: list[Pair] = []
    for line in result.stdout.splitlines():
        fields = line.split(";")
        if fields[0] == "our_pawns" and len(fields) == 2:
            our_pawns = int(fields[1])
            continue
        if len(fields) != 14:
            continue
        pairs.append(Pair(fields))
    return Frame(our_pawns, pairs)


def policy_old(pairs: list[Pair]) -> Pair | None:
    """Прежнее правило: сперва пары с круглым основанием (ранг 1)."""
    def ok(pair: Pair) -> bool:
        return (pair.score >= TRY_MIN and pair.size_diff <= SIZE_MAX and pair.adjacent)

    for pair in pairs:
        if ok(pair) and pair.rank_src == 1 and pair.rank_dst == 1:
            return pair
    for pair in pairs:
        if ok(pair):
            return pair
    return None


def policy_any(pairs: list[Pair]) -> Pair | None:
    """Без правила: первая пара с похожестью >= 0.78."""
    for pair in pairs:
        if pair.score >= TRY_MIN and pair.size_diff <= SIZE_MAX and pair.adjacent:
            return pair
    return None


def policy_estimate(frame: Frame) -> Pair | None:
    """Новое правило: пара с максимальной оценкой успеха.

    Лестница порогов — ровно как в боте (cpp/src/main.cpp): сначала очень
    похожие пары, потом порог 0.78, потом 0.72; последние два ступени
    работают, только когда доска забита (своих пешек >= 10).
    """
    def best(min_score: float, max_size: float, adjacent: bool) -> Pair | None:
        found = None
        for pair in frame.pairs:
            if pair.score < min_score or pair.size_diff > max_size:
                continue
            if adjacent and not pair.adjacent:
                continue
            if found is None or pair.estimate > found.estimate:
                found = pair
        return found

    crowded = frame.our_pawns >= 10
    steps = [
        (TRY_STRICT, SIZE_MAX, True),
        (TRY_MIN, SIZE_MAX, True),
        (TRY_MIN - 0.06, SIZE_MAX, True),
    ]
    if crowded:
        steps.append((TRY_FLOOR, SIZE_RELAXED, True))
        steps.append((TRY_FLOOR, SIZE_RELAXED, False))
    for min_score, max_size, adjacent in steps:
        found = best(min_score, max_size, adjacent)
        if found is not None:
            return found
    return None


POLICIES = {
    "круг (старое)": policy_old,
    "любая (без правила)": policy_any,
    "оценка (новое)": policy_estimate,
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    directory = Path(sys.argv[1])
    frames = sorted(directory.glob("*.png"))
    if not frames:
        print(f"в {directory} нет кадров")
        return 1
    if not PROBE.exists():
        print("не собран cpp/build/merge_probe (собери через make)")
        return 1

    totals = {name: [0, 0.0] for name in POLICIES}  # [кадров с парой, сумма оценок]
    frames_with_pairs = 0
    for path in frames:
        frame = probe(path)
        if frame is None or not frame.pairs:
            continue
        frames_with_pairs += 1
        for name, policy in POLICIES.items():
            chosen = policy(frame) if name.startswith("оценка") else policy(frame.pairs)
            if chosen is None:
                continue
            totals[name][0] += 1
            totals[name][1] += chosen.estimate

    print(f"кадров разобрано: {len(frames)}, из них с парами: {frames_with_pairs}\n")
    print(f"{'правило':24s} {'попыток':>8s} {'ожид. успехов':>14s} {'на кадр':>9s} {'оценка':>8s}")
    for name, (count, expected) in totals.items():
        per_frame = expected / frames_with_pairs if frames_with_pairs else 0.0
        quality = expected / count if count else 0.0
        print(f"{name:24s} {count:8d} {expected:14.1f} {per_frame:9.3f} {quality * 100:7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
