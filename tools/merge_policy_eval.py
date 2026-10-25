#!/usr/bin/env python3
"""Оценка правил выбора пары для слияния по данным живых боёв.

Скрипт читает логи бота (где на каждую попытку печатаются похожесть,
разница размеров, формы оснований и результат) и оценивает правила вида
«сливать только такие-то пары».

Показывается:
  * доля успеха среди попыток, удовлетворяющих правилу (качество);
  * охват — какая доля всех попыток попадает под правило (сколько пар
    бот вообще сможет слить: если охват мал, бот будет стоять);
  * ожидаемое число успешных слияний на 100 попыток (качество × охват).

Запуск:
    .venv/bin/python tools/merge_policy_eval.py /tmp/opencode/diag1.log ...
"""
from __future__ import annotations

import sys

from merge_pairs_stats import bucket_fill, parse


def evaluate(attempts, name: str, predicate) -> tuple[int, int, float]:
    chosen = [a for a in attempts if predicate(a)]
    if not chosen:
        return 0, 0, 0.0
    ok = sum(1 for a in chosen if a.ok)
    return len(chosen), ok, ok / len(chosen)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    attempts = []
    for path in paths:
        attempts.extend(parse(path))
    done = [a for a in attempts if a.ok is not None]
    if not done:
        print("нет завершённых попыток")
        return 1
    total = len(done)
    base_ok = sum(1 for a in done if a.ok)
    print(f"попыток {total}, успех {base_ok} ({base_ok / total * 100:.1f} %)")

    same = lambda a: bucket_fill(a.fill_src) == bucket_fill(a.fill_dst)  # noqa: E731

    rules = []
    for threshold in (0.70, 0.78, 0.82, 0.86, 0.90):
        rules.append((f"похожесть >= {threshold:.2f}",
                      lambda a, t=threshold: a.score >= t))
        rules.append((f"похожесть >= {threshold:.2f} и один тип",
                      lambda a, t=threshold: a.score >= t and same(a)))
    rules.append(("только один тип карты", same))
    rules.append(("только один тип и разница размеров <= 12%",
                  lambda a: same(a) and a.size_diff <= 0.12))

    print(f"\n{'правило':46s} {'попыток':>8s} {'успех':>12s} {'доля':>7s} {'ожид. на 100':>12s}")
    for name, predicate in rules:
        count, ok, rate = evaluate(done, name, predicate)
        if not count:
            print(f"{name:46s} {'—':>8s}")
            continue
        expected = rate * 100
        print(f"{name:46s} {count:8d} {ok:6d}/{count:<5d} {rate * 100:6.1f}% "
              f"{expected:12.1f}")
    return 0


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
