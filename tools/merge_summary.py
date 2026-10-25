#!/usr/bin/env python3
"""Сводка по логам боёв: попытки, успехи и успехи на один бой.

Доля успешных слияний сама по себе обманчива: чем больше бот пытается,
тем ниже доля (в каждом бою число попыток разное). Поэтому главная цифра —
**сколько успешных слияний приходится на один бой**.

Запуск:
    .venv/bin/python tools/merge_summary.py /tmp/opencode/ab_new.log ...
"""
from __future__ import annotations

import sys

from merge_attempts import parse


def summarize(path: str) -> dict:
    rounds = parse(path)
    pvp = [r for r in rounds if not r.coop]
    coop = [r for r in rounds if r.coop]
    ok = sum(r.ok for r in pvp)
    fail = sum(r.fail for r in pvp)
    attempts = ok + fail
    return {
        "path": path,
        "pvp_rounds": len(pvp),
        "coop_rounds": len(coop),
        "attempts": attempts,
        "ok": ok,
        "fail": fail,
        "rate": (ok / attempts) if attempts else 0.0,
        "per_round": (ok / len(pvp)) if pvp else 0.0,
        "attempts_per_round": (attempts / len(pvp)) if pvp else 0.0,
    }


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    print(f"{'лог':34s} {'боёв PvP':>8s} {'попыт.':>7s} {'успех':>6s} "
          f"{'доля':>7s} {'успех/бой':>10s} {'попыт./бой':>11s}")
    for path in paths:
        s = summarize(path)
        name = path.rsplit("/", 1)[-1]
        print(f"{name:34s} {s['pvp_rounds']:8d} {s['attempts']:7d} {s['ok']:6d} "
              f"{s['rate'] * 100:6.1f}% {s['per_round']:10.2f} "
              f"{s['attempts_per_round']:11.2f}")
    return 0


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
