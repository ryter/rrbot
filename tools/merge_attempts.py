#!/usr/bin/env python3
"""Разбор лога боя: сколько слияний вышло, а сколько нет.

Логи бота (cpp/build/rrbot) печатают на каждое действие:
    сливаю: (x1,y1) на (x2,y2), похожесть S, разница размеров D %, сдвиг (...)
    слияние вышло: ...
    слияние не вышло: ...

Скрипт режет лог по раундам, считает долю успешных слияний и собирает
распределения по похожести и разнице размеров. Раунды в Co-Op помечаются:
там пол не зелёный и слияния ведут себя иначе, их в статистику не берём.

Запуск:
    .venv/bin/python tools/merge_attempts.py /tmp/opencode/before.log ...
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field


@dataclass
class Round:
    index: int = 0
    coop: bool = False
    attempts: int = 0
    ok: int = 0
    fail: int = 0
    scores_ok: list[float] = field(default_factory=list)
    scores_fail: list[float] = field(default_factory=list)
    sizes_ok: list[float] = field(default_factory=list)
    sizes_fail: list[float] = field(default_factory=list)


RE_ROUND = re.compile(r"^== раунд (\d+)/(\d+) ==")
RE_ATTEMPT = re.compile(
    r"сливаю: \((\d+),(\d+)\) на \((\d+),(\d+)\), "
    r"похожесть ([\d.]+), разница размеров (-?\d+) %"
)
RE_COOP = re.compile(r"переключаюсь на Co-Op")


def parse(path: str) -> list[Round]:
    rounds: list[Round] = []
    cur: Round | None = None
    pending: tuple[float, float] | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = RE_ROUND.match(line)
            if m:
                cur = Round(index=int(m.group(1)))
                rounds.append(cur)
                continue
            if cur is None:
                continue
            if RE_COOP.search(line):
                cur.coop = True
                continue
            m = RE_ATTEMPT.search(line)
            if m:
                cur.attempts += 1
                pending = (float(m.group(5)), float(m.group(6)) / 100.0)
                continue
            if "слияние вышло" in line:
                cur.ok += 1
                if pending:
                    cur.scores_ok.append(pending[0])
                    cur.sizes_ok.append(pending[1])
                pending = None
                continue
            if "слияние не вышло" in line:
                cur.fail += 1
                if pending:
                    cur.scores_fail.append(pending[0])
                    cur.sizes_fail.append(pending[1])
                pending = None
                continue
    return rounds


def summarize(path: str) -> dict:
    rounds = parse(path)
    pvp = [r for r in rounds if not r.coop and (r.ok + r.fail) > 0]
    coop = [r for r in rounds if r.coop and (r.ok + r.fail) > 0]
    ok = sum(r.ok for r in pvp)
    fail = sum(r.fail for r in pvp)
    att = ok + fail
    return {
        "path": path,
        "rounds": rounds,
        "pvp_rounds": len(pvp),
        "coop_rounds": len(coop),
        "attempts": att,
        "ok": ok,
        "fail": fail,
        "rate": (ok / att) if att else 0.0,
        "scores_ok": [v for r in pvp for v in r.scores_ok],
        "scores_fail": [v for r in pvp for v in r.scores_fail],
        "sizes_ok": [v for r in pvp for v in r.sizes_ok],
        "sizes_fail": [v for r in pvp for v in r.sizes_fail],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        s = summarize(path)
        print(f"=== {path} ===")
        for r in s["rounds"]:
            mark = "Co-Op" if r.coop else "PvP  "
            tot = r.ok + r.fail
            rate = f"{r.ok / tot * 100:.0f}%" if tot else "—"
            print(f"  раунд {r.index} ({mark}): попыток {tot:3d}, "
                  f"успех {r.ok:3d}, провал {r.fail:3d}, доля {rate}")
        print(f"  итого PvP: попыток {s['attempts']}, успех {s['ok']}, "
              f"провал {s['fail']}, доля {s['rate'] * 100:.1f}%")
        if s["coop_rounds"]:
            print(f"  (раундов в Co-Op: {s['coop_rounds']} — в статистику не входят)")
        print(f"  похожесть: успешных в среднем {mean(s['scores_ok']):.3f}, "
              f"неудачных {mean(s['scores_fail']):.3f}")
        print(f"  разница размеров: успешных {mean(s['sizes_ok']) * 100:.1f}%, "
              f"неудачных {mean(s['sizes_fail']) * 100:.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
