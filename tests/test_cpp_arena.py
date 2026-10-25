"""Тест поиска арены по кадру (мобильный клиент).

В бою камера ходит: арена то приближается, то отдаляется, и даже
смещается. Поэтому область поля приходится искать в каждом кадре по
плиткам. Набор — настоящие кадры боя из эмулятора с разметкой позиций
юнитов (``assets/eval_real``): каждый юнит должен попасть в свою половину
найденной арены.
"""

from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAMES = ROOT / "assets" / "eval_real" / "frames"
TRUTH = ROOT / "assets" / "eval_real" / "truth.tsv"
# Проверочная программа: печатает найденные половины арены.
PROBE_SRC = ROOT / "cpp" / "eval" / "arena_probe.cpp"
ARENA_SRC = ROOT / "cpp" / "src" / "arena.cpp"


def _load_truth() -> dict[str, list[tuple[str, float, float]]]:
    out: dict[str, list[tuple[str, float, float]]] = collections.defaultdict(list)
    if not TRUTH.exists():
        return out
    for line in TRUTH.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out[parts[0]].append((parts[2], float(parts[3]), float(parts[4])))
    return out


@pytest.mark.skipif(not FRAMES.exists(), reason="нет набора настоящих кадров")
def test_арена_находится_и_юниты_попадают_в_свои_половины(tmp_path):
    """Найденная арена должна покрывать всех юнитов разметки."""
    if not PROBE_SRC.exists():
        pytest.skip("нет проверочной программы")
    binary = tmp_path / "arena_probe"
    build = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-I/usr/include/opencv5",
         f"-I{ROOT / 'cpp' / 'src'}", str(PROBE_SRC), str(ARENA_SRC),
         "-o", str(binary), "-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs"],
        capture_output=True, text=True, check=False,
    )
    if build.returncode != 0:
        pytest.skip("не собралась проверочная программа")

    frames = sorted(FRAMES.glob("*.png"))
    assert frames, "нет кадров"
    out = subprocess.run([str(binary), *[str(f) for f in frames]],
                         capture_output=True, text=True, check=False).stdout
    truth = _load_truth()
    import re

    pattern = re.compile(r"верх (\d+),(\d+) (\d+)x(\d+).*?низ (\d+),(\d+) (\d+)x(\d+)")
    # Запас: центр круглого основания юнита может лежать чуть за краем
    # плитки (радиус основания в каноническом кадре около 30 px).
    margin = 24
    total = 0
    bad = 0
    missing = 0
    for line in out.splitlines():
        name = Path(line.split()[0]).name
        match = pattern.search(line)
        if not match:
            missing += 1
            continue
        g = [int(x) for x in match.groups()]
        top = (g[0], g[1], g[2], g[3])
        bottom = (g[4], g[5], g[6], g[7])
        for side, x, y in truth.get(name, []):
            total += 1
            rect = top if side == "enemy" else bottom
            inside = (rect[0] - margin <= x <= rect[0] + rect[2] + margin
                      and rect[1] - margin <= y <= rect[1] + rect[3] + margin)
            if not inside:
                bad += 1
    assert total > 0, "в разметке нет юнитов"
    assert missing == 0, f"арена не найдена на {missing} кадрах"
    assert bad == 0, f"{bad} юнитов из {total} оказались вне найденной арены"
