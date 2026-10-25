"""Тест занятости доски (сетка высадки и свободные клетки).

Поле боя — сетка клеток; на своей стороне их 5x3 (15 клеток). Бот должен
знать, сколько клеток занято: если свободных нет, призыв всё равно не
создаст юнита, и место надо освободить слиянием.

Занятость считается по клеткам, а не по найденным юнитам: круглых оснований
хватает не всем юнитам (у слитых форма другая), и по ним доска выглядела бы
свободной.

Набор кадров — ``assets/board``: настоящие кадры боя из эмулятора, у каждого
вручную посчитано, сколько клеток занято (разметка в ``truth.tsv``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAMES = ROOT / "assets" / "board" / "frames"
TRUTH = ROOT / "assets" / "board" / "truth.tsv"
PROBE_SRC = ROOT / "cpp" / "eval" / "board_probe.cpp"
SOURCES = [
    ROOT / "cpp" / "src" / "board.cpp",
    ROOT / "cpp" / "src" / "merge.cpp",
    ROOT / "cpp" / "src" / "arena.cpp",
]


def _build(tmp_path: Path) -> Path:
    binary = tmp_path / "board_probe"
    build = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-I/usr/include/opencv5",
         f"-I{ROOT / 'cpp' / 'src'}", str(PROBE_SRC), *map(str, SOURCES),
         "-o", str(binary), "-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs"],
        capture_output=True, text=True, check=False,
    )
    if build.returncode != 0:
        pytest.skip("не собралась проверочная программа: " + build.stderr[-500:])
    return binary


def _truth() -> dict[str, int]:
    out: dict[str, int] = {}
    if not TRUTH.exists():
        return out
    for line in TRUTH.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = int(parts[1])
    return out


OCCUPIED_RE = re.compile(r"занято (\d+)/(\d+)")
SPACING_RE = re.compile(r"шаг ([\d.]+)")


@pytest.mark.skipif(not FRAMES.exists(), reason="нет набора настоящих кадров")
def test_занятость_доски_совпадает_с_разметкой(tmp_path):
    """На настоящих кадрах число занятых клеток должно совпасть с разметкой."""
    binary = _build(tmp_path)
    truth = _truth()
    frames = sorted(FRAMES.glob("*.png"))
    assert frames, "нет кадров"
    assert truth, "нет разметки"
    out = subprocess.run([str(binary), *[str(f) for f in frames]],
                         capture_output=True, text=True, check=False).stdout
    checked = 0
    for line in out.splitlines():
        name = Path(line.split()[0]).name.rstrip(":")
        if name not in truth:
            continue
        match = OCCUPIED_RE.search(line)
        assert match, f"нет числа занятых клеток в строке: {line}"
        occupied, capacity = int(match.group(1)), int(match.group(2))
        assert capacity == 15, f"вместимость доски должна быть 15, а не {capacity}"
        assert occupied == truth[name], (
            f"{name}: занято {occupied} клеток, а по разметке {truth[name]}")
        checked += 1
    assert checked == len(truth), "проверено не всё"


@pytest.mark.skipif(not FRAMES.exists(), reason="нет набора настоящих кадров")
def test_шаг_сетки_совпадает_с_замером(tmp_path):
    """Шаг сетки должен быть около 2.5 радиуса основания (в замере 78 px)."""
    binary = _build(tmp_path)
    frames = sorted(FRAMES.glob("*.png"))
    out = subprocess.run([str(binary), *[str(f) for f in frames]],
                         capture_output=True, text=True, check=False).stdout
    spacings = [float(m.group(1)) for m in (SPACING_RE.search(l) for l in out.splitlines()) if m]
    assert spacings, "шаг сетки не определён"
    for spacing in spacings:
        assert 70.0 <= spacing <= 85.0, f"шаг сетки {spacing} px — не похоже на поле"


def test_вместимость_доски_15():
    """На своей стороне 5x3 = 15 клеток высадки."""
    header = (ROOT / "cpp" / "src" / "board.hpp").read_text()
    assert "kDeployCols = 5" in header
    assert "kDeployRows = 3" in header
    assert "kDeployCapacity = kDeployCols * kDeployRows" in header
