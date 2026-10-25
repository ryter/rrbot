"""Тест таблицы оценки вероятности успеха слияния.

Бот выбирает пару для слияния по оценке `rr::merge_success_estimate`
(похожесть портретов + ранг + разница размеров). Оценка собрана по
замерам живых боёв (notes/ЗАДАЧА1_РАНГ.md, tools/merge_pairs_stats.py),
поэтому таблица проверяется на трёх свойствах:

  * оценка растёт с похожестью портретов;
  * пара одного ранга оценивается выше пары разных рангов с той же
    похожестью: игра сливает только юнитов одного ранга, а пара
    «ранг 1 + ранг 2» — отказ (живой опыт 24.10);
  * заметно разные по размеру пешки оцениваются ниже.

Значения таблицы тоже закреплены: если их поменять, замеры в notes
перестанут соответствовать коду.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "cpp" / "build" / "merge_estimate"


def estimate(score: float, same_rank: bool, size_diff: float = 0.0) -> float:
    result = subprocess.run(
        [str(TOOL), f"{score}", "1" if same_rank else "0", f"{size_diff}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip())


@pytest.fixture(scope="module", autouse=True)
def _tool_built():
    if not TOOL.exists():
        pytest.skip("не собран cpp/build/merge_estimate (собери через make)")


def test_table_values():
    # Значения из замеров: пара одного ранга с похожими портретами сливается
    # в 92 % случаев; пара разных рангов игра не принимает (живой опыт
    # 24.10), поэтому её оценка близка к нулю.
    assert estimate(0.95, True) == pytest.approx(0.92, abs=0.001)
    assert estimate(0.85, True) == pytest.approx(0.77, abs=0.001)
    assert estimate(0.79, True) == pytest.approx(0.50, abs=0.001)
    assert estimate(0.70, True) == pytest.approx(0.33, abs=0.001)
    assert estimate(0.95, False) == pytest.approx(0.05, abs=0.001)
    assert estimate(0.70, False) == pytest.approx(0.05, abs=0.001)


def test_grows_with_similarity():
    scores = [0.70, 0.78, 0.82, 0.86, 0.90, 0.95]
    values = [estimate(score, True) for score in scores]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_same_rank_wins():
    for score in (0.70, 0.80, 0.90, 0.99):
        assert estimate(score, True) > estimate(score, False)


def test_size_difference_lowers_estimate():
    # Пешки разного размера — хуже (замер: 42 % успеха против 68 %).
    assert estimate(0.95, True, 0.20) < estimate(0.95, True, 0.02)
