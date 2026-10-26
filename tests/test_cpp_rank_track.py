"""Трекер ранга: призыв = 1, успешное слияние поднимает цель на +1."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "cpp" / "build" / "rank_track"


@pytest.fixture(scope="module", autouse=True)
def _tool_built():
    if not TOOL.exists():
        pytest.skip("не собран cpp/build/rank_track (собери через make)")


def test_rank_track_selfcheck():
    result = subprocess.run([str(TOOL)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
