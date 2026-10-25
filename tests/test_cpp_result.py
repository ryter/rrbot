"""Тест определения исхода боя (победа/поражение).

На экране итогов посреди экрана лента: у победы синяя («ПОБЕДА»), у
поражения красная. Бот считает исходы боёв — без этого нельзя понять, играет
он лучше или хуже.

Проверяем на синтетических кадрах: лента из эталона
`assets/ui/result_victory.png` на пустом фоне — победа; пустой кадр и кадр
с синим фоном (главное меню) — неопознанный исход.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "cpp" / "build" / "rrbot"
RIBBON = ROOT / "assets" / "ui" / "result_victory.png"

import cv2  # noqa: E402  (после ROOT, чтобы было понятно, что это зависимость)


def result_of(frame: Path) -> str:
    proc = subprocess.run(
        [str(TOOL), "--backend", "adb", "--frame", str(frame), "--probe"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in proc.stdout.splitlines():
        if line.startswith("исход боя:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("в выводе нет строки про исход боя:\n" + proc.stdout)


def test_victory_banner(tmp_path):
    frame = np.zeros((1030, 941, 3), np.uint8)
    ribbon = cv2.imread(str(RIBBON))
    assert ribbon is not None, "нет эталона ленты"
    frame[476:476 + ribbon.shape[0], 235:235 + ribbon.shape[1]] = ribbon
    path = tmp_path / "victory.png"
    cv2.imwrite(str(path), frame)
    assert result_of(path) == "победа"


def test_empty_frame_is_unknown(tmp_path):
    frame = np.zeros((1030, 941, 3), np.uint8)
    path = tmp_path / "empty.png"
    cv2.imwrite(str(path), frame)
    assert result_of(path) == "непонятно"


def test_blue_background_is_not_victory(tmp_path):
    # Главное меню: синий фон. По одному цвету его не отличить от ленты
    # победы — поэтому проверяем, что шаблон действительно нужен.
    frame = np.zeros((1030, 941, 3), np.uint8)
    frame[:, :] = (200, 120, 30)
    path = tmp_path / "menu.png"
    cv2.imwrite(str(path), frame)
    assert result_of(path) == "непонятно"
