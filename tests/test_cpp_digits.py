"""Тесты чтения цифр C++-ботом по записанным вырезам энергии.

Набор вырезов лежит в ``assets/test_energy`` (картинка + ожидаемое
значение). Эталоны цифр эмулятора однажды оказались обрезаны неверно:
«6» не совпадала сама с собой, и число энергии читалось как «?0».
Тест ловит такое: он гоняет бот по всем вырезам и требует, чтобы все
числа прочитались верно.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "cpp" / "build" / "rrbot"
FIXTURES = ROOT / "assets" / "test_energy"
TEXT_RE = re.compile(r'цифры: "([0-9?]*)"')


def _labels() -> list[tuple[str, str]]:
    path = FIXTURES / "labels.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.split()
        out.append((name, value))
    return out


@pytest.mark.skipif(not BOT.exists(), reason="C++-бот не собран")
@pytest.mark.skipif(not _labels(), reason="нет набора вырезов энергии")
def test_энергия_читается_верно():
    wrong = []
    for name, expected in _labels():
        out = subprocess.run(
            [str(BOT), "--backend", "adb", "--digits-image", str(FIXTURES / name)],
            capture_output=True, text=True, check=False,
        ).stdout
        match = TEXT_RE.search(out)
        got = match.group(1) if match else ""
        if got != expected:
            wrong.append(f"{name}: ожидалось {expected}, получено {got or '—'}")
    assert not wrong, "число прочитано неверно:\n" + "\n".join(wrong)


@pytest.mark.skipif(not BOT.exists(), reason="C++-бот не собран")
def test_эталоны_цифр_есть_для_всех_цифр():
    # Без эталона цифры число не прочитается никогда.
    directory = ROOT / "assets" / "digits_emulator"
    digits = {p.stem.split("_")[0] for p in directory.glob("*.png")}
    assert digits == set("0123456789"), f"нет эталонов для цифр: {set('0123456789') - digits}"


@pytest.mark.skipif(not BOT.exists(), reason="C++-бот не собран")
@pytest.mark.skipif(not _labels(), reason="нет набора вырезов энергии")
def test_гаснущее_число_читается(tmp_path):
    """На потолке энергии цифры гаснут и сереют — число всё равно нужно.

    Проверяем на затемнённом вырезе: при яркости вдвое меньше обычный
    порог по яркости уже не находит цифр, но запасной проход читает.
    """
    import cv2
    import numpy as np

    name, expected = _labels()[0]
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"вырез не читается: {name}"
    dark = (image.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    path = tmp_path / "dark.png"
    cv2.imwrite(str(path), dark)
    out = subprocess.run(
        [str(BOT), "--backend", "adb", "--digits-image", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout
    match = TEXT_RE.search(out)
    got = match.group(1) if match else ""
    assert got == expected, f"гаснущее число прочитано как {got or '—'}, ожидалось {expected}"


@pytest.mark.skipif(not BOT.exists(), reason="C++-бот не собран")
@pytest.mark.skipif(not _labels(), reason="нет набора вырезов энергии")
def test_чтение_переживает_смену_разрешения(tmp_path):
    """Цифры должны читаться и при другом разрешении экрана.

    Размер окна Waydroid меняет разрешение Android, и бот приводит кадр к
    каноническому размеру холста. Если при увеличении использовать
    INTER_AREA (он работает почти как nearest), цифры перестают узнаваться:
    на наборе из 70 вырезов читается 18/70 вместо 40/70 у INTER_CUBIC.

    Здесь имитируем смену разрешения: вырез уменьшаем вдвое (как кадр с
    меньшего экрана) и увеличиваем обратно — так же, как это делает бот.
    """
    import cv2

    labels = _labels()[::6]  # вырезки по всему набору, а не подряд
    correct = 0
    for name, expected in labels:
        image = cv2.imread(str(FIXTURES / name), cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"вырез не читается: {name}"
        height, width = image.shape[:2]
        small = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        path = tmp_path / name
        cv2.imwrite(str(path), back)
        out = subprocess.run(
            [str(BOT), "--backend", "adb", "--digits-image", str(path)],
            capture_output=True, text=True, check=False,
        ).stdout
        match = TEXT_RE.search(out)
        if match and match.group(1) == expected:
            correct += 1
    # При INTER_AREA на этом наборе читается около четверти вырезов,
    # при INTER_CUBIC — больше половины. Требуем заметно лучше четверти.
    assert correct >= len(labels) // 3, (
        f"после смены разрешения прочитано {correct} из {len(labels)} — "
        "цифры перестали узнаваться (проверь интерполяцию при увеличении)"
    )
