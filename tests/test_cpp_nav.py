"""Тесты навигации C++-бота по записанным кадрам.

Набор лежит в ``assets/test_nav``: картинки экранов (950x1034, мобильный
клиент) и файл ``labels.txt`` с ожидаемым действием на каждом кадре.
Инструмент ``tools/check_nav.py`` гоняет бота по кадрам и сверяет его
кандидаты навигации с разметкой: первым бот должен нажать указанный
элемент, а запрещённых кандидатов в списке быть не должно.

Тест падает, когда бот где-то ошибается: правила навигации в
``cpp/src/hud.cpp`` правятся руками, и регрессия должна быть видна сразу,
без живого эмулятора.

ИЗВЕСТНЫЕ ДЕФЕКТЫ (состояние на 20.10.2026, в боте не исправлены) —
на них тест падает намеренно, список ниже описывает, что чинить:

1. Экран подбора соперника (``matchmaking_01/02``): кнопка «Отмена»
   (402,916) 147x50 не находится — правило ``cancel`` в
   ``cpp/src/hud.cpp`` отсекает её и по высоте (нужно 57 px, есть 50), и
   по площади (нужно 7456, есть 5861). Вместо неё бот предлагает нажать
   надпись «PvP» на баннере (446,169) — это не кнопка.
2. Экран итогов (``battle_victory``, ``result_victory``): лента «Победа»
   (синяя полоса 475x70 в середине экрана) принимается за синюю кнопку
   «Понятно» (``ok``), из-за неё бот не доходит до «Продолжить».
3. Окно предложения (``offer_01/02``): кнопка покупки «6,30 €»
   (399,867) не опознаётся как покупка — знак валюты не совпадает с
   шаблоном (0.47 при пороге 0.8), а цифры в цене не читаются, — и
   попадает в кандидаты под видом «PvP».
4. Экран награды (``reward_panel_01/02/03``): оранжевая кнопка
   «Заработать» (472,849) принимается за кнопку «PvP».
5. Главное меню (``menu_pvp_02``, ``menu_pvp_05``): красные значки HUD
   и бейдж скидки «-90%» принимаются за крестик закрытия окна:
   ``is_close_badge`` считает белым глифом все светлые пиксели, и
   одинокая белая точка рядом с надписью делает прямоугольник
   квадратным.
6. Экран героя (``hero_screen``): иконка в нижней полосе меню
   (437,928) принимается за кнопку «Продолжить».
7. Бой и загрузка (``battle_01``, ``battle_03``, ``loading``): цветные
   пятна на картинке принимаются за кнопки «Играть»/«PvP».
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "cpp" / "build" / "rrbot"
FIXTURES = ROOT / "assets" / "test_nav"
TOOL = ROOT / "tools" / "check_nav.py"
LABELS = FIXTURES / "labels.txt"

SCORE_RE = re.compile(r"совпало (\d+)/(\d+)")


def _labels() -> list[tuple[str, ...]]:
    if not LABELS.exists():
        return []
    out = []
    for line in LABELS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(tuple(line.split()))
    return out


@pytest.mark.skipif(not BOT.exists(), reason="C++-бот не собран")
@pytest.mark.skipif(not _labels(), reason="нет набора кадров навигации")
def test_навигация_по_кадрам_верна():
    """Бот должен нажимать на кадрах ровно то, что записано в разметке.

    Инструмент возвращает ненулевой код, если хоть один кадр разошёлся,
    и печатает, что ожидалось и что бот предложил.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True, text=True, check=False, cwd=str(ROOT),
    )
    match = SCORE_RE.search(result.stdout)
    assert match, (
        "инструмент не отчитался о кадрах:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    total = int(match.group(2))
    assert int(match.group(1)) == total, (
        f"навигация разошлась с разметкой: совпало {match.group(1)}/{total}\n"
        f"{result.stdout}"
    )


@pytest.mark.skipif(not _labels(), reason="нет набора кадров навигации")
def test_набор_кадров_целый():
    """Каждому кадру в разметке соответствует файл, и наоборот."""
    on_disk = {p.name for p in FIXTURES.glob("*.png")}
    in_labels = {label[0] for label in _labels()}
    missing = sorted(in_labels - on_disk)
    extra = sorted(on_disk - in_labels)
    assert not missing, f"в разметке есть кадры без файла: {missing}"
    assert not extra, f"в каталоге есть кадры без разметки: {extra}"


@pytest.mark.skipif(not _labels(), reason="нет набора кадров навигации")
def test_набор_покрывает_разные_экраны():
    """Набор должен закрывать все характерные состояния меню.

    Без этого набор превращается в проверку одного и того же экрана, и
    регрессии в правилах навигации остаются незамеченными.
    """
    labels = _labels()
    kinds = {label[1] for label in labels}
    expected = {"pvp", "close", "cancel", "continue", "none"}
    assert expected <= kinds, f"не покрыты виды действий: {expected - kinds}"

    # Виды, которые бот должен не только уметь, но и предлагать вторыми:
    # кнопка «Co-Op» рядом с «PvP» и синяя «Понятно» внутри окна.
    required = set()
    for label in labels:
        if label[8] != "-":
            required |= {part for part in label[8].split(";") if part}
    assert {"coop", "ok"} <= required, \
        f"не проверяются вторые кандидаты: {{'coop', 'ok'}} - {required}"

    # Все характерные экраны должны быть в наборе.
    families = {
        "menu_pvp": "главное меню",
        "matchmaking": "подбор соперника",
        "offer": "окно предложения",
        "chest": "окно сундука",
        "gold_dialog": "окно «Золото»",
        "hero_screen": "экран героя",
        "battle_": "бой",
        "result": "экран итогов",
        "reward_panel": "экран награды",
        "loading": "загрузка",
    }
    names = {label[0] for label in labels}
    for prefix, title in families.items():
        assert any(name.startswith(prefix) for name in names), \
            f"в наборе нет экрана «{title}» ({prefix}*)"
    assert len(labels) >= 15, "набор слишком мал: на каждый экран нужен кадр"


@pytest.mark.skipif(not _labels(), reason="нет набора кадров навигации")
def test_разметка_в_системе_координат_холста():
    """Координаты разметки — в системе холста 950x1034."""
    kinds = {"pvp", "coop", "close", "cancel", "ok", "continue", "none"}
    for label in _labels():
        name, kind = label[0], label[1]
        assert len(label) == 9, f"{name}: в строке должно быть 9 полей"
        x, y, w, h = (int(v) for v in label[2:6])
        assert 0 <= x < 950 and 0 <= y < 1034, f"{name}: точка вне холста"
        assert 0 <= w <= 950 and 0 <= h <= 1034, f"{name}: прямоугольник вне холста"
        assert kind in kinds, f"{name}: неизвестный вид действия «{kind}»"
        assert label[7] in {"да", "нет"}, f"{name}: неизвестный признак боя «{label[7]}»"
