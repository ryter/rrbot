"""Тесты C++-детектора юнитов и планировщика.

Замер идёт по эталонной разметке симулятора (тот же источник, что у
tools/evaluate_vision.py), поэтому числа C++-детектора можно сравнивать с
Python-прототипом.

Проверяется:
  * детектор находит юниты (recall/precision не ниже порога);
  * тип юнита и сторона определяются верно;
  * C++-детектор совпадает с Python-прототипом на одних и тех же кадрах;
  * планировщик выбирает юнита, который перебивает врага, ждёт без маны и
    умеет копить ману под более сильного юнита;
  * на настоящих кадрах (набор в assets/eval_real) инструмент отрабатывает
    без сбоя — сами эталоны для мобильного клиента пока не подходят,
    см. комментарий к тесту.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "tools" / "evaluate_vision_cpp.py"
VISION_TOOL = ROOT / "cpp" / "build" / "rrvision"
PLAN_TOOL = ROOT / "cpp" / "build" / "rrplan"
REAL_DATASET = ROOT / "assets" / "eval_real"

# Строка со сводкой от драйвера:
# «C++ порог 0.78 | recall 1.000 | precision 1.000 | тип 1.000 | сторона 1.000 | ...»
SUMMARY_RE = re.compile(
    r"C\+\+ порог (?P<threshold>[\d.]+) \| recall (?P<recall>[\d.]+) \| "
    r"precision (?P<precision>[\d.]+) \| тип (?P<type>[\d.]+) \| сторона (?P<team>[\d.]+)"
)
PYTHON_RE = re.compile(
    r"Python: recall (?P<recall>[\d.]+) \| precision (?P<precision>[\d.]+) \| "
    r"тип (?P<type>[\d.]+) \| сторона (?P<team>[\d.]+)"
)
TOOL_SUMMARY_RE = re.compile(r"^ИТОГО: (?P<body>.*)$", re.MULTILINE)

# Порог, ниже который детектор считается негодным.
#
# На симуляторе при пороге 0.78 детектор находит около 96% юнитов: часть
# юнитов перекрывается другими (выборка случайная, юниты могут встать друг
# на друга) — на них приходится и у Python-прототипа ровно столько же.
MIN_RECALL = 0.90
MIN_PRECISION = 0.95


def _run_driver(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(DRIVER), *args], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"инструмент завершился с кодом {result.returncode}:\n{result.stderr}"
    return result.stdout


@pytest.mark.skipif(not VISION_TOOL.exists(), reason="C++-утилита замера не собрана")
def test_детектор_находит_юнитов_по_эталонной_разметке():
    """Основной замер: C++-детектор на кадрах симулятора с разметкой."""
    out = _run_driver("--mode", "sim", "--steps", "30")
    match = SUMMARY_RE.search(out)
    assert match, f"в выводе нет сводки:\n{out}"
    values = {key: float(value) for key, value in match.groupdict().items()}
    assert values["recall"] >= MIN_RECALL, f"recall слишком низкий: {values}"
    assert values["precision"] >= MIN_PRECISION, f"precision слишком низкий: {values}"
    assert values["type"] >= 0.90, f"тип определяется плохо: {values}"
    assert values["team"] >= 0.90, f"сторона определяется плохо: {values}"


@pytest.mark.skipif(not VISION_TOOL.exists(), reason="C++-утилита замера не собрана")
def test_cpp_совпадает_с_python_прототипом():
    """Порт должен вести себя как оригинал: те же числа на тех же кадрах."""
    out = _run_driver("--mode", "sim", "--steps", "15", "--compare")
    cpp = SUMMARY_RE.search(out)
    python = PYTHON_RE.search(out)
    assert cpp and python, f"в выводе нет сводок:\n{out}"
    for key in ("recall", "precision", "type", "team"):
        assert abs(float(cpp.group(key)) - float(python.group(key))) < 0.02, (
            f"{key}: C++ {cpp.group(key)} против Python {python.group(key)}"
        )


@pytest.mark.skipif(not VISION_TOOL.exists(), reason="C++-утилита замера не собрана")
def test_инструмент_работает_на_настоящих_кадрах():
    """Набор настоящих кадров из эмулятора: инструмент отрабатывает без сбоя.

    Точность здесь низкая: эталоны в assets/real сняты с веб-клиента, а в
    мобильном клиенте юниты выглядят иначе. Тест проверяет только то, что
    конвейер работает и метрики считаются (подробности — в отчёте).
    """
    if not (REAL_DATASET / "truth.tsv").exists():
        pytest.skip("нет набора настоящих кадров")
    result = subprocess.run(
        [
            str(VISION_TOOL),
            "--frames", str(REAL_DATASET / "frames"),
            "--truth", str(REAL_DATASET / "truth.tsv"),
            "--templates", str(ROOT / "assets" / "real"),
            "--map", "knight=unit_00.png,archer=unit_01.png,golem=unit_02.png,"
                    "samurai=unit_03.png,warden=unit_04.png",
            "--types", "knight,archer,golem,samurai,warden",
            "--threshold", "0.60", "--scale", "1.2", "--radius", "30",
            "--team-by-position", "--split", "0.5",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"утилита упала:\n{result.stderr}"
    match = TOOL_SUMMARY_RE.search(result.stdout)
    assert match, f"в выводе нет сводки:\n{result.stdout}"
    assert "recall=" in match.group("body")


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
def test_планировщик_выбирает_юнита_который_перебивает_врага():
    out = subprocess.run([str(PLAN_TOOL), "--scenario", "A"], capture_output=True, text=True,
                         check=False).stdout
    assert "решение=spawn" in out
    assert "юнит=knight" in out, f"выбран не тот юнит:\n{out}"


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
def test_планировщик_ждёт_без_маны():
    out = subprocess.run([str(PLAN_TOOL), "--scenario", "B"], capture_output=True, text=True,
                         check=False).stdout
    assert "решение=wait" in out, f"без маны ждать надо:\n{out}"
    assert "мана" in out


@pytest.mark.skipif(not PLAN_TOOL.exists(), reason="C++-демо планировщика не собрана")
def test_планировщик_копит_ману_под_сильного_юнита():
    out = subprocess.run([str(PLAN_TOOL), "--scenario", "D"], capture_output=True, text=True,
                         check=False).stdout
    assert "НАКОПЛЕНИЕ да" in out, f"накопление маны не сработало:\n{out}"
    assert "копим ману" in out


def test_эталоны_из_конфига_на_месте():
    """Каждому типу из config/live.yaml соответствует существующий эталон."""
    import yaml

    config = yaml.safe_load((ROOT / "config" / "live.yaml").read_text(encoding="utf-8"))
    mapping = config["vision"]["template_map"]
    assert mapping, "в config/live.yaml нет сопоставления типов и эталонов"
    for kind, filename in mapping.items():
        path = ROOT / "assets" / "real" / filename
        assert path.exists(), f"эталон для {kind} не найден: {path}"


@pytest.mark.skipif(not VISION_TOOL.exists(), reason="C++-утилита замера не собрана")
def test_значения_в_cpp_совпадают_с_конфигом():
    """C++-дефолты продублированы из config/live.yaml — проверяем, что не разошлись.

    В C++ нет парсера yaml, поэтому характеристики юнитов и сопоставление
    «тип -> файл» пришлось продублировать. Тест ловит расхождение.
    """
    import yaml

    config = yaml.safe_load((ROOT / "config" / "live.yaml").read_text(encoding="utf-8"))
    out = subprocess.run([str(VISION_TOOL), "--print-defaults"], capture_output=True, text=True,
                         check=False).stdout
    templates: dict[str, str] = {}
    specs: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        if line.startswith("ЭТАЛОН "):
            _, kind, filename = line.split()
            templates[kind] = filename
        elif line.startswith("ТИП "):
            parts = line.split()
            name = parts[1]
            values: dict[str, str] = {}
            for item in parts[2:]:
                key, value = item.split("=", 1)
                values[key] = value
            specs[name] = values

    assert templates == config["vision"]["template_map"], "сопоставление типов и эталонов разошлось"
    for kind, unit in config["vision"]["units"].items():
        assert kind in specs, f"в C++ нет типа {kind}"
        values = specs[kind]
        assert float(values["cost"]) == float(unit["cost"])
        assert float(values["hp"]) == float(unit["hp"])
        assert float(values["damage"]) == float(unit["damage"])
        assert float(values["interval"]) == float(unit["attack_interval"])
        assert float(values["speed"]) == float(unit["speed"])
        assert float(values["range"]) == float(unit["attack_range"])
        assert float(values["radius"]) == float(unit["radius"])
        assert values["family"] == unit["family"]
        assert values["counters"] == ",".join(unit.get("counters", []))
