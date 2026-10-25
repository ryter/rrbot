#!/usr/bin/env python3
"""Проверка навигации бота по записанным кадрам.

Набор лежит в ``assets/test_nav``: картинки экранов (950x1034, мобильный
клиент) и файл ``labels.txt`` с ожидаемым действием на каждом кадре.

Скрипт прогоняет настоящий бот по каждому кадру
(``rrbot --backend adb --frame <кадр>``), берёт его список кандидатов
навигации и сверяет с разметкой: первым бот должен нажать указанный
элемент, а запрещённых кандидатов в списке быть не должно.

    .venv/bin/python tools/check_nav.py
    .venv/bin/python tools/check_nav.py --verbose
    .venv/bin/python tools/check_nav.py --only offer_01.png
    .venv/bin/python tools/check_nav.py --bot cpp/build/rrbot_nav

Бот разбирает кадр в своём холсте, а холст — это настоящее разрешение
устройства (у эмулятора 941x1030). Разметка записана в координатах самих
кадров (950x1034), поэтому все ожидаемые прямоугольники пересчитываются в
координаты холста бота: проверка не зависит от того, какое сейчас разрешение
у устройства.

Код возврата 0 — все кадры сошлись, 1 — есть расхождения.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "cpp" / "build" / "rrbot"
DEFAULT_DIR = ROOT / "assets" / "test_nav"
UI_DIR = ROOT / "assets" / "ui"

CAND_RE = re.compile(r"кандидаты навигации:(.*)")
BUTTON_RE = re.compile(r"кнопки:(.*)")
IN_BATTLE_RE = re.compile(r"в бою:\s*(да|нет)")
CANVAS_RE = re.compile(r"кадр:\s*(\d+)x(\d+)")
ELEMENT_RE = re.compile(r"(\w+)\((-?\d+),(-?\d+)(?:,(\d+)x(\d+))?\)")

# Размеры надписей-шаблонов: бот находит кнопку по буквам и жмёт в
# прямоугольник шаблона. Нужны, чтобы восстановить точку нажатия.
TEXT_TEMPLATES = {"pvp": "text_pvp.png", "coop": "text_coop.png"}


class Label:
    """Ожидаемое действие на одном кадре."""

    def __init__(self, name: str, kind: str, rect: tuple[int, int, int, int],
                 forbidden: list[str], in_battle: str | None,
                 required: list[str]):
        self.name = name
        self.kind = kind
        self.rect = rect
        self.forbidden = forbidden
        self.in_battle = in_battle
        self.required = required

    @property
    def target(self) -> str:
        if self.kind == "none":
            return "ничего не нажимать"
        x, y, w, h = self.rect
        return f"{self.kind} в ({x},{y}) {w}x{h}"


def parse_labels(path: Path) -> list[Label]:
    labels: list[Label] = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 9:
            raise SystemExit(
                f"{path}:{number}: ожидалось 9 полей, а их {len(parts)}: {line}")
        name, kind = parts[0], parts[1]
        x, y, w, h = (int(v) for v in parts[2:6])
        forbidden = parts[6].split(";") if len(parts) > 6 and parts[6] != "-" else []
        in_battle = None
        if len(parts) > 7 and parts[7] != "-":
            in_battle = parts[7]
        required = parts[8].split(";") if len(parts) > 8 and parts[8] != "-" else []
        labels.append(Label(name, kind, (x, y, w, h), forbidden, in_battle, required))
    return labels


def text_template_size(kind: str) -> tuple[int, int]:
    """Размер картинки-шаблона надписи (x, y) — из assets/ui."""
    name = TEXT_TEMPLATES.get(kind)
    if not name:
        return 0, 0
    path = UI_DIR / name
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is not None:
            return image.shape[1], image.shape[0]
    except ImportError:
        pass
    return 0, 0


def png_size(path: Path) -> tuple[int, int]:
    """Размер картинки PNG (ширина, высота) — из заголовка файла.

    Разметка в labels.txt записана в координатах самих кадров, поэтому
    размер кадра нужен, чтобы пересчитать ожидаемые прямоугольники в
    координаты холста бота.
    """
    try:
        data = path.read_bytes()[16:24]
    except OSError:
        return 0, 0
    if len(data) < 8:
        return 0, 0
    return (int.from_bytes(data[0:4], "big"), int.from_bytes(data[4:8], "big"))


def scale_rect(rect: tuple[int, int, int, int],
               scale: tuple[float, float]) -> tuple[int, int, int, int]:
    """Пересчитать прямоугольник в другую систему координат."""
    x, y, w, h = rect
    sx, sy = scale
    return (round(x * sx), round(y * sy), round(w * sx), round(h * sy))


def run_bot(frame: Path, backend: str,
            bot: Path = BOT) -> tuple[list[tuple[str, int, int]],
                                     list[tuple[str, int, int, int, int]],
                                     str | None, tuple[int, int]]:
    """Запустить бота на кадре и разобрать его ответ.

    Если бот как раз пересобирается (файл перезаписывается), запуск может
    не удаться — тогда ждём и пробуем ещё раз.
    """
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            result = subprocess.run(
                [str(bot), "--backend", backend, "--frame", str(frame)],
                capture_output=True, text=True, check=False,
            )
            break
        except OSError as error:  # битый или пересобираемый бинарник
            last_error = error
            time.sleep(0.5)
    else:
        raise SystemExit(f"не удалось запустить бота: {last_error}")

    out = result.stdout
    if CAND_RE.search(out) is None:
        raise SystemExit(
            "бот не напечатал список кандидатов навигации — "
            "разбор вывода изменился, поправь tools/check_nav.py")
    buttons: list[tuple[str, int, int, int, int]] = []
    match = BUTTON_RE.search(out)
    if match:
        for kind, x, y, w, h in ELEMENT_RE.findall(match.group(1)):
            buttons.append((kind, int(x), int(y), int(w), int(h)))
    candidates: list[tuple[str, int, int]] = []
    match = CAND_RE.search(out)
    if match:
        for kind, x, y, _w, _h in ELEMENT_RE.findall(match.group(1)):
            candidates.append((kind, int(x), int(y)))
    battle = IN_BATTLE_RE.search(out)
    # Размер холста, в котором бот разбирал кадр: он зависит от настоящего
    # разрешения устройства, а разметка записана в координатах кадров.
    canvas = (0, 0)
    match = CANVAS_RE.search(out)
    if match:
        canvas = (int(match.group(1)), int(match.group(2)))
    return candidates, buttons, battle.group(1) if battle else None, canvas


def candidate_rect(kind: str, x: int, y: int,
                   buttons: list[tuple[str, int, int, int, int]],
                   scale: tuple[float, float] = (1.0, 1.0)) -> tuple[int, int, int, int]:
    """Восстановить прямоугольник кандидата.

    Бот печатает левый верхний угол прямоугольника, а жмёт в его центр.
    Чтобы понять, куда именно пришёлся бы клик, прямоугольник надо
    восстановить: для кнопок он есть в списке «кнопки», для надписей —
    это размер шаблона, а плашка «ok» — это 30 % нижней части кнопки.
    """
    if kind == "tap":
        return x, y, 20, 20
    if kind == "ok":
        for bkind, bx, by, bw, bh in buttons:
            if bkind != "continue" or bx != x:
                continue
            if abs(by + int(bh * 0.7) - y) <= 2:
                return bx, by + int(bh * 0.7), bw, int(bh * 0.3)
        return x, y, 0, 0
    for bkind, bx, by, bw, bh in buttons:
        if bkind != kind or (bx, by) != (x, y):
            continue
        return bx, by, bw, bh
    if kind == "coop":
        # Запасной путь: кнопка Co-Op может быть найдена по геометрии
        # (как часть синей кнопки), тогда она лежит на «continue».
        for bkind, bx, by, bw, bh in buttons:
            if bkind == "continue" and (bx, by) == (x, y):
                return bx, by, bw, bh
    if kind in ("pvp", "coop"):
        # Кнопка найдена по надписи: размер берём из шаблона (в координатах
        # холста бота, поэтому размер шаблона тоже пересчитываем).
        tw, th = text_template_size(kind)
        if tw and th:
            return x, y, round(tw * scale[0]), round(th * scale[1])
    return x, y, 0, 0


def click_point(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def inside(point: tuple[int, int], rect: tuple[int, int, int, int],
           margin: tuple[int, int] = (0, 0)) -> bool:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return False
    mx, my = margin
    return (x - mx) <= point[0] < x + w + mx and (y - my) <= point[1] < y + h + my


def check_frame(label: Label, frame: Path, backend: str,
                bot: Path = BOT) -> tuple[list[str], tuple[int, int]]:
    """Проверить один кадр.

    Возвращает список расхождений (пустой — ок) и размер холста, в котором
    бот разбирал кадр.
    """
    problems: list[str] = []
    candidates, buttons, in_battle, canvas = run_bot(frame, backend, bot)
    if not candidates and not buttons:
        return [f"{label.name}: бот не ответил на кадр"], canvas

    # Разметка записана в координатах кадра, а бот печатает кандидаты в
    # координатах холста: холст — это настоящее разрешение устройства, и он
    # может отличаться от разрешения записанных кадров. Поэтому все
    # прямоугольники пересчитываем в систему координат бота.
    frame_w, frame_h = png_size(frame)
    scale = (1.0, 1.0)
    if frame_w > 0 and frame_h > 0 and canvas[0] > 0 and canvas[1] > 0:
        scale = (canvas[0] / frame_w, canvas[1] / frame_h)
    # Допуск на округление: бот считает в каноническом холсте, а разметка
    # записана в координатах кадра, — координаты расходятся на единицу.
    margin = (max(1, round(2 * scale[0])), max(1, round(2 * scale[1])))

    # Первый кандидат — то, что бот нажмёт первым.
    first_kind, first_x, first_y = candidates[0] if candidates else ("—", 0, 0)
    first_rect = candidate_rect(first_kind, first_x, first_y, buttons, scale)
    first_point = click_point(first_rect)

    if label.kind == "none":
        extra = [c for c in candidates if c[0] != "tap"]
        if extra:
            listing = ", ".join(f"{k}({x},{y})" for k, x, y in extra)
            problems.append(
                f"{label.name}: жать тут нечего, а бот предлагает {listing}")
    else:
        x, y, w, h = scale_rect(label.rect, scale)
        if first_kind != label.kind:
            problems.append(
                f"{label.name}: ожидался первый кандидат «{label.kind}» "
                f"в ({x},{y}) {w}x{h}, бот предложил «{first_kind}» "
                f"({first_x},{first_y})")
        elif not inside(first_point, (x, y, w, h), margin):
            problems.append(
                f"{label.name}: кандидат «{first_kind}» жмёт в "
                f"({first_point[0]},{first_point[1]}), а нужен элемент "
                f"({x},{y}) {w}x{h}")

    # Запрещённые кандидаты.
    for rule in label.forbidden:
        kind, sep, rect = rule.partition("@")
        if kind:
            bad = [c for c in candidates if c[0] == kind]
            if bad:
                listing = ", ".join(f"{k}({x},{y})" for k, x, y in bad)
                problems.append(
                    f"{label.name}: вид «{kind}» на этом экране запрещён, "
                    f"бот предложил {listing}")
        if sep and rect:
            rx, ry, rw, rh = scale_rect(
                (int(v) for v in rect.split(",")), scale)
            for kind, x, y in candidates:
                point = click_point(candidate_rect(kind, x, y, buttons, scale))
                if inside(point, (rx, ry, rw, rh)):
                    problems.append(
                        f"{label.name}: нельзя жать ({rx},{ry}) {rw}x{rh} "
                        f"(кнопка покупки), а бот предлагает «{kind}» жать в "
                        f"({point[0]},{point[1]})")

    # Обязательные кандидаты: бот должен их предложить.
    for rule in label.required:
        kind, _, rect = rule.partition("@")
        present = [c for c in candidates if c[0] == kind]
        if not present:
            problems.append(f"{label.name}: бот не предложил «{kind}»")
            continue
        if rect:
            rx, ry, rw, rh = scale_rect(
                (int(v) for v in rect.split(",")), scale)
            points = [click_point(candidate_rect(k, x, y, buttons, scale))
                      for k, x, y in present]
            if not any(inside(point, (rx, ry, rw, rh)) for point in points):
                problems.append(
                    f"{label.name}: «{kind}» предложен, но не там: "
                    f"{' '.join(f'({p[0]},{p[1]})' for p in points)}, "
                    f"а нужен ({rx},{ry}) {rw}x{rh}")

    # Признак боя.
    if label.in_battle and in_battle and label.in_battle != in_battle:
        problems.append(
            f"{label.name}: ожидалось «в бою: {label.in_battle}», "
            f"бот сказал «{in_battle}»")
    return problems, canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка навигации по кадрам")
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
                        help="каталог с кадрами и labels.txt")
    parser.add_argument("--backend", default="adb",
                        help="бэкенд бота (adb — раскладка мобильного клиента)")
    parser.add_argument("--only", action="append", default=None,
                        help="проверить только указанный кадр")
    parser.add_argument("--verbose", action="store_true",
                        help="показать каждый кадр, а не только ошибки")
    parser.add_argument("--bot", default=str(BOT),
                        help="каким бинарником бота проверять (по умолчанию — "
                             "собранный в cpp/build/rrbot)")
    args = parser.parse_args()

    bot = Path(args.bot)
    if not bot.exists():
        # Бот может как раз пересобираться: подождём, появится ли он.
        for _ in range(20):
            time.sleep(0.5)
            if bot.exists():
                break
    if not bot.exists():
        print(f"нет собранного бота: {bot} (сначала make в cpp/)", file=sys.stderr)
        return 1

    directory = Path(args.dir)
    labels_path = directory / "labels.txt"
    if not labels_path.exists():
        print(f"нет файла {labels_path}", file=sys.stderr)
        return 1

    labels = parse_labels(labels_path)
    if args.only:
        wanted = set(args.only)
        labels = [label for label in labels if label.name in wanted]
    if not labels:
        print("в labels.txt нет ни одного кадра", file=sys.stderr)
        return 1

    total = 0
    correct = 0
    problems: list[str] = []
    canvases: set[tuple[int, int]] = set()
    frame_sizes: set[tuple[int, int]] = set()
    for label in labels:
        frame = directory / label.name
        if not frame.exists():
            problems.append(f"{label.name}: нет файла {frame}")
            total += 1
            continue
        total += 1
        issues, canvas = check_frame(label, frame, args.backend, bot)
        if canvas[0] > 0:
            canvases.add(canvas)
        size = png_size(frame)
        if size[0] > 0:
            frame_sizes.add(size)
        if issues:
            problems.extend(issues)
            status = "ОШИБКА"
        else:
            correct += 1
            status = "ок"
        if args.verbose or issues:
            print(f"  {status:6s} {label.name:22s} ожидалось: {label.target}")

    # Бот разбирает кадр в своём холсте, а холст — это настоящее разрешение
    # устройства. Кадры набора записаны в своём разрешении (950x1034), и оно
    # чуть отличается от устройства (941x1030) — это норма. А вот настоящее
    # отличие говорит, что интерфейс на устройстве поехал: починить —
    # adb shell wm size 941x1030.
    for canvas in sorted(canvases):
        print(f"холст бота: {canvas[0]}x{canvas[1]}")
    for size in sorted(frame_sizes):
        if canvases and any(abs(canvas[0] - size[0]) > size[0] * 0.05 or
                            abs(canvas[1] - size[1]) > size[1] * 0.05
                            for canvas in canvases):
            print(f"ВНИМАНИЕ: кадры набора {size[0]}x{size[1]}, а бот разбирал их "
                  f"в другом холсте — настоящее разрешение устройства отличается "
                  f"от 941x1030 (adb shell wm size 941x1030)")

    print(f"совпало {correct}/{total}")
    for problem in problems:
        print(f"  {problem}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
