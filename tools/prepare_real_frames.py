#!/usr/bin/env python3
"""Подготовка набора настоящих кадров с эталонной разметкой.

Кадры боя из Android-эмулятора (мобильный клиент Rush Royale). Разметка
готовится скриптом:

* арена находится по плиткам (две половины: сверху чужая, снизу своя);
* позиции юнитов — по круглым основаниям, которые находятся преобразованием
  Хафа (каждый юнит в игре стоит на круглом основании);
* масштаб кадра приводится к одному: по медианному радиусу оснований, чтобы
  ход камеры (она то приближается, то отдаляется) не влиял на размер юнитов.

Результат проверяется глазами: скрипт рисует найденные основания на кадре
(``--preview``), и список позиций можно поправить руками в ``truth.tsv``.

На выходе: кадры в каноническом размере (две половины арены одна под другой,
между ними полоса волны) и файл разметки в формате TSV:

    <имя кадра> <TAB> <тип> <TAB> <сторона> <TAB> <x> <TAB> <y>

Тип в настоящей игре по основанию не определяется, поэтому в разметке он
записан как «-» (неизвестно), а сторона — по половине арены.

Запуск:

    .venv/bin/python tools/prepare_real_frames.py --frames /tmp/opencode/frames2 \\
        --out assets/eval_real --count 12 --start 105 --step 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Радиус основания юнита в каноническом кадре (все кадры приводятся к нему).
REF_RADIUS = 31.0
# Размер половины арены в каноническом кадре.
HALF_W = 400
HALF_H = 240
# Полоса между половинами арены (там счётчик волны и таймер).
GAP = 42
BOARD_W = HALF_W
BOARD_H = HALF_H * 2 + GAP
# Радиус основания юнита в кадре эмулятора.
RING_MIN = 26
RING_MAX = 44


def find_arenas(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Найти половины арены по плиткам.

    Плитки арены — яркие жёлто-зелёные (H≈34, S≈190, V≈200). Верхняя
    половина (чужая) и нижняя (своя) делятся кадр пополам; в каждую группу
    попадают все куски плиток, и по ним берётся общий прямоугольник — так
    арена находится даже когда её часть закрыта эффектом волны.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    mask = ((hue >= 28) & (hue <= 40) & (sat > 170) & (val > 170)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    groups: list[list[tuple[int, int, int, int]]] = [[], []]
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 400:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        groups[0 if y + h / 2 < image.shape[0] / 2 else 1].append((x, y, w, h))

    boxes = []
    for group in groups:
        if not group:
            return []
        # Крупнейший кусок — сама арена. Остальные куски добавляются, только
        # если они лежат рядом с ней: иначе в прямоугольник попали бы
        # посторонние яркие пятна (иконки карточек, портал).
        wide = [b for b in group if b[2] >= 150 and b[3] >= 100 and b[2] * b[3] >= 8000]
        if not wide:
            return []
        seed = max(wide, key=lambda b: b[2] * b[3])
        x1, y1 = seed[0], seed[1]
        x2, y2 = seed[0] + seed[2], seed[1] + seed[3]
        for _ in range(4):
            changed = False
            for x, y, w, h in group:
                cx, cy = x + w / 2, y + h / 2
                near_y = y1 - 40 <= cy <= y2 + 40
                near_x = x1 - 90 <= cx <= x2 + 90
                if near_y and near_x:
                    nx1, ny1 = min(x1, x), min(y1, y)
                    nx2, ny2 = max(x2, x + w), max(y2, y + h)
                    if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                        changed = True
            if not changed:
                break
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    top, bottom = boxes
    if bottom[1] - (top[1] + top[3]) < 40:
        return []
    return [top, bottom]


def find_units(image: np.ndarray, arenas: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
    """Найти основания юнитов (круги) внутри половин арены.

    Возвращает список (x, y, радиус).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20, param1=100,
                               param2=30, minRadius=RING_MIN, maxRadius=RING_MAX)
    found: list[tuple[int, int, int]] = []
    if circles is None:
        return found
    for x, y, radius in circles[0]:
        cx, cy = int(x), int(y)
        for ax, ay, aw, ah in arenas:
            if ax - 12 <= cx <= ax + aw + 12 and ay - 12 <= cy <= ay + ah + 12:
                found.append((cx, cy, int(radius)))
                break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка кадров с разметкой")
    parser.add_argument("--frames", default="/tmp/opencode/frames2", help="каталог с кадрами")
    parser.add_argument("--out", default=str(ROOT / "assets" / "eval_real"))
    parser.add_argument("--count", type=int, default=12, help="сколько кадров взять")
    parser.add_argument("--start", type=int, default=0, help="с какого кадра начинать")
    parser.add_argument("--step", type=int, default=1, help="шаг по кадрам")
    parser.add_argument("--preview", default="/tmp/opencode/eval_real_preview.png")
    args = parser.parse_args()

    source = Path(args.frames)
    files = sorted(source.glob("*.png"))
    if not files:
        print(f"в каталоге {source} нет кадров", file=sys.stderr)
        return 1

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.tsv"

    previews = []
    written = 0
    with truth_path.open("w", encoding="utf-8") as truth:
        for path in files[args.start :: args.step]:
            if written >= args.count:
                break
            image = cv2.imread(str(path))
            if image is None:
                continue
            arenas = find_arenas(image)
            if not arenas:
                continue
            units = find_units(image, arenas)
            if not units:
                continue

            # Масштаб кадра: приводим радиус оснований к опорному, чтобы
            # размер юнита не зависел от положения камеры.
            radii = sorted(u[2] for u in units)
            median_radius = radii[len(radii) // 2]
            factor = REF_RADIUS / max(1.0, float(median_radius))

            board = np.zeros((BOARD_H, BOARD_W, 3), dtype=np.uint8)
            offsets = []
            for index, (x, y, w, h) in enumerate(arenas):
                half = image[y : y + h, x : x + w]
                resized = cv2.resize(half, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                                     interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
                top = 0 if index == 0 else HALF_H + GAP
                # Центрируем по горизонтали: кадр обрезан по-разному.
                left = max(0, (BOARD_W - resized.shape[1]) // 2)
                # Обрезка по краям: кадр может оказаться шире холста.
                src_x = max(0, -left)
                src_y = max(0, -top)
                dst_x = max(0, left)
                dst_y = max(0, top)
                width = min(BOARD_W - dst_x, resized.shape[1] - src_x)
                height = min(BOARD_H - dst_y, resized.shape[0] - src_y)
                if width > 0 and height > 0:
                    board[dst_y : dst_y + height, dst_x : dst_x + width] = \
                        resized[src_y : src_y + height, src_x : src_x + width]
                offsets.append((x, y, w, h, factor, left, top))

            name = path.name
            cv2.imwrite(str(frames_dir / name), board)
            for cx, cy, _ in units:
                index = 0
                for i, (ax, ay, aw, ah) in enumerate(arenas):
                    if ax - 12 <= cx <= ax + aw + 12 and ay - 12 <= cy <= ay + ah + 12:
                        index = i
                        break
                ax, ay, aw, ah, factor, left, top = offsets[index]
                bx = (cx - ax) * factor + left
                by = (cy - ay) * factor + top
                team = "enemy" if index == 0 else "ally"
                truth.write(f"{name}\t-\t{team}\t{bx:.1f}\t{by:.1f}\n")

            # Превью: найденные основания на исходном кадре.
            preview = image.copy()
            for cx, cy, radius in units:
                cv2.circle(preview, (cx, cy), radius, (0, 0, 255), 2)
            previews.append(cv2.resize(preview, (235, 258)))
            written += 1
            print(f"{name}: арены {arenas}, оснований {len(units)}, радиус {median_radius}, "
                  f"коэффициент {factor:.3f}")

    print(f"записано кадров: {written} -> {frames_dir}")
    print(f"разметка: {truth_path}")
    if previews:
        sheet = np.hstack(previews[:8])
        cv2.imwrite(args.preview, sheet)
        print(f"превью: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
