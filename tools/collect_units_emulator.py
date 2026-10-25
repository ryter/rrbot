#!/usr/bin/env python3
"""Сбор эталонов юнитов мобильного клиента (Rush Royale в Android-эмуляторе).

Эталоны в ``assets/real`` сняты с веб-клиента: другой спрайт и другой
масштаб, поэтому на мобильном клиенте детектор находит шум. Этот скрипт
собирает эталоны заново — из настоящих кадров боя.

Как устроен мобильный клиент (в отличие от веб-клиента):

* юнит стоит на круглом основании, а сверху нарисован сам юнит;
* основание у разных юнитов разного размера (у крупных — больше), поэтому
  масштаб кадра приводится к одному по радиусу основания *каждого* юнита
  (``REF_RADIUS``);
* камера в бою ходит (арена то приближается, то отдаляется), поэтому размер
  юнита в кадре меняется — эталон приводится к каноническому масштабу по
  радиусу основания, а не по общему масштабу кадра;
* тело юнита обведено чёрной обводкой: по ней вырезается силуэт самого
  юнита (каталог ``sprite/`` в выходном каталоге).

Что попадает в эталон, которым работает детектор: вырез вокруг юнита —
тело и кольцо основания вокруг него. Основание в эталоне нужно: это яркая
кольцевая метка, по ней юнит находится надёжнее, чем по одному спрайту
(у тёмных юнитов силуэт почти без контраста, и совпадение по одному силуэту
падает до 0.2-0.4, а по вырезу с основанием — 0.65-0.75). Цвет основания у
одного и того же типа юнита меняется (свои и чужие юниты стоят на разных
подставках), поэтому цвет — только признак группировки, а не часть эталона.
Чистый силуэт без основания тоже сохраняется — в подкаталог ``sprite/``.

Эталоны квадратные (81x81 px): детектор берёт размер эталона по его высоте
и сравнивает с квадратным куском поля, и неквадратный эталон приводит к
падению.

Кадры на входе — либо полные снимки экрана эмулятора (950x1034), либо уже
канонические кадры ``assets/eval_real/frames`` (две половины арены одна
под другой). Поле ищется по кадру (плитки арены), а не по константе.

Запуск:

    # из готовых канонических кадров с разметкой позиций
    .venv/bin/python tools/collect_units_emulator.py \\
        assets/eval_real/frames/*.png --truth assets/eval_real/truth.tsv \\
        --out assets/real_emulator --types-count 8 --variants 3

    # из полных снимков экрана (позиции юнитов ищутся сами)
    .venv/bin/python tools/collect_units_emulator.py /tmp/opencode/frames2/*.png

    # тот же результат через каталог кадров и манифест позиций
    .venv/bin/python tools/collect_units_emulator.py --frames assets/eval_real/frames

На выходе: каталог с эталонами (по одному на тип юнита), подкаталог
``sprite/`` с телами юнитов без основания и файл ``index.tsv`` — что за
эталон, сколько раз встретился, где найден. Имя файла описывает вид
юнита, а не игровое название: название типа по кадру не определяется
(в ``types.tsv`` записано, какой карточке колоды соответствует спрайт).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import prepare_real_frames as PRF  # noqa: E402

# Радиус основания юнита в каноническом кадре: к нему приводятся все эталоны.
REF_RADIUS = 31.0
# Сколько примеров юнита нужно, чтобы признать его отдельным типом.
MIN_MEMBERS = 2
# На сколько типов делить найденных юнитов (в колоде пять-шесть юнитов,
# на поле попадают и чужие).
TYPES_COUNT = 8
# Геометрия выреза вокруг юнита (в долях радиуса основания).
CROP_HALF_W = 1.3
CROP_UP = 1.5
CROP_DOWN = 1.1

# Веса частей признаков типа: цвет выреза и форма силуэта.
HIST_WEIGHT = 5.0
SILHOUETTE_WEIGHT = 0.15

# Описания вида юнита по цвету — из них складывается имя файла эталона.
COLOR_NAMES = [
    ((0, 12), "red"), ((12, 26), "orange"), ((26, 40), "yellow"),
    ((40, 62), "lime"), ((62, 90), "green"), ((90, 130), "teal"),
    ((130, 160), "blue"), ((160, 175), "violet"), ((175, 181), "pink"),
]


def color_name(hue: float, sat: float, val: float) -> str:
    """Название цвета по оттенку (шкала OpenCV 0..180)."""
    if val < 70:
        return "dark"
    if sat < 45:
        return "gray"
    for (low, high), name in COLOR_NAMES:
        if low <= hue < high:
            return name
    return "mixed"


def refine_radius(gray: np.ndarray, cx: float, cy: float, hint: int) -> float:
    """Уточнить радиус основания по яркому кольцу вокруг юнита.

    Кольцо основания — яркая окружность; берём медианную яркость вдоль
    окружностей разного радиуса и ищем скачок. Радиус от Хафа берётся
    только как начальное приближение: он часто цепляется за другие круги.
    """
    height, width = gray.shape
    angles = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    best_radius = float(hint)
    best_jump = 0.0
    profile: list[tuple[float, float]] = []
    for radius in np.arange(max(8.0, hint - 12.0), min(60.0, hint + 14.0), 1.0):
        xs = cx + radius * np.cos(angles)
        ys = cy + radius * np.sin(angles)
        values = []
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                values.append(gray[yi, xi])
        profile.append((radius, float(np.median(values)) if values else 0.0))
    for (r0, v0), (r1, v1) in zip(profile, profile[1:]):
        jump = v1 - v0
        if jump > best_jump:
            best_jump = jump
            best_radius = r1
    if best_jump < 12.0:
        return float(hint)
    return best_radius


def sprite_region(gray: np.ndarray, cx: float, cy: float, radius: float):
    """Тело юнита: прямоугольник и маска силуэта (без основания).

    Спрайт обведён чёрной обводкой: по ней находится тело юнита, а круглое
    основание под юнитом в силуэт не входит. Внутренность обводки
    заливается — так остаётся только сам юнит, без подставки и без травы.
    """
    x0 = max(0, int(cx - 1.5 * radius))
    y0 = max(0, int(cy - 1.9 * radius))
    x1 = min(gray.shape[1], int(cx + 1.5 * radius))
    y1 = min(gray.shape[0], int(cy + 1.2 * radius))
    window = gray[y0:y1, x0:x1]
    if window.size == 0:
        return None
    height, width = window.shape
    dark = (window < 112).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    yy, xx = np.mgrid[0:height, 0:width]
    centre = np.hypot(xx - (cx - x0), yy - (cy - y0)) < 0.6 * radius
    best = -1
    best_score = 0
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 40:
            continue
        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]
        w = stats[index, cv2.CC_STAT_WIDTH]
        h = stats[index, cv2.CC_STAT_HEIGHT]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        # Крупная компонента, упирающаяся в край окна, — это фон поля или
        # полоса между половинами арены, а не юнит.
        if touches_border and area > 0.25 * window.size:
            continue
        score = int((labels == index)[centre].sum())
        if score > best_score:
            best_score = score
            best = index
    if best < 0 or best_score < 12:
        return None
    outline = (labels == best).astype(np.uint8)
    closed = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    # Заливка внутренности обводки: заливается всё, что не связано с краем окна.
    outside = (closed == 0).astype(np.uint8)
    scratch = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(outside, scratch, (0, 0), 2)
    body = ((closed > 0) | (outside == 1)).astype(np.uint8)
    ys, xs = np.nonzero(body)
    if len(xs) == 0:
        return None
    bx = x0 + int(xs.min())
    by = y0 + int(ys.min())
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    mask = body[ys.min():ys.min() + bh, xs.min():xs.min() + bw]
    return (bx, by, bw, bh), mask


def unit_crop(image: np.ndarray, cx: float, cy: float, radius: float):
    """Вырез вокруг юнита, приведённый к каноническому масштабу.

    Вырез содержит тело юнита и кольцо основания вокруг него: основание —
    яркая метка, по которой юнит находится надёжно. Масштаб приводится к
    ``REF_RADIUS`` по радиусу основания (у разных типов юнитов основание
    разного размера, у камеры — свой масштаб).
    """
    x0 = int(round(cx - CROP_HALF_W * radius))
    x1 = int(round(cx + CROP_HALF_W * radius))
    y0 = int(round(cy - CROP_UP * radius))
    y1 = int(round(cy + CROP_DOWN * radius))
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    patch = image[y0:y1, x0:x1]
    factor = REF_RADIUS / radius
    # Эталон всегда квадратный: детектор берёт размер эталона по его высоте
    # и сравнивает с квадратным куском поля.
    side = max(4, int(round(max(2.0 * CROP_HALF_W, CROP_UP + CROP_DOWN) * REF_RADIUS)))
    patch = cv2.resize(patch, (side, side),
                       interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
    return patch


def load_truth(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Прочитать разметку: имя кадра -> список (x, y, сторона)."""
    truth: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    if not path.exists():
        return truth
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        frame, _, team, x, y = parts[:5]
        truth[frame].append((float(x), float(y), team))
    return truth


def find_units(image: np.ndarray, truth: list[tuple[float, float, str]]):
    """Позиции юнитов в кадре: из разметки или поиском кругов (Хаф)."""
    if truth:
        return [(x, y, 0) for x, y, _ in truth]
    arenas = PRF.find_arenas(image)
    if not arenas:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in PRF.find_units(image, arenas)]


def base_color(hsv: np.ndarray, cx: float, cy: float, radius: float) -> tuple[float, float, float]:
    """Цвет кольца основания под юнитом.

    Кольцо — метка редкости юнита, у одного типа оно всегда одного цвета.
    Спрайт закрывает верхнюю часть кольца, поэтому цвет берётся по нижней
    дуге основания.
    """
    values = []
    height, width = hsv.shape[:2]
    for factor in (0.80, 0.88, 0.95):
        for angle in np.linspace(np.deg2rad(55.0), np.deg2rad(125.0), 25):
            x = int(round(cx + factor * radius * np.cos(angle)))
            y = int(round(cy + factor * radius * np.sin(angle)))
            if 0 <= x < width and 0 <= y < height:
                values.append(hsv[y, x])
    if not values:
        return (0.0, 0.0, 0.0)
    array = np.array(values, dtype=float)
    return (float(np.median(array[:, 0])), float(np.median(array[:, 1])),
            float(np.median(array[:, 2])))


def features(crop: np.ndarray, sprite: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Признаки юнита для группировки: цвет выреза и форма силуэта.

    Цвет берётся по всему вырезу: основание занимает заметную часть, и его
    цвет вместе с цветом тела отличает один тип юнита от другого. Форма
    силуэта добирает типы с похожим цветом.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0].ravel()
    sats = hsv[:, :, 1].ravel()
    hist = np.zeros(18, dtype=np.float32)
    for hue, sat in zip(hues, sats):
        if sat > 40:
            hist[min(17, int(hue) // 10)] += 1
    hist /= max(1.0, float(hist.sum()))
    silhouette = cv2.resize(mask.astype(np.float32), (24, 24), interpolation=cv2.INTER_AREA)
    silhouette = (silhouette - silhouette.mean()) / (silhouette.std() + 1e-6)
    return np.concatenate([HIST_WEIGHT * hist, SILHOUETTE_WEIGHT * silhouette.ravel()])


def kmeans_once(matrix: np.ndarray, clusters: int, iterations: int = 40) -> np.ndarray:
    """Одна попытка k-means с начальными центрами k-means++."""
    count = len(matrix)
    clusters = max(1, min(clusters, count))
    centers = [matrix[np.random.randint(count)]]
    while len(centers) < clusters:
        distances = np.min(np.stack([
            np.sum((matrix - center) ** 2, axis=1) for center in centers]), axis=0)
        total = float(distances.sum())
        if total <= 0:
            centers.append(matrix[np.random.randint(count)])
            continue
        probabilities = distances / total
        centers.append(matrix[int(np.random.choice(count, p=probabilities))])
    centers = np.stack(centers)
    labels = np.zeros(count, dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.sum((matrix - center) ** 2, axis=1) for center in centers])
        labels = np.argmin(distances, axis=0)
        for index in range(clusters):
            members = matrix[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return labels


def kmeans(matrix: np.ndarray, clusters: int, iterations: int = 40, restarts: int = 5):
    """Разбить признаки на заданное число групп (k-means, k-means++).

    Несколько попыток с разными начальными центрами: k-means сходится к
    локальному минимуму, а от него зависит и набор эталонов. Побеждает то
    разбиение, в котором юниты одного типа похожи друг на друга сильнее
    всего (критерий выбирается мерой ``quality``).
    """
    count = len(matrix)
    clusters = max(1, min(clusters, count))
    best_labels = np.zeros(count, dtype=int)
    best_quality = None
    for attempt in range(restarts):
        np.random.seed(1000 + attempt)
        # k-means++: первые центры берём подальше друг от друга
        centers = [matrix[np.random.randint(count)]]
        while len(centers) < clusters:
            distances = np.min(np.stack([
                np.sum((matrix - center) ** 2, axis=1) for center in centers]), axis=0)
            total = float(distances.sum())
            if total <= 0:
                centers.append(matrix[np.random.randint(count)])
                continue
            probabilities = distances / total
            centers.append(matrix[int(np.random.choice(count, p=probabilities))])
        centers = np.stack(centers)
        labels = np.zeros(count, dtype=int)
        for _ in range(iterations):
            distances = np.stack([np.sum((matrix - center) ** 2, axis=1) for center in centers])
            labels = np.argmin(distances, axis=0)
            for index in range(clusters):
                members = matrix[labels == index]
                if len(members):
                    centers[index] = members.mean(axis=0)
        quality = quality_of_groups(labels, matrix)
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_labels = labels.copy()
    return best_labels


def quality_of_groups(labels: np.ndarray, matrix: np.ndarray) -> float:
    """Насколько похожи юниты внутри групп (больше — лучше).

    Мера та же, по которой детектор ищет юнита: корреляция картинок
    одного типа. Считается по нормированным признакам: чем выше, тем
    ровнее получились эталоны.
    """
    score = 0.0
    count = 0
    for index in set(labels.tolist()):
        members = matrix[labels == index]
        if len(members) < 2:
            continue
        center = members.mean(axis=0)
        spread = members - center
        norms = np.sqrt((spread ** 2).sum(axis=1))
        norms[norms == 0] = 1e-6
        spread = spread / norms[:, None]
        similarity = spread @ spread.T
        np.fill_diagonal(similarity, 1.0)
        score += float(similarity.sum() - len(members))
        count += len(members) * (len(members) - 1)
    return score / max(1, count)


def main() -> int:
    global CROP_HALF_W, CROP_UP, CROP_DOWN
    parser = argparse.ArgumentParser(description="Сбор эталонов юнитов мобильного клиента")
    parser.add_argument("frames", nargs="*", help="кадры боя (png)")
    parser.add_argument("--frames", dest="frames_dir", default="",
                        help="каталог с кадрами (вместо списка файлов)")
    parser.add_argument("--out", default=str(ROOT / "assets" / "real_emulator"),
                        help="каталог для эталонов")
    parser.add_argument("--truth", default="", help="разметка позиций (tsv)")
    parser.add_argument("--manifest", default="",
                        help="манифест позиций (tsv) вместо разметки")
    parser.add_argument("--min-members", type=int, default=MIN_MEMBERS,
                        help="сколько встреч нужно, чтобы признать юнита отдельным типом")
    parser.add_argument("--types-count", type=int, default=TYPES_COUNT,
                        help="на сколько типов делить найденных юнитов")
    parser.add_argument("--preview", default="/tmp/opencode/emulator_units.png")
    parser.add_argument("--variants", type=int, default=1,
                        help="сколько разных кадров сохранять на один тип юнита")
    parser.add_argument("--crop", default=f"{CROP_HALF_W},{CROP_UP},{CROP_DOWN}",
                        help="геометрия выреза: полуширина, вверх, вниз (в долях радиуса)")
    args = parser.parse_args()

    try:
        CROP_HALF_W, CROP_UP, CROP_DOWN = (float(v) for v in args.crop.split(","))
    except ValueError:
        print(f"не разобран --crop: {args.crop}", file=sys.stderr)
        return 2

    frames = list(args.frames)
    if args.frames_dir:
        frames += sorted(str(p) for p in Path(args.frames_dir).glob("*.png"))
    if not frames:
        print("нужны кадры: файлами или каталогом --frames", file=sys.stderr)
        return 2

    np.random.seed(0)
    truth = load_truth(Path(args.truth)) if args.truth else {}
    if args.manifest:
        # Манифест: позиции юнитов уже размечены (файл <эталон> <позиции>).
        for row in load_manifest(Path(args.manifest)):
            for frame_name, position in row:
                truth.setdefault(frame_name, []).append((position[0], position[1], "-"))

    samples: list[dict] = []
    for name in frames:
        path = Path(name)
        image = cv2.imread(str(path))
        if image is None:
            print(f"кадр не прочитан: {path}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for (cx, cy, hint) in find_units(image, truth.get(path.name, [])):
            radius = refine_radius(gray, cx, cy, int(hint) if hint else 30)
            if radius < 8:
                continue
            crop = unit_crop(image, cx, cy, radius)
            if crop is None or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            region = sprite_region(gray, cx, cy, radius)
            if region is None:
                continue
            (bx, by, bw, bh), mask = region
            if bx < 0 or by < 0 or bx + bw > image.shape[1] or by + bh > image.shape[0]:
                continue
            sprite = image[by:by + bh, bx:bx + bw]
            samples.append({
                "crop": crop,
                "sprite": sprite,
                "mask": mask,
                "ring": base_color(hsv, cx, cy, radius),
                "frame": path.name,
                "x": cx,
                "y": cy,
                "radius": radius,
                "size": (crop.shape[1], crop.shape[0]),
            })

    if not samples:
        print("юниты не найдены", file=sys.stderr)
        return 1

    # Группировка по виду юнита: цвет выреза и форма силуэта.
    # Кластеры считаются методом k-means: число типов задаётся ключом
    # --types-count (в бою в колоде пять-шесть юнитов, но на поле попадают
    # и чужие).
    matrix = np.stack([features(s["crop"], s["sprite"], s["mask"]) for s in samples])
    groups = kmeans(matrix, args.types_count)
    clusters = []
    for index in sorted(set(groups)):
        members = [samples[i] for i in np.nonzero(groups == index)[0]]
        if len(members) < args.min_members:
            continue
        clusters.append({"samples": members})
    clusters.sort(key=lambda c: -len(c["samples"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sprite_dir = out_dir / "sprite"
    sprite_dir.mkdir(exist_ok=True)

    index_path = out_dir / "index.tsv"
    rows = []
    used_names: set[str] = set()
    written_files: set[str] = set()
    for number, cluster in enumerate(clusters):
        samples_c = cluster["samples"]
        # представитель типа — «медоид»: юнит, который больше всех похож на
        # остальные в группе (по корреляции вырезов). Он лучше других
        # находится детектором.
        def gray_of(sample):
            g = cv2.cvtColor(sample["crop"], cv2.COLOR_BGR2GRAY).astype(np.float32)
            return (g - g.mean()) / (g.std() + 1e-6)
        grays = [gray_of(s) for s in samples_c]
        best_index, best_score = 0, -1e9
        for i, g in enumerate(grays):
            score = sum(float((g * other).mean()) for j, other in enumerate(grays) if j != i)
            if score > best_score:
                best_score, best_index = score, i
        representative = samples_c[best_index]
        # дополнительные эталоны типа: юниты анимируются, и один кадр
        # совпадает с живым юнитом хуже, чем несколько разных
        variants = [representative]
        if args.variants > 1:
            def signature_of(sample):
                return features(sample["crop"], sample["sprite"], sample["mask"])
            chosen = [signature_of(representative)]
            for _ in range(args.variants - 1):
                best = None
                best_distance = -1.0
                for sample in samples_c:
                    if any(sample is v for v in variants):
                        continue
                    sig = signature_of(sample)
                    distance = min(float(np.mean((sig - c) ** 2)) for c in chosen)
                    if distance > best_distance:
                        best_distance = distance
                        best = sample
                if best is None:
                    break
                variants.append(best)
                chosen.append(signature_of(best))
        hsv = cv2.cvtColor(representative["sprite"], cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        saturated = pixels[(pixels[:, 1] > 60) & (representative["mask"].ravel() > 0)]
        if len(saturated):
            hue = float(np.median(saturated[:, 0]))
            sat = float(np.median(saturated[:, 1]))
            val = float(np.median(saturated[:, 2]))
        else:
            hue, sat, val = 0.0, 0.0, float(np.median(representative["crop"]))
        ring = representative["ring"]
        body_descriptor = color_name(hue, sat, val)
        ring_descriptor = color_name(*ring)
        descriptor = f"{body_descriptor}_on_{ring_descriptor}"
        name = f"unit_em_{number:02d}_{descriptor}"
        suffix = 0
        while name in used_names:
            suffix += 1
            name = f"unit_em_{number:02d}_{descriptor}_{suffix}"
        used_names.add(name)
        gray = cv2.cvtColor(representative["crop"], cv2.COLOR_BGR2GRAY)
        # тело юнита отдельно: силуэт без основания
        sprite_gray = cv2.cvtColor(representative["sprite"], cv2.COLOR_BGR2GRAY)
        body = np.where(representative["mask"] > 0, sprite_gray, 0).astype(np.uint8)
        files = []
        for variant_index, variant in enumerate(variants):
            suffix = "" if variant_index == 0 else f"_{variant_index}"
            variant_name = f"{name}{suffix}"
            variant_gray = cv2.cvtColor(variant["crop"], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(out_dir / f"{variant_name}.png"), variant_gray)
            written_files.add(f"{variant_name}.png")
            files.append(f"{variant_name}.png")
            if variant_index == 0:
                cv2.imwrite(str(sprite_dir / f"{name}.png"), body)
        examples = "; ".join(f"{s['frame']}:{s['x']:.0f},{s['y']:.0f}" for s in samples_c[:4])
        rows.append({
            "file": files[0],
            "files": ",".join(files),
            "sprite": f"sprite/{name}.png",
            "size": f"{gray.shape[1]}x{gray.shape[0]}",
            "count": len(samples_c),
            "body_color": body_descriptor,
            "base_color": ring_descriptor,
            "hue": f"{hue:.0f}",
            "sat": f"{sat:.0f}",
            "val": f"{val:.0f}",
            "base_hue": f"{ring[0]:.0f}",
            "base_sat": f"{ring[1]:.0f}",
            "base_val": f"{ring[2]:.0f}",
            "radius": f"{np.median([s['radius'] for s in samples_c]):.1f}",
            "examples": examples,
        })
        print(f"{name}: {gray.shape[1]}x{gray.shape[0]} px, встреч {len(samples_c)}, "
              f"файлов {len(files)}, тело {body_descriptor} "
              f"(H{hue:.0f} S{sat:.0f} V{val:.0f})")

    # Старые эталоны убираем только после того, как новые записаны.
    for old in out_dir.glob("*.png"):
        if old.name not in written_files:
            old.unlink()
    for old in sprite_dir.glob("*.png"):
        if old.name not in {Path(r["file"]).name for r in rows}:
            old.unlink()

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Превью: по одному примеру каждого типа.
    tiles = []
    for cluster in clusters:
        crop = cluster["samples"][0]["crop"]
        tile = cv2.resize(crop, (72, 72), interpolation=cv2.INTER_NEAREST)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tiles.append(tile)
    columns = 4
    rowsimg = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    width = max(row.shape[1] for row in rowsimg)
    padded = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) for row in rowsimg]
    cv2.imwrite(args.preview, np.vstack(padded))
    print(f"эталонов: {len(clusters)} -> {out_dir}")
    print(f"тела юнитов: {sprite_dir}")
    print(f"индекс: {index_path}")
    print(f"превью: {args.preview}")
    return 0


def load_manifest(path: Path) -> list[list[tuple[str, tuple[float, float]]]]:
    """Прочитать манифест позиций: файл <эталон> <кадр:позиции;...>."""
    rows: list[list[tuple[str, tuple[float, float]]]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        positions = []
        for item in parts[1].split(";"):
            frame_name, _, position = item.strip().partition(":")
            try:
                x, y = (float(v) for v in position.split(","))
            except ValueError:
                continue
            positions.append((frame_name, (x, y)))
        if positions:
            rows.append(positions)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
