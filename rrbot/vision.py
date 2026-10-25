"""Зрение: сопоставление шаблонов, NMS, определение стороны и маны.

Шаблоны ищутся всегда в каноническом пространстве поля, поэтому детектор
не зависит от разрешения экрана и от того, как игрок расположил окно.
"""

from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .config import AppConfig, FactionConfig, VisionConfig
from .geometry import Calibration, warp_to_board
from .models import Detection, GameState, Lane, Team, Unit, UnitSpec

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Разделение слипшихся спрайтов.
#
# Обычное сопоставление по шаблону находит один максимум на каждый юнит.
# Когда два юнита стоят рядом, их тайлы накладываются, и шаблон перестаёт
# различать их: остаётся одна находка вместо двух. Ниже — параметры
# разделения. Они заданы прямо здесь, а не в конфиге, чтобы Python-прототип
# и C++-порт считали одинаково (в C++ конфиг yaml не разбирается).
#
# * SPLIT_MIN_SEPARATION — насколько два юнита должны быть разнесены, чтобы
#   их можно было считать разными (меньше — это дубликат одной находки);
# * SPLIT_PEAK_FLOOR — минимальный отклик, ниже которого кандидат в
#   слипшейся находке не рассматривается;
# * SPLIT_MIN_VISIBLE — сколько пикселей шаблона должно остаться видно,
#   чтобы оценка «по видимой части» была Meaningful;
# * SPLIT_RADIUS — как далеко от видимого юнита искать спрятанный;
# * SPLIT_RING_CANDIDATES — сколько лучших точек отклика берётся в кольце
#   вокруг каждой находки;
# * SPLIT_MIN_SCORE — оценка «по видимой части», при которой кандидат
#   считается юнитом;
# * SPLIT_EXCLUDE_MARGIN — на сколько расширяется зона перекрытия, когда
#   выбрасываются закрытые пиксели (находка может быть смещена на пиксель).
# --------------------------------------------------------------------------
SPLIT_MIN_SEPARATION = 11.0
SPLIT_PEAK_FLOOR = 0.40
SPLIT_MIN_VISIBLE = 150
SPLIT_RADIUS = 48.0
SPLIT_RING_CANDIDATES = 4
SPLIT_MIN_SCORE = 0.75
SPLIT_EXCLUDE_MARGIN = 2

import os as _os  # noqa: E402  (временный перебор параметров)

SPLIT_PEAK_FLOOR = float(_os.environ.get("RR_SPLIT_FLOOR", SPLIT_PEAK_FLOOR))
SPLIT_MIN_SCORE = float(_os.environ.get("RR_SPLIT_SCORE", SPLIT_MIN_SCORE))
SPLIT_MIN_VISIBLE = int(_os.environ.get("RR_SPLIT_VISIBLE", SPLIT_MIN_VISIBLE))
SPLIT_MIN_SEPARATION = float(_os.environ.get("RR_SPLIT_SEP", SPLIT_MIN_SEPARATION))
SPLIT_RING_CANDIDATES = int(_os.environ.get("RR_SPLIT_RING", SPLIT_RING_CANDIDATES))
SPLIT_EXCLUDE_MARGIN = int(_os.environ.get("RR_SPLIT_MARGIN", SPLIT_EXCLUDE_MARGIN))
SPLIT_USE_RING = int(_os.environ.get("RR_SPLIT_USE_RING", 1))


@dataclass
class Template:
    """Шаблон одного типа юнита."""

    name: str
    image: np.ndarray  # оттенки серого (uint8)
    bgr: np.ndarray  # цветная версия
    size: int
    # Маска «содержимого» тайла: пиксели, которые принадлежат самому юниту
    # (тело, контур, глиф). Чёрные углы тайла — это фон, а не юнит: у соседних
    # юнитов они перекрывают друг друга, и из-за этого обычный отклик
    # падает. Оценка «по видимой части» их не учитывает.
    content: np.ndarray  # bool, того же размера, что и image


class TemplateBank:
    """Загрузка и хранение шаблонов."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.templates: dict[str, Template] = {}

    def load(self, names: Iterable[str], mapping: dict[str, str] | None = None) -> None:
        """Загрузить шаблоны. ``mapping`` переопределяет имя файла для типа."""
        self.templates.clear()
        mapping = mapping or {}
        for name in names:
            filename = mapping.get(name, f"{name}.png")
            path = self.directory / filename
            if not path.exists():
                log.warning("шаблон %s не найден по пути %s", name, path)
                continue
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                log.warning("не удалось прочитать шаблон %s", path)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.templates[name] = Template(
                name=name,
                image=gray,
                bgr=bgr,
                size=bgr.shape[0],
                content=_content_mask(gray),
            )
        # Остальные файлы каталога грузим как дополнительные эталоны: в живой
        # игре типов юнитов больше, чем описано в конфиге, и лишние эталоны
        # помогают находить юниты вообще.
        used = {mapping.get(n, f"{n}.png") for n in names}
        if self.directory.exists():
            for path in sorted(self.directory.glob("*.png")):
                if path.name in used:
                    continue
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                stem = path.stem
                self.templates.setdefault(
                    stem,
                    Template(
                        name=stem,
                        image=gray,
                        bgr=bgr,
                        size=bgr.shape[0],
                        content=_content_mask(gray),
                    ),
                )
        log.info("загружено шаблонов: %d", len(self.templates))

    def get(self, name: str) -> Template | None:
        return self.templates.get(name)

    def __len__(self) -> int:
        return len(self.templates)


def non_max_suppression(dets: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Отсечение перекрывающихся находок."""
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: list[Detection] = []
    for det in ordered:
        keep_box = _box_of(det)
        if all(_iou(keep_box, _box_of(k)) < iou_threshold for k in keep):
            keep.append(det)
    return keep


def _box_of(det: Detection) -> tuple[float, float, float, float]:
    x, y, w, h = det.box
    return (float(x), float(y), float(x + w), float(y + h))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _hue_in_ranges(hue: float, ranges: Sequence[tuple[float, float]]) -> bool:
    for low, high in ranges:
        if low <= high:
            if low <= hue <= high:
                return True
        else:  # диапазон через 0 градусов
            if hue >= low or hue <= high:
                return True
    return False


def _content_mask(gray: np.ndarray) -> np.ndarray:
    """Какие пиксели тайла принадлежат юниту, а какие — фону.

    Тайл спрайта квадратный, и углы у него чёрные (фон). У соседних юнитов
    эти чёрные углы перекрывают друг друга, и обычный отклик по всему
    прямоугольнику падает. Оценка «по видимой части» такие пиксели не
    учитывает, поэтому нужно знать, где кончается юнит.
    """
    mask = gray > 0
    # Если фон не чёрный (настоящие эталоны из живой игры), маска покрывает
    # весь тайл — оценка «по видимой части» тогда совпадает с обычной.
    if mask.all():
        return np.ones(gray.shape, dtype=bool)
    return mask


def local_maxima(res: np.ndarray, floor: float) -> list[tuple[int, int, float]]:
    """Локальные максимумы карты отклика (окно 3x3), отсортированные по убыванию.

    Возвращает список ``(x, y, score)`` — координаты левого верхнего угла
    окна шаблона.
    """
    if res.size == 0:
        return []
    kernel = np.ones((3, 3), np.float32)
    dilated = cv2.dilate(res, kernel)
    ys, xs = np.where((res >= dilated - 1e-6) & (res >= floor))
    if len(xs) == 0:
        return []
    scores = res[ys, xs]
    order = np.argsort(-scores, kind="stable")
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


class VisionEngine:
    """Полный конвейер: кадр -> состояние поля."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.vision: VisionConfig = config.vision
        self.bank = TemplateBank(self.vision.templates_dir)
        self.specs: dict[str, UnitSpec] = self.vision.specs()
        self._loaded = False

    # ------------------------------------------------------------------ setup
    def load_templates(self) -> None:
        self.bank.load(self.vision.units.keys(), self.vision.template_map)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_templates()

    # --------------------------------------------------------------- matching
    def match_template(
        self,
        board: np.ndarray,
        name: str,
        threshold: float | None = None,
    ) -> list[Detection]:
        """Найти все вхождения шаблона в каноническом поле."""
        dets, _ = self.match_template_map(board, name, threshold)
        return dets

    def match_template_map(
        self,
        board: np.ndarray,
        name: str,
        threshold: float | None = None,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Найти вхождения шаблона и вернуть вместе с ними карту отклика.

        Карта отклика нужна для разделения слипшихся спрайтов: внутри одной
        находки могут прятаться ещё юниты, и их видно по второму максимуму
        отклика.
        """
        self.ensure_loaded()
        tpl = self.bank.get(name)
        if tpl is None:
            return [], None
        th = self.vision.match_threshold if threshold is None else threshold
        haystack = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY) if self.vision.grayscale else board
        needle = tpl.image if self.vision.grayscale else tpl.bgr
        if haystack.shape[0] < needle.shape[0] or haystack.shape[1] < needle.shape[1]:
            return [], None
        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= th)
        if len(xs) == 0:
            return [], res
        scores = res[ys, xs]
        # Кандидатов на низком пороге бывают десятки тысяч (почти каждый
        # пиксель поля «похож» на силуэт). Строить по ним Detection и потом
        # давить дубликаты — квадратичная работа, кадр не успевает за игрой.
        # Оставляем только самые сильные: NMS всё равно оставит не больше
        # max_per_template находок, и они будут самыми сильными.
        if len(scores) > self.vision.max_candidates:
            top = np.argpartition(-scores, self.vision.max_candidates)[: self.vision.max_candidates]
            ys, xs, scores = ys[top], xs[top], scores[top]
        dets: list[Detection] = []
        h, w = needle.shape[:2]
        for y, x, score in zip(ys.tolist(), xs.tolist(), scores.tolist()):
            dets.append(
                Detection(
                    name=name,
                    score=float(score),
                    cx=float(x + w / 2.0),
                    cy=float(y + h / 2.0),
                    box=(int(x), int(y), int(w), int(h)),
                )
            )
        dets = non_max_suppression(dets, self.vision.nms_iou)
        return dets[: self.vision.max_per_template], res

    def detect_units(self, board: np.ndarray) -> list[Detection]:
        """Найти все юниты и определить их сторону по цвету."""
        all_dets: list[Detection] = []
        responses: dict[str, np.ndarray] = {}
        for name in self.vision.units:
            dets, res = self.match_template_map(board, name)
            all_dets.extend(dets)
            if res is not None:
                responses[name] = res
        all_dets = non_max_suppression(all_dets, self.vision.nms_iou)
        all_dets.extend(self.split_merged(board, all_dets, responses))
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        for det in all_dets:
            hue, sat, val = self._sample_color(hsv, det)
            det.hue, det.saturation = hue, sat
            det.team = self.classify_team(hue, sat, val)
        return all_dets

    # ------------------------------------------------------- разделение пар
    def split_merged(
        self,
        board: np.ndarray,
        dets: list[Detection],
        responses: dict[str, np.ndarray],
    ) -> list[Detection]:
        """Найти юниты, спрятанные под другими юнитами.

        Обычное сопоставление по шаблону даёт один максимум отклика на
        слипшуюся пару: тайлы юнитов накладываются, и шаблон перестаёт их
        различать. Видимый юнит при этом находится точно, а вот спрятанный
        под ним теряется.

        Спрятанный юнит виден по «второму максимуму» отклика рядом с
        найденной находкой. Его отклик сам по себе невелик (шаблон
        наполовину перекрыт), поэтому он оценивается не по всему
        прямоугольнику, а **по видимой части**: из шаблона выбрасываются
        пиксели, закрытые другими юнитами, и корреляция считается по
        оставшимся. Там, где видна настоящая форма юнита, отклик снова
        близок к единице.

        Кандидаты ищутся не только среди локальных максимумов отклика:
        у спрятанного юнита своего максимума может не быть вовсе (слипшийся
        спрайт даёт один общий максимум). Поэтому рядом с каждой находкой
        берётся несколько лучших точек отклика в кольце вокруг неё — как
        при поиске объектов со взаимным запретом перекрытий.
        """
        if not dets or not responses:
            return []
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY) if self.vision.grayscale else board
        # Зоны перекрытия рисуем один раз в координатах поля: раньше для
        # каждого кандидата перебирались все находки и вычитались из
        # шаблона по кускам — на двадцать шагов это десятки тысяч вырезаний.
        occupied = np.zeros(gray.shape[:2], dtype=bool)
        near = np.zeros(gray.shape[:2], dtype=bool)
        height, width = gray.shape[:2]

        def mark(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True

        for d in dets:
            x0, y0, w, h = d.box
            mark(occupied, x0 - SPLIT_EXCLUDE_MARGIN, y0 - SPLIT_EXCLUDE_MARGIN,
                 x0 + w + SPLIT_EXCLUDE_MARGIN, y0 + h + SPLIT_EXCLUDE_MARGIN)
            # Зона, где вообще может прятаться юнит: рядом с находкой.
            mark(near, x0 - SPLIT_RADIUS, y0 - SPLIT_RADIUS,
                 x0 + w + SPLIT_RADIUS, y0 + h + SPLIT_RADIUS)
        candidates: list[tuple[float, str, int, int, int]] = []
        for name, res in responses.items():
            tpl = self.bank.get(name)
            if tpl is None:
                continue
            spots: list[tuple[int, int]] = []
            for x, y, _ in local_maxima(res, SPLIT_PEAK_FLOOR):
                # Спрятанный юнит может прятаться только рядом с уже
                # найденным: вдали от находок смотреть нечего. Без этой
                # отсечки считалось около двух тысяч оценок на один тип —
                # половина из них заведомо в пустоте, и замер вставал колом.
                if not near[y, x]:
                    continue
                spots.append((x, y))
            if SPLIT_USE_RING:
                for x, y, _ in self._ring_candidates(res, dets, tpl.size):
                    spots.append((x, y))
            for x, y in spots:
                score = self._visible_score(gray, tpl, x, y, occupied)
                if score >= SPLIT_MIN_SCORE:
                    candidates.append((score, name, x, y, tpl.size))
        # Лучшие кандидаты побеждают: сначала по оценке, потом жадно, чтобы
        # один юнит не дал несколько находок.
        candidates.sort(key=lambda item: -item[0])
        taken = [(d.cx, d.cy) for d in dets]
        extra: list[Detection] = []
        for score, name, x, y, size in candidates:
            cx = x + size / 2.0
            cy = y + size / 2.0
            if any(math.hypot(cx - ox, cy - oy) < SPLIT_MIN_SEPARATION for ox, oy in taken):
                continue
            extra.append(
                Detection(
                    name=name,
                    score=float(score),
                    cx=float(cx),
                    cy=float(cy),
                    box=(int(x), int(y), int(size), int(size)),
                )
            )
            taken.append((cx, cy))
        return extra

    def _ring_candidates(
        self,
        res: np.ndarray,
        dets: Sequence[Detection],
        size: int,
    ) -> list[tuple[int, int, float]]:
        """Лучшие точки отклика в кольце вокруг уже найденных находок.

        Внутри самого бокса находки искать нечего: там уже есть юнит.
        Берётся несколько самых сильных точек рядом, разнесённых между
        собой, — ими могут быть и спрятанные юниты.
        """
        out: list[tuple[int, int, float]] = []
        height, width = res.shape[:2]
        half = size / 2.0
        for det in dets:
            x0 = int(det.cx - half - SPLIT_RADIUS)
            x1 = int(det.cx - half + SPLIT_RADIUS) + 1
            y0 = int(det.cy - half - SPLIT_RADIUS)
            y1 = int(det.cy - half + SPLIT_RADIUS) + 1
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            win = res[y0:y1, x0:x1]
            ccx = det.cx - half - x0
            ccy = det.cy - half - y0
            yy, xx = np.mgrid[0 : win.shape[0], 0 : win.shape[1]]
            far = (xx + 0.5 - ccx) ** 2 + (yy + 0.5 - ccy) ** 2 > (SPLIT_MIN_SEPARATION + 2.0) ** 2
            masked = np.where(far, win, -1.0e9)
            flat = masked.ravel()
            count = min(flat.size, SPLIT_RING_CANDIDATES * 8)
            top = np.argpartition(-flat, count - 1)[:count]
            picked: list[tuple[int, int]] = []
            for index in top:
                value = float(flat[index])
                if value < SPLIT_PEAK_FLOOR:
                    break
                py, px = divmod(int(index), win.shape[1])
                gx, gy = x0 + px, y0 + py
                if any(abs(gx - ax) <= 6 and abs(gy - ay) <= 6 for ax, ay in picked):
                    continue
                picked.append((gx, gy))
                if len(picked) >= SPLIT_RING_CANDIDATES:
                    break
            out.extend((gx, gy, float(res[gy, gx])) for gx, gy in picked)
        return out

    def _visible_score(
        self,
        gray: np.ndarray,
        tpl: Template,
        x: int,
        y: int,
        occupied: np.ndarray,
    ) -> float:
        """Корреляция шаблона с видимой частью картинки.

        Из шаблона выбрасываются: чёрные углы тайла (это фон, а не юнит) и
        те пиксели, которые закрыты другими юнитами (``occupied`` — зоны
        перекрытия в координатах поля). Корреляция считается по тому, что
        осталось, — это и есть видимая часть юнита.
        """
        size = tpl.size
        h, w = gray.shape[:2]
        if x < 0 or y < 0 or x + size > w or y + size > h:
            return 0.0
        patch = gray[y : y + size, x : x + size]
        mask = tpl.content & ~occupied[y : y + size, x : x + size]
        if int(mask.sum()) < SPLIT_MIN_VISIBLE:
            return 0.0
        a = patch[mask].astype(np.float64)
        b = tpl.image[mask].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den <= 0.0:
            return 0.0
        return float((a * b).sum() / den)

    def _sample_color(self, hsv: np.ndarray, det: Detection) -> tuple[float, int, int]:
        """Цвет «тела» юнита.

        Глиф типа внутри юнита белый (насыщенность ~0), а тело — насыщенное.
        Поэтому берём кольцо вокруг центра, отбрасываем ненасыщенные пиксели
        и усредняем оттенок по оставшимся.
        """
        x, y, w, h = det.box
        patch = hsv[y : y + h, x : x + w]
        if patch.size == 0:
            return 0.0, 0, 0

        # Кольцо вокруг центра: внутренний край задаёт patch_inset
        # (доля от полуразмера бокса), внешний — 90%, чтобы не лезть в фон.
        cy, cx = h / 2.0, w / 2.0
        half = min(w, h) / 2.0
        inner = max(0.05, min(0.8, self.vision.patch_inset))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
        ring = (dist >= half * inner) & (dist <= half * 0.9)
        sel = patch[ring]
        if sel.size == 0:
            sel = patch.reshape(-1, 3)

        sat = sel[:, 1]
        keep = sat >= 80
        if keep.sum() >= 4:
            sel = sel[keep]
            sat = sel[:, 1]
        hues = sel[:, 0].astype(np.float32)
        vals = sel[:, 2]

        # Hue круговой: усредняем по вектору, чтобы не ломать переход через 0.
        rad = np.deg2rad(hues * 2.0)
        mean_cos = float(np.mean(np.cos(rad)))
        mean_sin = float(np.mean(np.sin(rad)))
        mean_hue = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) / 2.0) % 180.0
        return mean_hue, int(np.median(sat)), int(np.median(vals))

    def classify_team(self, hue: float, sat: int, val: int) -> Team | None:
        """Определить сторону по цвету."""
        cfg: FactionConfig = self.vision.faction
        if sat < cfg.min_saturation or val < cfg.min_value:
            return None
        if _hue_in_ranges(hue, cfg.ally_hue_ranges):
            return Team.ALLY
        if _hue_in_ranges(hue, cfg.enemy_hue_ranges):
            return Team.ENEMY
        return None

    # ------------------------------------------------------------------- mana
    def detect_available(self, frame: np.ndarray) -> tuple[str, ...]:
        """Определить, какие юниты доступны к спавну, по подсвеченным карточкам.

        Работает и в симуляторе, и на реальном экране: карточка считается
        доступной, если внутри неё есть насыщенные пиксели (цветная иконка).
        """
        available: list[str] = []
        for name, rect in self.vision.spawn_cards.items():
            x, y, w, h = (int(rect.get("x", 0)), int(rect.get("y", 0)),
                          int(rect.get("w", 0)), int(rect.get("h", 0)))
            if w <= 0 or h <= 0:
                continue
            h_img, w_img = frame.shape[:2]
            x1, y1 = min(x + w, w_img), min(y + h, h_img)
            patch = frame[y:y1, x:x1]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            frac_sat = float((hsv[:, :, 1] > self.vision.card_min_saturation).mean())
            if frac_sat >= self.vision.card_min_saturated_fraction:
                available.append(name)
        return tuple(available)

    def read_mana(self, frame: np.ndarray) -> tuple[float, float]:
        """Оценить ману по заполнению полоски HUD (координаты в системе кадра).

        Полоска маны находится вне игрового поля, поэтому читается напрямую
        из исходного кадра, а не из варпнутого.
        """
        bar = self.vision.mana_bar
        if not bar:
            return 0.0, 0.0
        x, y = int(bar.get("x", 0)), int(bar.get("y", 0))
        w, h = int(bar.get("w", 0)), int(bar.get("h", 0))
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        h_img, w_img = frame.shape[:2]
        x1, y1 = min(x + w, w_img), min(y + h, h_img)
        patch = frame[y : y1, x : x1]
        if patch.size == 0:
            return 0.0, 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        min_sat = int(self.vision.mana_min_saturation)
        mask = sat > min_sat
        if not mask.any():
            return 0.0, 0.0

        # Заливка идёт слева направо. Рамка полоски может перекрывать первый
        # пиксель, поэтому ищем непрерывный отрезок насыщенных колонок,
        # начиная с первой найденной, а не строго от нуля.
        cols = mask.any(axis=0)
        indices = np.flatnonzero(cols)
        first = int(indices[0])
        last = first
        for i in range(first, len(cols)):
            if not cols[i]:
                break
            last = i
        # Внутренняя ширина полоски (без рамки).
        inner_width = max(1, w - 2)
        filled = last - first + 1
        fraction = min(1.0, filled / inner_width)
        return fraction * self.vision.mana_max, fraction

    # ------------------------------------------------------------- assembling
    def build_state(
        self,
        board: np.ndarray,
        dets: list[Detection] | None = None,
        mana: float | None = None,
        mana_conf: float | None = None,
        spawn_points: dict[str, Lane] | None = None,
        available: Sequence[str] | None = None,
        source: str = "live",
    ) -> GameState:
        """Собрать GameState из канонического кадра поля."""
        self.ensure_loaded()
        dets = dets if dets is not None else self.detect_units(board)
        if mana is None or mana_conf is None:
            mana_val, mana_conf_val = self.read_mana(board)
            mana = mana_val if mana is None else mana
            mana_conf = mana_conf_val if mana_conf is None else mana_conf

        state = GameState(
            mana=mana,
            mana_max=self.vision.mana_max,
            mana_confidence=mana_conf,
            detections=dets,
            spawn_points=spawn_points or {},
            available_units=tuple(available) if available is not None else None,
            source=source,
        )
        for det in dets:
            spec = self.specs.get(det.name)
            if spec is None:
                continue
            # Сторона определяется по цвету, и цвет может не определиться
            # (юнит на границе экрана, засветка, незнакомый оттенок). Таких
            # юнитов считаем вражескими: переоценить врага безопаснее, чем
            # потерять его из плана (иначе бот не заметит прорыва).
            team = det.team or Team.ENEMY
            hp = spec.hp
            unit = Unit(
                uid=self._uid_for(det),
                kind=det.name,
                team=team,
                x=det.cx,
                y=det.cy,
                hp=hp,
                max_hp=hp,
                spec=spec,
            )
            if team is Team.ALLY:
                state.ally_units.append(unit)
            else:
                state.enemy_units.append(unit)
        return state

    def _uid_for(self, det: Detection) -> int:
        """Устойчивый идентификатор юнита.

        Раньше здесь был ``hash()`` строки, а он в Python рандомизируется при
        каждом запуске процесса (PYTHONHASHSEED). Один и тот же юнит получал
        разные uid в разных запусках, и сопоставить их между кадрами или
        между сессиями было нельзя. Считаем crc32 от имени типа — он
        стабилен между процессами и версиями.
        """
        x, y, w, h = det.box
        name_hash = zlib.crc32(det.name.encode("utf-8"))
        return int((name_hash ^ (x * 7919 + y * 104729 + w * 31 + h * 17)) & 0x7FFFFFFF)

    def analyze(
        self,
        frame: np.ndarray,
        cal: Calibration,
        spawn_points: dict[str, Lane] | None = None,
        available: Sequence[str] | None = None,
    ) -> tuple[GameState, np.ndarray]:
        """Основная точка входа: экранный кадр -> (состояние, канонический кадр)."""
        board = warp_to_board(frame, cal)
        dets = self.detect_units(board)
        mana, mana_conf = self.read_mana(frame)
        state = self.build_state(
            board,
            dets=dets,
            mana=mana,
            mana_conf=mana_conf,
            spawn_points=spawn_points,
            available=available,
        )
        return state, board
