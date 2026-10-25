"""Распознавание числовых показателей HUD (энергия, таймер).

В веб-клиенте Rush Royale энергия выводится числом поверх декоративной
шкалы, поэтому её нужно читать как текст. Подход: бинаризация по яркости,
разбиение на глифы по колонкам и сопоставление с эталонами.

Эталоны лежат в ``assets/digits/<цифра>.png``. Новые цифры добавляются
инструментом ``tools/label_digits.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DIGIT_DIR = Path(__file__).resolve().parent.parent / "assets" / "digits"
# Минимальная ширина цифры в пикселях: группы уже — часть цифры.
MIN_DIGIT_WIDTH = 7
# Максимальная ширина одной цифры: шире — значит, цифры слиплись.
MAX_DIGIT_WIDTH = 14
# Размер эталона цифры (нормализованный).
TEMPLATE_W = 8
TEMPLATE_H = 14
# Минимальная уверенность, при которой глиф считается опознанной цифрой.
# Замеры на игровых кадрах: верные совпадения дают 0.34-0.46, ошибки — 0.25-0.28.
MIN_DIGIT_CONFIDENCE = 0.75


@dataclass
class DigitReading:
    """Результат чтения числа."""

    text: str
    value: float
    confidence: float

    @property
    def ok(self) -> bool:
        return bool(self.text) and "?" not in self.text


class DigitReader:
    """Чтение чисел с экрана по эталонам цифр."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else DIGIT_DIR
        self.templates: dict[str, np.ndarray] = {}
        self.reload()

    def reload(self) -> None:
        self.templates.clear()
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Имя файла может быть «3.png» или «3_1.png» (несколько образцов
            # одной цифры) — цифра всегда в начале имени.
            digit = path.stem.split("_")[0]
            self.templates.setdefault(digit, []).append(img)

    @property
    def known(self) -> set[str]:
        return set(self.templates.keys())

    # ------------------------------------------------------------- разбор
    def _glyphs(self, region: np.ndarray) -> list[np.ndarray]:
        """Выделить глифы в области изображения."""
        if region.size == 0:
            return []
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV) if region.ndim == 3 else None
        if hsv is not None:
            # Цифры — светлые (высокая яркость) на тёмном фоне.
            mask = ((gray > 150) & (hsv[:, :, 1] < 120)).astype(np.uint8) * 255
        else:
            mask = (gray > 150).astype(np.uint8) * 255
        if mask.sum() == 0:
            return []
        cols = mask.any(axis=0)
        # Первичные группы — строго непрерывные колонки.
        raw: list[list[int]] = []
        for i, present in enumerate(cols):
            if not present:
                continue
            if raw and i == raw[-1][-1] + 1:
                raw[-1].append(i)
            else:
                raw.append([i])
        # Узкая группа — это часть цифры (например, «1» из флагштока и
        # ствола), поэтому присоединяем её к соседней.
        groups: list[list[int]] = []
        for g in raw:
            if groups and (len(groups[-1]) < MIN_DIGIT_WIDTH or len(g) < MIN_DIGIT_WIDTH):
                groups[-1].extend(g)
            else:
                groups.append(list(g))
        glyphs: list[np.ndarray] = []
        for g in groups:
            x0, x1 = g[0], g[-1]
            if x1 - x0 < 2:
                continue
            sub = mask[:, x0 : x1 + 1]
            rows = sub.any(axis=1)
            ys = np.flatnonzero(rows)
            if len(ys) == 0:
                continue
            y0, y1 = int(ys[0]), int(ys[-1])
            crop = sub[y0 : y1 + 1]
            glyphs.extend(self._split_if_needed(crop))
        return glyphs

    def _split_if_needed(self, crop: np.ndarray) -> list[np.ndarray]:
        """Разделить слипшиеся цифры.

        Между цифрами в шрифте игры бывают полностью пустые колонки — по
        ним и режем («690» — три цифры одним куском). Разрез подтверждаем
        шириной частей: настоящая цифра шире 8 px, а дырка внутри цифры
        дала бы слишком узкий кусок.
        """
        width = crop.shape[1]
        if width <= MAX_DIGIT_WIDTH:
            return [self._normalize(crop)]

        profile = crop.sum(axis=0) / 255.0
        cuts = self._find_gaps(profile)
        if not cuts:
            return [self._normalize(crop)]
        parts: list[np.ndarray] = []
        start = 0
        for cut in cuts:
            if cut - start >= 8 and width - cut >= 8:
                parts.append(self._normalize(crop[:, start:cut]))
                start = cut
        if width - start >= 8:
            parts.append(self._normalize(crop[:, start:]))
        return parts or [self._normalize(crop)]

    @staticmethod
    def _find_gaps(profile: np.ndarray) -> list[int]:
        """Колонки с разрывом между цифрами.

        Разрыв — это пустые или почти пустые колонки: у «9» и «0», стоящих
        вплотную, внизу остаётся пара пикселей контура.
        """
        if profile.size == 0:
            return []
        level = max(0.5, float(profile.max()) * 0.18)
        cuts: list[int] = []
        index = 0
        while index < profile.size:
            if profile[index] > level:
                index += 1
                continue
            start = index
            while index < profile.size and profile[index] <= level:
                index += 1
            cuts.append((start + index) // 2)
        return cuts

    @staticmethod
    def _normalize(crop: np.ndarray) -> np.ndarray:
        """Привести глиф к общему виду: вписать с сохранением пропорций.

        «1» узкая, «8» широкая: без нормализации по ink-боксу сжатие
        искажает начертание и цифры начинают путаться.
        """
        h, w = crop.shape
        scale = min(TEMPLATE_W / max(w, 1), TEMPLATE_H / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.uint8)
        ox = (TEMPLATE_W - new_w) // 2
        oy = (TEMPLATE_H - new_h) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = resized
        return canvas

    def _match(self, glyph: np.ndarray) -> tuple[str, float]:
        """Сопоставить глиф с эталонами цифр.

        Используется нормированная корреляция: она даёт 1.0 для одинаковых
        глифов независимо от сглаживания, в отличие от поэлементного
        произведения, где результат зависит от контраста.
        """
        if not self.templates:
            return "?", 0.0
        best, best_score = "?", -1.0
        for name, samples in self.templates.items():
            for tpl in samples:
                score = self._correlate(glyph, tpl)
                if score > best_score:
                    best_score, best = score, name
        confidence = max(0.0, best_score)
        # Если совпадение слабое, цифра не опознана: честнее сказать «?», чем
        # выдать за известную цифру чужую.
        if confidence < MIN_DIGIT_CONFIDENCE:
            return "?", confidence
        return best, confidence

    @staticmethod
    def _correlate(a: np.ndarray, b: np.ndarray) -> float:
        """Нормированная корреляция двух глифов."""
        x = a.astype(np.float64).ravel()
        y = b.astype(np.float64).ravel()
        if x.size != y.size or x.size == 0:
            return 0.0
        x -= x.mean()
        y -= y.mean()
        denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
        if denom <= 1e-9:
            return 0.0
        return float((x * y).sum() / denom)

    def read(self, region: np.ndarray) -> DigitReading:
        """Прочитать число из области изображения."""
        glyphs = self._glyphs(region)
        if not glyphs:
            return DigitReading("", 0.0, 0.0)
        chars: list[str] = []
        scores: list[float] = []
        for g in glyphs:
            ch, score = self._match(g)
            chars.append(ch)
            scores.append(score)
        text = "".join(chars)
        confidence = sum(scores) / len(scores) if scores else 0.0
        try:
            value = float(text)
        except ValueError:
            value = 0.0
        return DigitReading(text, value, confidence)

    def learn(self, glyph: np.ndarray, digit: str) -> Path:
        """Сохранить эталон цифры.

        Несколько образцов на цифру — нормализация не убирает различия
        в начертании, поэтому берём лучшее совпадение из всех.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = self.templates.get(digit, [])
        index = len(existing)
        path = self.directory / (f"{digit}.png" if index == 0 else f"{digit}_{index}.png")
        cv2.imwrite(str(path), glyph)
        self.templates.setdefault(digit, []).append(glyph)
        return path
