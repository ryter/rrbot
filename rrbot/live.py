"""Работа с живой игрой Rush Royale в браузере.

Модуль знает раскладку веб-клиента: окно браузера, игровой холст внутри
него, полосу маны и ряд карточек юнитов. Элементы интерфейса находятся по
цвету, а не по шаблонам: кнопки плоские и однотонные, поэтому цвет надёжнее
сопоставления с картинкой.

Координаты внутри модуля — в системе холста игры (0..492 по X, 0..804 по Y).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .digits import DigitReader

log = logging.getLogger(__name__)

# Раскладка веб-клиента (измерена через DevTools Protocol).
CANVAS_W = 492
CANVAS_H = 804
# Смещение холста внутри окна браузера.
CANVAS_IN_WINDOW = (225, 139)
# Размер окна браузера по умолчанию.
WINDOW_SIZE = (941, 1030)

@dataclass(frozen=True)
class ButtonRule:
    """Правило поиска кнопки по цвету и размеру."""

    name: str
    hue_ranges: tuple[tuple[int, int], ...]
    sat: tuple[int, int]
    val: tuple[int, int]
    min_area: int
    min_w: int
    min_h: int
    max_w: int
    max_h: int


# Правила поиска кнопок.
# H задаётся списком пар, потому что красный в шкале OpenCV (0..180)
# разрывается: крестик закрытия имеет оттенок ~171, а бейдж со скидкой ~5.
# Размеры важны: рядом с крестиком стоит крупный красный бейдж, и по одному
# цвету их не различить.
#
# «Отмена» на экране подбора соперника — красно-оранжевая (H≈4-9), крупная
# (117x48). Её не брал ни один прежний диапазон: чистый красный ловил
# только мелкие значки, а оранжевый — кнопку PvP.
BUTTON_RULES = (
    ButtonRule("close", ((0, 10), (165, 180)), (120, 255), (110, 255), 60, 20, 20, 48, 48),
    ButtonRule("cancel", ((0, 12), (168, 180)), (150, 255), (180, 255), 2000, 60, 30, 220, 80),
    ButtonRule("continue", ((100, 132),), (130, 255), (120, 255), 1200, 40, 22, 400, 260),
    ButtonRule("pvp", ((12, 26),), (140, 255), (140, 255), 1500, 40, 22, 300, 300),
    ButtonRule("fight", ((14, 34),), (150, 255), (150, 255), 800, 40, 22, 300, 300),
    ButtonRule("watch", ((140, 168),), (130, 255), (130, 255), 1200, 40, 22, 300, 300),
)

# Диапазоны HUD (координаты холста).
# Полоса маны — зелёная шкала рядом с числом энергии.
MANA_BAR_HINT = (112, 670, 70, 26)
# Ряд карточек юнитов в нижней части экрана.
CARD_ROW_Y = (712, 790)
CARD_COUNT = 5
CARD_SLOT_WIDTH = 63
CARD_FIRST_X = 30
# Игровое поле в координатах холста. В веб-клиенте арена разделена:
# сверху половина противника, снизу своя, между ними полоса с волной.
# Детектор работает по обеим половинам сразу.
FIELD_QUAD = ((66.0, 52.0), (400.0, 52.0), (400.0, 664.0), (66.0, 664.0))
# Канонический размер поля — в нём работает детектор.
FIELD_W = 334
FIELD_H = 612
# Максимальная энергия в игре (для перевода доли шкалы в число).
MANA_MAX = 500.0
# Область с числом энергии (цифры поверх декоративной шкалы).
ENERGY_REGION = (138, 682, 38, 16)
# Голубой флакон энергии слева от числа — есть только в бою.
ENERGY_FLASK = (88, 672, 40, 36)
# Кнопка призыва: карточка в оранжевом виджете с значком маны, который
# стоит над рядом карточек в нижней части экрана. Нажатие тратит ману и
# призывает юнита — проверено на живом бою (своих юнитов стало на одного
# больше, энергия упала на стоимость).
SUMMON_BUTTON = (228.0, 685.0)
# Область, где ищем оранжевый виджет маны.
SUMMON_SEARCH = (170, 650, 300, 715)
# Порог совпадения с шаблоном кнопки.
SUMMON_MIN_SCORE = 0.55
# Каталог эталонов символов валюты (€ и прочие).
CURRENCY_DIR = Path(__file__).resolve().parent.parent / "assets" / "ui"
# Каталог эталонов надписей (assets/ui/text_*.png): «PvP», «Co-Op».
TEXT_DIR = CURRENCY_DIR
# Надписи «PvP» и «Co-Op» стоят в меню, в ряду кнопок у нижнего края.
# Ищем только ниже этой отметки (доля высоты холста): так поиск быстрее
# и меньше шансов сматчить надпись на постороннем экране.
TEXT_SEARCH_TOP = 0.40
# Порог совпадения с эталоном надписи. На настоящих кадрах верное
# совпадение даёт 0.74-0.79, а на посторонних экранах — не больше 0.46.
TEXT_MIN_SCORE = 0.62

# Кнопка «Co-Op» в главном меню. Фон меню — тёмно-синий (H≈106, V≈110),
# а сама кнопка яркая (V≈245). Обычный диапазон «continue» (V>=120) сливает
# кнопку с фоном в один кусок, и найти её не удаётся: по яркости кнопка от
# фона отделяется.
COOP_BUTTON_RULE = ButtonRule(
    "coop", ((100, 128),), (140, 255), (185, 255), 1500, 60, 30, 320, 130
)

# Режимы подбора матча: «auto» — сначала PvP, при неудаче Co-Op.
MATCH_MODES = ("auto", "pvp", "coop")
# Сколько раз подряд сервер не нашёл соперника, прежде чем перейти на Co-Op.
PVP_WAITS_BEFORE_COOP = 2


@dataclass
class GameWindow:
    """Положение окна браузера с игрой на экране."""

    x: int = 967
    y: int = 38
    width: int = WINDOW_SIZE[0]
    height: int = WINDOW_SIZE[1]

    @property
    def region(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def detect(
        cls, title_hint: str = "chrome", game_title: str = "rush royale"
    ) -> "GameWindow":
        """Найти окно Chrome с игрой через Hyprland.

        Окно может быть перемещено пользователем в любой момент, поэтому
        положение уточняется перед каждой синхронизацией, а не один раз.
        """
        try:
            out = subprocess.run(
                ["hyprctl", "clients", "-j"], capture_output=True, timeout=5, check=False
            ).stdout
            clients = json.loads(out or b"[]")
        except Exception as exc:  # pragma: no cover
            log.debug("hyprctl не отработал: %s", exc)
            return cls()
        fallback: "GameWindow | None" = None
        for c in clients:
            if title_hint.lower() not in (c.get("class") or "").lower():
                continue
            at = c.get("at", [0, 0])
            size = c.get("size", [0, 0])
            window = cls(int(at[0]), int(at[1]), int(size[0]), int(size[1]))
            if game_title in (c.get("title") or "").lower():
                return window
            fallback = fallback or window
        return fallback or cls()


@dataclass
class UiElement:
    """Найденный элемент интерфейса в координатах холста."""

    kind: str
    x: int
    y: int
    w: int
    h: int
    # Найден по надписи, а не по цвету. Такую кнопку нельзя отбросить
    # проверкой «покупка»: оранжевая кнопка «PvP» по цвету не отличается
    # от золотой кнопки магазина, а по буквам отличается однозначно.
    by_text: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def screen_changed(
    before: np.ndarray | None, after: np.ndarray | None, threshold: float = 1.5
) -> bool:
    """Изменился ли экран между двумя кадрами.

    Помогает понять, что клик куда-то попал. Начать матч по нему нельзя:
    экран подбора соперника анимируется, и «изменилось» истинно всегда.
    """
    if before is None or after is None:
        return True
    if before.shape != after.shape:
        return True
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return float(diff.mean()) > threshold


def wait_for_battle(
    grab,
    is_battle,
    timeout: float = 30.0,
    interval: float = 1.0,
    sleep=time.sleep,
    clock=time.monotonic,
) -> bool:
    """Ждать начала боя после нажатия «PvP» или «Co-Op».

    Подбор соперника идёт по сети и может занять полминуты. Пока он идёт,
    экран анимируется, поэтому судить по кадру «изменилось» бесполезно:
    признаком начала матча служит только сам бой.

    ``grab`` — функция, возвращающая кадр (или ``None``), ``is_battle`` —
    проверка «идёт ли бой» по кадру. Обе передаются снаружи, поэтому
    логика проверяется без живой игры.
    """
    deadline = clock() + timeout
    while True:
        frame = grab()
        if frame is not None and is_battle(frame):
            return True
        now = clock()
        if now >= deadline:
            return False
        sleep(min(interval, max(0.0, deadline - now)))


@dataclass
class Matchmaking:
    """Как искать матч: PvP, Co-Op или «сначала PvP, потом Co-Op».

    Сервер PvP часто не находит соперников: подбор висит, а боя так и нет.
    Поэтому считаем неудачные попытки подбора и после двух подряд
    переключаемся на Co-Op.

    Попытка — это нажатие «PvP»/«Co-Op» и её исход. Неудача попытки
    считается ровно один раз: и «истекло время», и «нажали Отмена»
    относятся к одной и той же попытке.
    """

    mode: str = "auto"
    # Неудачные попытки подбора подряд с последнего успешного матча.
    pvp_waits: int = 0
    # Сколько раз за забег переключались с PvP на Co-Op.
    pvp_fails: int = 0
    # После скольких неудачных попыток переключаться.
    waits_before_coop: int = PVP_WAITS_BEFORE_COOP
    # Идёт ли сейчас попытка подбора.
    _attempt: bool = False
    # Посчитана ли уже неудача этой попытки.
    _counted: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MATCH_MODES:
            raise ValueError(
                f"неизвестный режим подбора: {self.mode} (доступны: {', '.join(MATCH_MODES)})"
            )

    @property
    def use_coop(self) -> bool:
        """Искать Co-Op вместо PvP."""
        if self.mode == "coop":
            return True
        return self.mode == "auto" and self.pvp_fails > 0

    @property
    def mode_name(self) -> str:
        """Что сейчас ищем: «PvP» или «Co-Op»."""
        return "Co-Op" if self.use_coop else "PvP"

    def begin_attempt(self) -> None:
        """Началась попытка подбора: нажали «PvP» или «Co-Op»."""
        self._attempt = True
        self._counted = False

    def on_match_started(self) -> None:
        """Матч начался — счётчик неудачных попыток обнуляется."""
        self._attempt = False
        self._counted = False
        self.pvp_waits = 0

    def on_search_failed(self) -> bool:
        """Попытка не дала матча: истекло время ожидания.

        Возвращает ``True``, если пора переключиться на Co-Op.
        """
        if not self._attempt or self._counted:
            return False
        self._counted = True
        self.pvp_waits += 1
        if self.mode != "auto" or self.pvp_waits < self.waits_before_coop:
            return False
        self.pvp_waits = 0
        self.pvp_fails += 1
        return True

    def on_search_cancelled(self) -> bool:
        """Подбор отменён: нажали «Отмена» или закрыли «соперник не найден».

        Это конец той же попытки, что и ожидание по таймауту, поэтому
        неудача не считается второй раз.
        """
        if not self._attempt:
            return False
        failed = self.on_search_failed()
        self._attempt = False
        return failed


def find_canvas_rect(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Прямоугольник холста игры в снимке окна браузера.

    У холста есть тёмно-синяя рамка (в шкале OpenCV H≈106, S≈210) — по
    ней и находится область игры. Раньше для этого спрашивали страницу
    через DevTools Protocol, но тогда требовался Chrome с отладочным
    портом; теперь достаточно снимка экрана.

    Рамка ищется по профилям: у неё есть горизонтальные полосы (сверху и
    снизу) и вертикальные (слева и справа). Это устойчиво к затемнению
    экрана (на модальных окнах рамка почти чернеет, но оттенок остаётся)
    и к тому, что окно разрывает кольцо рамки на части.

    Возвращает ``(x, y, w, h)`` в координатах кадра или ``None``.
    """
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (98, 150, 10), (116, 255, 255))
    height, width = mask.shape
    rows = mask.sum(axis=1) / 255.0
    cols = mask.sum(axis=0) / 255.0
    row_max = float(rows.max())
    col_max = float(cols.max())
    # Полосы рамки: широкая по горизонтали и заметная по вертикали.
    if row_max < width * 0.4 or col_max < height * 0.15:
        return None
    row_thr = row_max * 0.8
    col_thr = col_max * 0.3
    top = next((y for y in range(height) if rows[y] >= row_thr), None)
    bottom = next((y for y in range(height - 1, -1, -1) if rows[y] >= row_thr), None)
    left = next((x for x in range(width) if cols[x] >= col_thr), None)
    right = next((x for x in range(width - 1, -1, -1) if cols[x] >= col_thr), None)
    if None in (top, bottom, left, right):
        return None
    if bottom - top < height * 0.5 or right - left < width * 0.3:
        return None
    return (left, top, right - left + 1, bottom - top + 1)


class LiveGame:
    """Экранная обёртка над живой игрой."""

    def __init__(self, window: GameWindow | None = None):
        self.window = window or GameWindow.detect()
        self.last_canvas: np.ndarray | None = None
        self.digits = DigitReader()
        # Прямоугольник холста внутри окна: обновляется по рамке на снимке.
        self.canvas_rect = CANVAS_IN_WINDOW
        # Положение холста на экране (в координатах экрана).
        self.canvas_offset = (
            self.window.x + CANVAS_IN_WINDOW[0],
            self.window.y + CANVAS_IN_WINDOW[1],
        )
        self.canvas_size = (CANVAS_W, CANVAS_H)
        # Положение холста в координатах страницы (CSS-пиксели) — от него
        # считаются координаты кликов DevTools Protocol.
        self.canvas_page = (0, 0)
        self._summon_template = None
        self._currency_templates = self._load_currency_templates()
        self._text_templates = self._load_text_templates()

    # --------------------------------------------------- кнопки покупки
    def _load_currency_templates(self) -> list[np.ndarray]:
        """Эталоны символов валюты (€ и прочие)."""
        templates: list[np.ndarray] = []
        if not CURRENCY_DIR.exists():
            return templates
        for path in sorted(CURRENCY_DIR.glob("currency_*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size:
                templates.append(img)
        return templates

    def _load_text_templates(self) -> dict[str, np.ndarray]:
        """Эталоны надписей: «PvP», «Co-Op» и прочие (assets/ui/text_*.png)."""
        templates: dict[str, np.ndarray] = {}
        if not TEXT_DIR.exists():
            return templates
        for path in sorted(TEXT_DIR.glob("text_*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size:
                templates[path.stem[len("text_") :]] = img
        return templates

    def _is_solid_red(self, canvas: np.ndarray, element: "UiElement") -> bool:
        """Круглый красный значок, а не бейдж с белым текстом.

        У крестика закрытия окна красного много (0.55-0.60), у бейджа
        скидки «-90%» — почти нет: там белые цифры. По этой доле и
        отличаем одно от другого.
        """
        x, y, w, h = element.x, element.y, element.w, element.h
        region = canvas[max(0, y) : y + h, max(0, x) : x + w]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 120, 110), (10, 255, 255)) | cv2.inRange(
            hsv, (165, 120, 110), (180, 255, 255)
        )
        return float(red.mean()) / 255 >= 0.35

    def is_purchase(self, canvas: np.ndarray, element: "UiElement") -> bool:
        """Кнопка покупки — по ней бот не кликает никогда.

        Признаки два: кнопка золотая (в игре так выглядят все покупки) и
        на ней нарисован символ валюты. Эталон символа снят с настоящей
        кнопки «4,99 €».
        """
        x, y, w, h = element.x, element.y, element.w, element.h
        region = canvas[max(0, y) : y + h, max(0, x) : x + w]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, (18, 170, 150), (40, 255, 255))
        # Кнопка покупки — крупная золотая площадка. Мелкие золотые значки
        # (монетки в иконках) покупкой не считаются.
        n, _, stats, _ = cv2.connectedComponentsWithStats(gold, 8)
        for i in range(1, n):
            gx, gy, gw, gh, area = stats[i]
            if gw >= 60 and gh >= 25 and area >= 1200:
                return True
        if not self._currency_templates:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        for tpl in self._currency_templates:
            if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            if res.size and float(res.max()) > 0.55:
                return True
        return False

    # ------------------------------------------------------------- геометрия
    def sync_geometry(self, session=None) -> tuple[int, int]:
        """Уточнить положение окна и холста игры.

        С DevTools Protocol координаты холста берутся прямо у страницы —
        это точнее, чем искать рамку на экране. Без сессии (или если
        запрос не удался) остаётся запасной путь: холст ищется по рамке
        в снимке окна (см. :func:`find_canvas_rect`).
        """
        self.window = GameWindow.detect()
        if session is not None:
            rect = self._canvas_rect_from_page(session)
            if rect is not None:
                self.set_canvas_rect(rect)
                return self.canvas_offset
        return self.canvas_offset

    def _canvas_rect_from_page(self, session) -> tuple[int, int, int, int] | None:
        """Прямоугольник холста в координатах окна (по данным страницы)."""
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "tools"))
            from cdp import evaluate  # type: ignore[import-not-found]

            frame_rect = evaluate(
                session,
                """(() => {
                    const f = Array.from(document.querySelectorAll('iframe'))
                        .find(x => /rush\\.my\\.games/.test(x.src || ''));
                    if (!f) return null;
                    const r = f.getBoundingClientRect();
                    return {x: r.left, y: r.top, w: r.width, h: r.height,
                            ow: window.outerWidth, iw: window.innerWidth,
                            oh: window.outerHeight, ih: window.innerHeight,
                            dpr: window.devicePixelRatio};
                })()""",
            )
            if not isinstance(frame_rect, dict):
                return None
            canvas_rect = self._canvas_in_frame(session)
            if canvas_rect is None:
                return None
            dpr = float(frame_rect.get("dpr") or 1)
            border_top = float(frame_rect.get("oh", 0) - frame_rect.get("ih", 0))
            border_left = float(frame_rect.get("ow", 0) - frame_rect.get("iw", 0)) / 2
            page_x = float(frame_rect.get("x", 0)) + canvas_rect[0]
            page_y = float(frame_rect.get("y", 0)) + canvas_rect[1]
            # Экранные координаты холста.
            screen_x = self.window.x + border_left + page_x * dpr
            screen_y = self.window.y + border_top + page_y * dpr
            self.canvas_page = (page_x, page_y)
            width = int(round(canvas_rect[2] * dpr))
            height = int(round(canvas_rect[3] * dpr))
            # Прямоугольник в координатах окна.
            return (
                int(round(screen_x - self.window.x)),
                int(round(screen_y - self.window.y)),
                width,
                height,
            )
        except Exception as exc:  # pragma: no cover
            log.debug("не удалось уточнить геометрию через CDP: %s", exc)
            return None

    def _canvas_in_frame(self, session) -> tuple[float, float, float, float] | None:
        """Прямоугольник canvas внутри игрового фрейма (через DevTools)."""
        tree = session.send("Page.getFrameTree", {}).get("result", {}).get("frameTree", {})

        def find(node) -> str | None:
            frame = node.get("frame", {})
            if "rush.my.games" in (frame.get("url") or ""):
                return frame.get("id")
            for child in node.get("childFrames", []) or []:
                found = find(child)
                if found:
                    return found
            return None

        frame_id = find(tree)
        if not frame_id:
            return None
        ctx = session.send(
            "Page.createIsolatedWorld",
            {"frameId": frame_id, "worldName": "rrbot-geom", "grantUniveralAccess": True},
        ).get("result", {})
        context_id = ctx.get("executionContextId")
        if context_id is None:
            return None
        result = session.send(
            "Runtime.evaluate",
            {
                "expression": (
                    "(() => { const c = document.querySelector('canvas');"
                    " if (!c) return null; const r = c.getBoundingClientRect();"
                    " return {x: r.left, y: r.top, w: r.width, h: r.height}; })()"
                ),
                "contextId": context_id,
                "returnByValue": True,
            },
        )
        value = result.get("result", {}).get("result", {}).get("value")
        if not isinstance(value, dict):
            return None
        return (
            float(value.get("x", 0)),
            float(value.get("y", 0)),
            float(value.get("w", 0)),
            float(value.get("h", 0)),
        )

    def to_canvas(self, sx: int, sy: int) -> tuple[int, int]:
        """Экранные координаты -> координаты холста."""
        ox, oy = self.canvas_offset
        w, h = getattr(self, "canvas_size", (CANVAS_W, CANVAS_H))
        return (
            int((sx - ox) * CANVAS_W / max(w, 1)),
            int((sy - oy) * CANVAS_H / max(h, 1)),
        )

    def to_screen(self, cx: int, cy: int) -> tuple[int, int]:
        """Координаты холста -> экранные.

        ``canvas_offset`` уже хранится в экранных координатах.
        """
        ox, oy = self.canvas_offset
        w, h = getattr(self, "canvas_size", (CANVAS_W, CANVAS_H))
        return (
            int(ox + cx * w / CANVAS_W),
            int(oy + cy * h / CANVAS_H),
        )

    def to_viewport(self, cx: int, cy: int) -> tuple[float, float]:
        """Координаты холста -> координаты viewport страницы (для CDP)."""
        px, py = self.canvas_page
        return (px + cx, py + cy)

    def click_screen(self, session, sx: int, sy: int) -> tuple[int, int]:
        """Кликнуть по экранным координатам через DevTools Protocol.

        Перевод делается через холст: экран -> холст -> страница. Раньше
        экранные координаты подставлялись в координаты страницы напрямую,
        и клик уходил мимо холста.
        """
        vx, vy = self.to_viewport(*self.to_canvas(sx, sy))
        session.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": vx, "y": vy})
        time.sleep(0.15)
        session.send(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": vx, "y": vy, "button": "left", "clickCount": 1, "buttons": 1},
        )
        time.sleep(0.08)
        session.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": vx, "y": vy, "button": "left", "clickCount": 1, "buttons": 0},
        )
        return (sx, sy)

    def element_to_screen(self, element: "UiElement") -> tuple[int, int]:
        """Центр элемента интерфейса в экранных координатах."""
        return self.to_screen(*element.center)

    # --------------------------------------------------------------- захват
    def grab_canvas(self, source=None) -> np.ndarray | None:
        """Снять игровой холст в его каноническом разрешении 492x804.

        Область холста ищется по рамке в снимке окна: у холста есть
        тёмно-синяя рамка, по ней видно, где кончается игра и начинается
        страница. Положение окна уточняется у композитора, поэтому окно
        можно перемещать.
        """
        if source is None:
            from .capture import GrimSource

            source = GrimSource(self.window.region)
        else:
            source = self._ensure_source_covers(source)
        frame = source.grab()
        if frame is None:
            return None
        rect = find_canvas_rect(frame)
        if rect is not None:
            self.set_canvas_rect(rect)
        x, y, w, h = self.canvas_rect
        canvas = frame[y : y + h, x : x + w]
        if canvas.size == 0:
            return None
        if canvas.shape[0] != CANVAS_H or canvas.shape[1] != CANVAS_W:
            canvas = cv2.resize(canvas, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        self.last_canvas = canvas
        return canvas

    def set_canvas_rect(self, rect: tuple[int, int, int, int]) -> None:
        """Запомнить прямоугольник холста (в координатах окна)."""
        x, y, w, h = rect
        self.canvas_rect = (int(x), int(y), int(w), int(h))
        self.canvas_size = (int(w), int(h))
        self.canvas_offset = (self.window.x + int(x), self.window.y + int(y))

    def _ensure_source_covers(self, source):
        """Пересоздать источник, если он не покрывает область холста.

        Окно игры может развернуться на весь экран, тогда источник,
        привязанный к прежней области, снимал бы не то место.
        """
        region = getattr(source, "region", None)
        if region is None:
            return source
        sx, sy, sw, sh = region
        ox, oy = self.canvas_offset
        w, h = getattr(self, "canvas_size", (CANVAS_W, CANVAS_H))
        covers = sx <= ox and sy <= oy and sx + sw >= ox + w and sy + sh >= oy + h
        if covers:
            return source
        from .capture import GrimSource

        return GrimSource(self.window.region)

    # --------------------------------------------------- поиск элементов UI
    def find_buttons(
        self, canvas: np.ndarray, rules: tuple[ButtonRule, ...] = BUTTON_RULES
    ) -> list[UiElement]:
        """Найти цветные кнопки на холсте."""
        found: list[UiElement] = []
        for rule in rules:
            found.extend(self._find_by_rule(canvas, rule))
        return found

    @staticmethod
    def _find_by_rule(canvas: np.ndarray, rule: ButtonRule) -> list[UiElement]:
        """Найти компоненты одного правила (по цвету и размеру)."""
        if canvas is None or canvas.size == 0:
            return []
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        for lo, hi in rule.hue_ranges:
            mask |= cv2.inRange(
                hsv,
                (lo, rule.sat[0], rule.val[0]),
                (hi, rule.sat[1], rule.val[1]),
            )
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found: list[UiElement] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < rule.min_area:
                continue
            if w < rule.min_w or h < rule.min_h:
                continue
            if w > rule.max_w or h > rule.max_h:
                continue
            if w * h > CANVAS_W * CANVAS_H * 0.6:
                continue  # слишком крупное — это фон
            found.append(UiElement(rule.name, int(x), int(y), int(w), int(h)))
        return found

    def find_text(
        self, canvas: np.ndarray, name: str, min_score: float = TEXT_MIN_SCORE
    ) -> UiElement | None:
        """Найти надпись (например, «Co-Op») по эталону из assets/ui.

        Надпись однозначно указывает на кнопку, даже если рядом такой же фон,
        поэтому она надёжнее поиска по цвету. Эталоны сняты с мобильного
        клиента, в веб-клиенте надпись примерно в полтора раза меньше —
        поэтому ищем перебором масштабов.

        Ищем в нижней части холста: надписи «PvP» и «Co-Op» стоят в меню,
        в ряду кнопок у нижнего края. Это ещё и ускоряет поиск.
        """
        template = self._text_templates.get(name)
        if template is None or canvas is None or canvas.size == 0:
            return None
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        top = int(gray.shape[0] * TEXT_SEARCH_TOP)
        region = gray[top:, :]
        if region.size == 0:
            return None
        # Сначала редкая сетка масштабов, потом уточнение вокруг лучшего:
        # полный перебор мелким шагом слишком долгий.
        coarse = tuple(round(0.40 + 0.08 * i, 2) for i in range(9))
        best = self._match_text(region, template, coarse)
        if best is None:
            return None
        scale = best[2]
        fine = tuple(round(scale + step, 2) for step in (-0.04, -0.02, 0.02, 0.04))
        refined = self._match_text(region, template, fine)
        if refined is not None and refined[2] > best[2]:
            best = refined
        if best[2] < min_score:
            return None
        x, y, score, width, height = best
        return UiElement(name, x, y + top, width, height, by_text=True)

    @staticmethod
    def _match_text(
        region: np.ndarray, template: np.ndarray, scales: tuple[float, ...]
    ) -> tuple[int, int, float, int, int] | None:
        """Лучшее совпадение эталона надписи в области (с перебором масштабов)."""
        best: tuple[int, int, float, int, int] | None = None
        for scale in scales:
            width = int(round(template.shape[1] * scale))
            height = int(round(template.shape[0] * scale))
            if width < 8 or height < 8:
                continue
            if width > region.shape[1] or height > region.shape[0]:
                continue
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            if not res.size:
                continue
            # На однотонных кадрах нормализованная корреляция даёт NaN.
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or float(score) > best[2]:
                best = (int(loc[0]), int(loc[1]), float(score), width, height)
        return best

    def find_pvp_button(self, canvas: np.ndarray) -> UiElement | None:
        """Кнопка «PvP» в главном меню.

        Сначала ищем надпись — она надёжнее цвета: рядом с кнопкой много
        оранжевого (значки, рамки), и цветные пятна сливаются в один
        большой блок. Если надписи нет, берём оранжевую кнопку по цвету.
        """
        text = self.find_text(canvas, "pvp")
        if text is not None:
            return UiElement("pvp", text.x, text.y, text.w, text.h, by_text=True)
        pvps = [
            b
            for b in self.find_buttons(canvas)
            if b.kind == "pvp" and b.y > CANVAS_H * 0.55
        ]
        if not pvps:
            return None
        return min(pvps, key=lambda b: b.x)

    def find_coop_button(self, canvas: np.ndarray) -> UiElement | None:
        """Кнопка «Co-Op» в главном меню.

        Сначала ищем надпись «Co-Op». Если её нет (другой клиент, другая
        локализация), ищем ярко-синюю кнопку в одной строке с оранжевой
        «PvP» и правее неё. Позиция не фиксирована — окно браузера может
        быть любого размера, поэтому ищем по геометрии, а не по координатам.
        """
        text = self.find_text(canvas, "coop")
        if text is not None:
            return UiElement("coop", text.x, text.y, text.w, text.h, by_text=True)
        # Запасной путь: ярко-синяя кнопка в одной строке с оранжевой «PvP».
        # Сравниваем именно с кнопкой, а не с надписью: надпись занимает
        # только часть кнопки, и геометрия по ней получается неточной.
        pvps = [
            b
            for b in self.find_buttons(canvas)
            if b.kind == "pvp" and b.y > CANVAS_H * 0.55
        ]
        if not pvps:
            return None
        pvp = min(pvps, key=lambda b: b.x)
        blues = [
            b
            for b in self.find_buttons(canvas, (COOP_BUTTON_RULE,))
            if b.x >= pvp.x + pvp.w - 12
            and b.y < pvp.y + pvp.h
            and b.y + b.h > pvp.y
        ]
        if not blues:
            return None
        best = min(blues, key=lambda b: b.x)
        return UiElement("coop", best.x, best.y, best.w, best.h)

    def find_mana_bar(self, canvas: np.ndarray) -> tuple[int, int, int, int] | None:
        """Найти полосу маны в HUD (зелёная заливка рядом с числом)."""
        x, y, w, h = MANA_BAR_HINT
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        region = hsv[y : y + h, x : x + w]
        # Зелёная заливка шкалы.
        mask = cv2.inRange(region, (35, 90, 60), (85, 255, 255))
        if mask.sum() == 0:
            return None
        return (x, y, w, h)

    def find_cards(self, canvas: np.ndarray) -> list[UiElement]:
        """Найти карточки юнитов в нижнем ряду HUD."""
        y0, y1 = CARD_ROW_Y
        strip = canvas[y0:y1]
        if strip.size == 0:
            return []
        # Карточки — светлые прямоугольники с тёмной обводкой; ищем рамки
        # по контрасту с тёмным фоном панели.
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        cols = edges.sum(axis=0)
        peaks = [i for i, v in enumerate(cols) if v > strip.shape[0] * 0.25]
        if not peaks:
            return []
        # Группируем близкие пики в карточки.
        groups: list[list[int]] = [[peaks[0]]]
        for p in peaks[1:]:
            if p - groups[-1][-1] <= 6:
                groups[-1].append(p)
            else:
                groups.append([p])
        cards: list[UiElement] = []
        for g in groups:
            if len(g) < 2:
                continue
            x0, x1 = g[0], g[-1]
            if x1 - x0 < 25:
                continue
            cards.append(UiElement("card", int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        return cards

    def read_mana(self, canvas: np.ndarray) -> tuple[float, float]:
        """Прочитать энергию.

        В веб-клиенте энергия выводится числом поверх декоративной шкалы,
        поэтому число распознаётся по эталонам цифр. Если распознавание
        не удалось, используется заполнение шкалы как запасной вариант.
        """
        x, y, w, h = ENERGY_REGION
        region = canvas[y : y + h, x : x + w]
        if region.size:
            reading = self.digits.read(region)
            if reading.ok and reading.value > 0:
                return reading.value, reading.confidence
        return self._mana_from_bar(canvas)

    def _mana_from_bar(self, canvas: np.ndarray) -> tuple[float, float]:
        """Запасной способ: доля заполнения зелёной шкалы."""
        x, y, w, h = MANA_BAR_HINT
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        patch = hsv[y : y + h, x : x + w]
        mask = cv2.inRange(patch, (35, 90, 60), (85, 255, 255))
        if mask.sum() == 0:
            return 0.0, 0.0
        cols = mask.any(axis=0)
        indices = np.flatnonzero(cols)
        if len(indices) == 0:
            return 0.0, 0.0
        first = int(indices[0])
        last = first
        for i in range(first, len(cols)):
            if not cols[i]:
                break
            last = i
        fraction = min(1.0, (last - first + 1) / max(1, w - 2))
        return fraction * MANA_MAX, fraction

    def in_battle(self, canvas: np.ndarray) -> bool:
        """Идёт ли сейчас матч.

        Признаки: число энергии в HUD и флакон рядом с ним, ряд карточек
        юнитов и зелёные плитки арены. Все три нужны: на экране итогов
        тоже есть карточки с числами (стоимости), а в меню — зелёный фон.
        """
        ex, ey, ew, eh = ENERGY_REGION
        region = canvas[ey : ey + eh, ex : ex + ew]
        if region.size == 0:
            return False
        if not self.digits._glyphs(region):
            return False
        if not self._has_energy_flask(canvas):
            return False
        if not self._has_card_row(canvas):
            return False
        return self._has_arena(canvas)

    def _has_energy_flask(self, canvas: np.ndarray) -> bool:
        """Есть ли голубой флакон энергии слева от числа."""
        fx, fy, fw, fh = ENERGY_FLASK
        region = canvas[fy : fy + fh, fx : fx + fw]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (85, 90, 130), (125, 255, 255))
        return float(mask.mean()) > 0.05

    def _has_arena(self, canvas: np.ndarray) -> bool:
        """Видно ли игровое поле: зелёные плитки арены.

        На экране итогов вместо поля — красно-синие панели статистики,
        поэтому зелёного там нет.
        """
        x0, y0 = FIELD_QUAD[0]
        x1, y1 = FIELD_QUAD[2]
        region = canvas[int(y0) : int(y1), int(x0) : int(x1)]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        tiles = cv2.inRange(hsv, (40, 60, 90), (95, 255, 255))
        return float(tiles.mean()) > 0.05

    def _has_card_row(self, canvas: np.ndarray) -> bool:
        """Есть ли ряд карточек юнитов в нижней части холста.

        Считаем сами карточки: вертикальные рёбра группируются в пять
        прямоугольников. Одной кнопки «Продолжить» для этого мало.
        """
        y0, y1 = CARD_ROW_Y
        strip = canvas[y0:y1]
        if strip.size == 0:
            return False
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 140)
        cols = edges.sum(axis=0)
        peaks = [i for i, v in enumerate(cols) if v > strip.shape[0] * 0.25]
        if not peaks:
            return False
        groups: list[list[int]] = [[peaks[0]]]
        for p in peaks[1:]:
            if p - groups[-1][-1] <= 8:
                groups[-1].append(p)
            else:
                groups.append([p])
        cards = [g for g in groups if g[-1] - g[0] >= 25]
        return 4 <= len(cards) <= 7

    def find_play_button(self, canvas: np.ndarray) -> tuple[int, int] | None:
        """Найти большую зелёную кнопку «играть» на заставках и экранах загрузки.

        На окнах предложений зелёным бывает фон панели — это не кнопка.
        Отличаем по общей доле зелёного: у кнопки зелёного мало, у фона
        панели — много.
        """
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 120, 120), (85, 255, 255))
        if float(mask.mean()) > 0.12:
            # Зелёного много — это фон, а не кнопка.
            return None
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < 30 or h < 30 or w > 160 or h > 160:
                continue
            if abs(w - h) > 0.35 * max(w, h):
                continue
            if area < 1200:
                continue
            if y < 120:
                # Верхняя полоса — это панель валют с кнопками «+» и
                # зелёная стрелка выбора версии: жать их нельзя.
                continue
            if best is None or area > best[4]:
                best = (int(x), int(y), int(w), int(h), int(area))
        if best is None:
            return None
        x, y, w, h, _ = best
        return (x + w // 2, y + h // 2)

    def _coop_buttons(
        self, buttons: list[UiElement], pvps: list[UiElement]
    ) -> list[UiElement]:
        """Кнопка Co-Op — синяя, справа от оранжевой PvP (запасной путь).

        Обе кнопки стоят в один ряд и в маске синего цвета склеиваются в
        одну компоненту, поэтому кнопку Co-Op вычисляем по геометрии: это
        часть синей полосы правее кнопки PvP. Надёжнее работает
        :meth:`find_coop_button` (надпись и ярко-синяя кнопка), сюда
        попадаем, только если та не нашлась.
        """
        found: list[UiElement] = []
        pvp = min(pvps, key=lambda p: p.x) if pvps else None
        for b in buttons:
            if b.kind != "continue" or not (560 < b.y < 670):
                continue
            if b.center[0] <= CANVAS_W * 0.58:
                continue
            if pvp is not None and b.x < pvp.x + pvp.w:
                # Компонента захватывает и PvP — отрезаем её левую часть.
                left = pvp.x + pvp.w + 5
                if left < b.x + b.w:
                    found.append(UiElement("coop", left, b.y, b.x + b.w - left, b.h))
                continue
            found.append(UiElement("coop", b.x, b.y, b.w, b.h))
        return found

    def navigation_candidates(self, canvas: np.ndarray) -> list[UiElement]:
        """Кандидаты-кнопки для перехода дальше, в порядке приоритета.

        Порядок:
          1. крестик закрытия модального окна (красный круг у правого края
             диалога — он бывает на разной высоте, а не только в углу);
          2. «Понятно»/«Продолжить» внутри диалога;
          3. «Продолжить» на экране итогов (внизу по центру);
          4. PvP и Co-Op в главном меню;
          5. «Отмена» на экране подбора соперника;
          6. большая зелёная кнопка «играть» на заставках;
          7. клик в центр для экранов «нажмите, чтобы продолжить».
        """
        buttons = self.find_buttons(canvas)
        out: list[UiElement] = []

        # «Отмена» на экране подбора соперника: красно-оранжевая кнопка
        # внизу по центру. Без неё бот не может выйти из бесконечного
        # поиска. Идёт первым: это однозначная кнопка, в отличие от
        # красных значков, которые похожи на крестик.
        cancels = [
            b
            for b in buttons
            if b.kind == "cancel"
            and b.y > CANVAS_H * 0.85
            and abs(b.center[0] - CANVAS_W / 2) < 110
        ]
        if cancels:
            out.append(min(cancels, key=lambda b: b.y))

        # Крестик закрытия. Он стоит у правого края диалога, а высота у
        # разных окон разная: у окна с наградой он на середине экрана.
        # Поэтому ищем по правому краю и верхней половине холста, а не
        # по фиксированному углу. Красные значки юнитов в панели колоды
        # слева такому фильтру не отвечают.
        closes = [
            b
            for b in buttons
            if b.kind == "close"
            and b.y < CANVAS_H * 0.55
            and b.x + b.w > CANVAS_W * 0.78
            and b.w <= 46
            and self._is_solid_red(canvas, b)
        ]
        if closes:
            out.append(min(closes, key=lambda b: b.y))

        # PvP в главном меню — оранжевая кнопка. Считаем её заранее: по ней
        # отделяем склеенную с ней синюю кнопку Co-Op.
        pvps = [b for b in buttons if b.kind == "pvp" and b.y > CANVAS_H * 0.55]

        # Кнопка Co-Op: надпись «Co-Op» или ярко-синяя кнопка справа от PvP.
        # Запасной путь (синяя полоса по геометрии) — если надпись не нашлась.
        coop_button = self.find_coop_button(canvas)
        coops = (
            [coop_button]
            if coop_button is not None
            else self._coop_buttons(buttons, pvps)
        )

        # Синяя кнопка внутри диалога: «Понятно», «Продолжить», «Забрать».
        # Кнопка Co-Op в главном меню тоже синяя, поэтому её исключаем:
        # она лежит справа, в нижней части экрана.
        oks = [
            b
            for b in buttons
            if b.kind == "continue"
            and CANVAS_H * 0.25 < b.y < CANVAS_H * 0.85
            and 50 <= b.w <= 260
            and 22 <= b.h <= 70
            and not any(
                b.x < c.x + c.w and c.x < b.x + b.w and b.y < c.y + c.h and c.y < b.y + b.h
                for c in coops
            )
        ]
        if oks:
            ok = min(oks, key=lambda b: (abs(b.center[0] - CANVAS_W / 2), b.y))
            # Плашка «Понятно» чуть выше самой кнопки, поэтому жмём нижнюю
            # часть: так попадаем точно в активную область.
            out.append(UiElement("ok", ok.x, ok.y + int(ok.h * 0.7), ok.w, int(ok.h * 0.3)))

        # Верхняя полоса HUD — не трогаем: там валюты с кнопками «+» и
        # значки меню (в том числе «три полоски» — вход в меню игры).
        # Исключение — красные крестики модальных окон, они идут выше.
        out = [
            b
            for b in out
            if not (b.kind in ("ok", "continue", "pvp", "coop") and b.y < 130)
        ]

        # «Продолжить» на экране итогов — внизу по центру. Важно: раньше PvP
        # стоял раньше него, и бот нажимал по баннеру «ПОРАЖЕНИЕ».
        # Кроме «Продолжить» в нижней полосе живут иконки меню (магазин,
        # колода, щит) — их не трогаем, берём только кнопку у центра.
        continues = [
            b
            for b in buttons
            if b.kind == "continue"
            and b.y > CANVAS_H * 0.85
            and abs(b.center[0] - CANVAS_W / 2) < 70
        ]
        if continues:
            out.append(min(continues, key=lambda b: (abs(b.center[0] - CANVAS_W / 2), -b.y)))

        # PvP в главном меню. Надпись «PvP» надёжнее цвета: рядом с кнопкой
        # много оранжевого, и цветные пятна сливаются в один большой блок.
        pvp_button = self.find_pvp_button(canvas)
        if pvp_button is not None:
            out.append(pvp_button)

        if coops:
            out.append(min(coops, key=lambda b: b.x))
        # «Отмена» на экране подбора соперника: тёмно-красная кнопка внизу
        # по центру. Без неё бот не может выйти из бесконечного поиска.
        cancels = [
            b
            for b in buttons
            if b.kind in ("close", "fight")
            and b.y > CANVAS_H * 0.88
            and b.w >= 55
            and abs(b.center[0] - CANVAS_W / 2) < 90
        ]
        if cancels:
            out.append(min(cancels, key=lambda b: b.y))

        # Зелёная кнопка «играть» на заставках — в самом конце: на окнах
        # предложений зелёным бывает фон, и клик по нему бесполезен.
        play = self.find_play_button(canvas)
        if play is not None:
            out.append(UiElement("play", play[0] - 20, play[1] - 20, 40, 40))

        out.append(UiElement("tap", CANVAS_W // 2 - 10, CANVAS_H // 2 - 10, 20, 20))
        # Кнопки покупки не нажимаем никогда. Кнопка, найденная по надписи,
        # проверку проходит: она опознана буквами, а не цветом (оранжевая
        # «PvP» по цвету неотличима от золотой кнопки магазина).
        return [b for b in out if b.by_text or not self.is_purchase(canvas, b)]

    def navigation_button(self, canvas: np.ndarray) -> UiElement | None:
        """Кнопка, которую нужно нажать, чтобы продвинуться по меню.

        Правила и порядок — те же, что у :meth:`navigation_candidates`,
        здесь берётся первый подходящий кандидат. Дублировать правила в
        двух местах было причиной расхождений: окна предложений и
        модальные диалоги требуют разных кнопок.
        """
        candidates = self.navigation_candidates(canvas)
        return candidates[0] if candidates else None

    def find_summon_button(self, canvas: np.ndarray) -> tuple[int, int]:
        """Найти кнопку призыва — карточку в оранжевом виджете маны.

        Виджет стоит над рядом карточек в нижней части экрана: слева в нём
        карточка юнита (её и надо нажать), справа — счётчик зелий.
        """
        x0, y0, x1, y1 = SUMMON_SEARCH
        region = canvas[y0:y1, x0:x1]
        if region.size:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (8, 120, 120), (26, 255, 255))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            best = None
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if w < 40 or h < 20:
                    continue
                if best is None or area > best[4]:
                    best = (int(x), int(y), int(w), int(h), int(area))
            if best is not None:
                x, y, w, h, _ = best
                # Жмём по карточке — она в левой части виджета.
                return (x0 + x + w // 3, y0 + y + h // 2)
        return (int(SUMMON_BUTTON[0]), int(SUMMON_BUTTON[1]))

    def summon_ready(self, canvas: np.ndarray) -> bool:
        """Хватает ли энергии на призыв.

        Виджет призыва — надёжный источник истины: когда энергии хватает,
        он оранжевый, а когда не хватает — серый (и цена в нём гаснет).
        Распознавать цифры цены не нужно: состояние видно по цвету.
        """
        x0, y0, x1, y1 = SUMMON_SEARCH
        region = canvas[y0:y1, x0:x1]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, (8, 120, 120), (26, 255, 255))
        return float(orange.mean()) > 0.03

    def _load_summon_template(self) -> np.ndarray | None:
        path = Path(__file__).resolve().parent.parent / "assets" / "ui" / "summon.png"
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def split_canvas_y(self) -> int:
        """Высота в координатах холста, разделяющая половины арены.

        В веб-клиенте сверху половина противника, снизу своя, между ними
        полоса с номером волны. Различать стороны по цвету нельзя: фигурки
        юнитов тёплые у обеих сторон.
        """
        return 348

    def assign_teams(self, state) -> None:
        """Разметить стороны юнитов по их положению на арене."""
        from .models import Team

        split = self.split_canvas_y()
        x0, y0 = FIELD_QUAD[0]
        x1, y1 = FIELD_QUAD[2]
        scale = (y1 - y0) / max(1.0, FIELD_H)
        for unit in state.ally_units + state.enemy_units:
            canvas_y = y0 + unit.y * scale
            unit.team = Team.ENEMY if canvas_y < split else Team.ALLY
        ally = [u for u in state.enemy_units if u.team is Team.ALLY]
        state.ally_units.extend(ally)
        state.enemy_units = [u for u in state.enemy_units if u.team is Team.ENEMY]

    # --------------------------------------------------------------- карточки
    def card_slots(self, canvas: np.ndarray) -> list[UiElement]:
        """Позиции карточек юнитов в нижнем ряду.

        Ряд фиксированной ширины, поэтому позиции считаются от левого края
        холста; это надёжнее, чем детектировать рамки отдельных карточек.
        """
        slots: list[UiElement] = []
        y0, y1 = CARD_ROW_Y
        for i in range(CARD_COUNT):
            x = CARD_FIRST_X + i * CARD_SLOT_WIDTH
            slots.append(UiElement("card", x, y0, CARD_SLOT_WIDTH - 8, y1 - y0))
        return slots

    def card_center(self, index: int) -> tuple[int, int]:
        """Центр карточки по индексу (0 — первая слева)."""
        y0, y1 = CARD_ROW_Y
        x = CARD_FIRST_X + index * CARD_SLOT_WIDTH
        return (x + (CARD_SLOT_WIDTH - 8) // 2, (y0 + y1) // 2)

    # ------------------------------------------------------------------ поле
    def field_image(self, canvas: np.ndarray) -> np.ndarray:
        """Вырезать игровое поле из холста по калибровке."""
        src = np.asarray(FIELD_QUAD, dtype=np.float32)
        dst = np.asarray(
            [
                (0.0, 0.0),
                (FIELD_W - 1.0, 0.0),
                (FIELD_W - 1.0, FIELD_H - 1.0),
                (0.0, FIELD_H - 1.0),
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            canvas, matrix, (FIELD_W, FIELD_H), flags=cv2.INTER_LINEAR
        )

    def field_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Точка канонического поля -> точка холста."""
        src = np.asarray(
            [
                (0.0, 0.0),
                (FIELD_W - 1.0, 0.0),
                (FIELD_W - 1.0, FIELD_H - 1.0),
                (0.0, FIELD_H - 1.0),
            ],
            dtype=np.float32,
        )
        dst = np.asarray(FIELD_QUAD, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        pt = np.asarray([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, matrix)[0][0]
        return (float(out[0]), float(out[1]))

    # --------------------------------------------------------------- отчёт
    def describe(self) -> str:
        x, y, w, h = self.canvas_rect
        return (
            f"окно {self.window.x},{self.window.y} {self.window.width}x{self.window.height}; "
            f"холст {CANVAS_W}x{CANVAS_H} при смещении {x},{y} в окне"
        )


def _mouse_clicker():
    """Клик настоящей мышью: своё устройство uinput, иначе ydotool."""
    try:
        from .input import UinputMouseBackend

        backend = UinputMouseBackend()

        def _click(x: int, y: int) -> None:
            backend.click(x, y)

        log.info("ввод: виртуальная мышь (uinput)")
        return _click
    except Exception as exc:  # pragma: no cover
        log.debug("uinput-мышь недоступна: %s", exc)

    try:
        from .input import YdotoolBackend

        backend = YdotoolBackend()

        def _ydo_click(x: int, y: int) -> None:
            backend.click(x, y)

        log.info("ввод: ydotool")
        return _ydo_click
    except Exception as exc:  # pragma: no cover
        log.info("ввод недоступен (%s) — только рекомендации", exc)
    return None


def make_clicker(prefer_cdp: bool = True, mouse: bool = False, game=None):
    """Собрать функцию клика по экранным координатам.

    Порядок предпочтений:
      1. виртуальная мышь (uinput) — курсор виден, попадает точно, работает
         с любым приложением;
      2. DevTools Protocol — клики невидимы, зато работают без доступа к
         ``/dev/uinput``;
      3. ничего — бот только считает и показывает рекомендации.
    """
    if mouse:
        return _mouse_clicker()
    if prefer_cdp:
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "tools"))
            from cdp import Session, click_at, pick_page  # type: ignore[import-not-found]

            session = Session(pick_page())

            if game is not None:

                def _cdp_click(x: int, y: int) -> None:
                    game.click_screen(session, x, y)

            else:

                def _cdp_click(x: int, y: int) -> None:
                    click_at(session, x, y)

            log.info("ввод: DevTools Protocol (Chrome)")
            return _cdp_click
        except Exception as exc:  # pragma: no cover
            log.debug("CDP недоступен: %s", exc)

    return _mouse_clicker()


def load_layout(path: str | Path) -> dict:
    """Загрузить раскладку из JSON (для переноса на другое разрешение)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class LiveStep:
    """Результат одного шага живого бота."""

    canvas: np.ndarray | None
    field: np.ndarray | None
    state: object | None  # GameState
    action: object | None  # Action
    mana: float = 0.0
    mana_confidence: float = 0.0
    clicked: list[tuple[int, int]] = field(default_factory=list)
    affordable: bool = False  # хватает ли энергии на призыв (по виджету)
    in_battle: bool = True  # шёл ли в кадре бой (иначе шаг пропущен)


class LiveBot:
    """Полный цикл бота на живой игре: кадр -> зрение -> решение -> клики.

    Клики выполняются переданной функцией ``clicker(x, y)`` — так один и
    тот же бот работает и через DevTools Protocol (браузер), и через
    ydotool (любое приложение).
    """

    def __init__(
        self,
        config,
        game: LiveGame,
        clicker=None,
        source=None,
        real_templates: str | Path | None = None,
    ):
        from .planner import Planner
        from .vision import VisionEngine

        self.config = config
        self.game = game
        self.clicker = clicker
        self.source = source
        self.vision = VisionEngine(config)
        if real_templates:
            self.vision.vision.templates_dir = str(real_templates)
        self.vision.load_templates()
        self.planner = Planner(config)
        self.deck_order = list(config.vision.units.keys())
        self._last_summon = 0.0
        self.summon_cooldown = 2.0
        # Цены в конфиге описывают симулятор. В живой игре цену призыва
        # показывает виджет (и он серый, когда энергии не хватает), поэтому
        # планировщику отдаём единичную стоимость: решение «звать или ждать»
        # принимает виджет, а не конфиг.
        import dataclasses

        self.planner.specs = {
            name: dataclasses.replace(spec, cost=1) for name, spec in self.planner.specs.items()
        }

    def card_index(self, kind: str) -> int | None:
        """Индекс карточки юнита в колоде (порядок как в конфиге)."""
        try:
            return self.deck_order.index(kind)
        except ValueError:
            return None

    def step(self) -> LiveStep:
        """Выполнить один шаг: захват, анализ, решение и действие."""
        canvas = self.game.grab_canvas(self.source)
        if canvas is None:
            return LiveStep(None, None, None, None)

        # Сначала проверяем, что идёт бой. На экране итогов в области энергии
        # лежит совсем другое: бот считал бы оттуда случайные числа и кликал
        # бы в пустое место. В меню — тем более.
        if not self.game.in_battle(canvas):
            log.debug("бой не идёт — шаг пропущен")
            return LiveStep(canvas, None, None, None, in_battle=False)

        field = self.game.field_image(canvas)
        mana, confidence = self.game.read_mana(canvas)
        # Хватает ли энергии на призыв — видно по виджету: оранжевый значит
        # можно звать, серый — нельзя.
        affordable = self.game.summon_ready(canvas)

        state = self.vision.build_state(
            field,
            mana=mana if affordable else 0.0,
            mana_conf=confidence,
            available=self.available_kinds(mana if affordable else 0.0),
        )
        self.game.assign_teams(state)
        action = self.planner.decide(state)

        clicked: list[tuple[int, int]] = []
        if action.kind == "spawn" and action.unit_kind:
            clicked = self.deploy(action, mana)

        return LiveStep(
            canvas, field, state, action, mana, confidence, clicked, affordable, True
        )

    def available_kinds(self, mana: float) -> tuple[str, ...]:
        """Какие юниты доступны: есть карточка в колоде.

        Стоимость в живой игре у всех одна и та же (её показывает виджет),
        поэтому отсеиваем только по наличию карточки.
        """
        return tuple(k for k in self.config.vision.units if self.card_index(k) is not None)

    def deploy(self, action, mana_before: float) -> list[tuple[int, int]]:
        """Призвать юнита: нажать кнопку «Призвать» в своей половине арены.

        В веб-клиенте карточки нажимать не нужно: кнопка призыва сама
        берёт доступного юнита и списывает его стоимость. Если энергия не
        списалась — юнита выставить нечем, ждём следующего тика.
        """
        if self.clicker is None:
            return []
        if time.time() - self._last_summon < self.summon_cooldown:
            # Недавно уже призывали — не спамим кнопку.
            return []
        canvas = self.game.last_canvas
        if canvas is None:
            return []
        if not self.game.in_battle(canvas):
            # Бой закончился: призывать нельзя, в области энергии уже лежит
            # совсем другое.
            log.debug("бой не идёт — призыв пропущен")
            return []
        if not self.game.summon_ready(canvas):
            # Виджет серый: энергии не хватает, кнопка не сработает.
            log.debug("энергии не хватает (виджет серый)")
            return []
        point = self.game.find_summon_button(canvas)
        screen = self.game.to_screen(*point)
        self.clicker(*screen)
        time.sleep(0.7)
        # Проверяем по свежему кадру: энергия должна списаться.
        fresh = self.game.grab_canvas(self.source)
        if fresh is not None and not self.game.in_battle(fresh):
            # Бой закончился прямо во время призыва: энергию на экране
            # итогов читать нельзя.
            log.debug("бой закончился во время призыва")
            return [screen]
        after = self.game.read_mana(fresh)[0] if fresh is not None else mana_before
        if mana_before - after > 1:
            self._last_summon = time.time()
            log.info("юнит призван (энергия %.0f -> %.0f)", mana_before, after)
        else:
            log.debug("энергия не списалась (%.0f -> %.0f)", mana_before, after)
        return [screen]
