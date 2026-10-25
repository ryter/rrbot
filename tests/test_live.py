"""Тесты живой игры: цифры, распознавание интерфейса, калибровка."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402

from rrbot.config import load_config  # noqa: E402
from rrbot.digits import DigitReader  # noqa: E402
from rrbot.live import (  # noqa: E402
    ENERGY_REGION,
    SUMMON_SEARCH,
    GameWindow,
    LiveBot,
    LiveGame,
    Matchmaking,
    UiElement,
    find_canvas_rect,
    screen_changed,
    wait_for_battle,
)

RECORDED = Path("/tmp/opencode/observe/frames")
# Значения энергии на записанных кадрах (прочитаны вручную).
TRUTH = ["60", "80", "110", "130", "20", "60", "70", "80", "90", "120"]


class TestDigits:
    def test_эталоны_загружаются(self):
        reader = DigitReader()
        assert reader.known, "эталоны цифр не найдены"
        assert {"0", "1", "2"} <= reader.known

    def test_число_читается_верно(self, tmp_path):
        reader = DigitReader()
        # Рисуем цифры из эталонов в поле — проверяем на своих же шаблонах.
        region = np.zeros((16, 40), dtype=np.uint8)
        x = 2
        for ch in "120":
            tpl = reader.templates[ch][0]
            region[1:15, x : x + 8] = tpl
            x += 9
        reading = reader.read(region)
        assert reading.text == "120"
        assert reading.value == 120
        assert reading.confidence > 0.9

    def test_пустая_область_даёт_пустое_чтение(self):
        reader = DigitReader()
        reading = reader.read(np.zeros((16, 40), dtype=np.uint8))
        assert reading.text == ""
        assert reading.value == 0

    @pytest.mark.skipif(not RECORDED.exists(), reason="нет записанных кадров")
    def test_точность_на_игровых_кадрах(self):
        reader = DigitReader()
        frames = sorted(RECORDED.glob("*.png"))[: len(TRUTH)]
        correct = 0
        for path, expected in zip(frames, TRUTH):
            canvas = cv2.imread(str(path))
            if canvas is None:
                continue
            x, y, w, h = ENERGY_REGION
            reading = reader.read(canvas[y : y + h, x : x + w])
            if reading.text == expected:
                correct += 1
        assert correct >= len(TRUTH) - 1, f"верно {correct} из {len(TRUTH)}"

    def test_каждая_известная_цифра_узнаётся(self):
        """Цифра, взятая из своего эталона, должна читаться верно."""
        reader = DigitReader()
        for digit, samples in reader.templates.items():
            reading = reader.read(samples[0])
            assert reading.text == digit, f"эталон «{digit}» прочитан как «{reading.text}»"


class TestLiveLayout:
    def test_энергия_читается_с_игра(self):
        game = LiveGame()
        assert game.read_mana(np.zeros((804, 492, 3), dtype=np.uint8))[0] == 0.0

    def test_кнопки_найдены_на_бою(self):
        frames = Path("/tmp/opencode/observe/frames")
        if not frames.exists():
            pytest.skip("нет записанных кадров")
        game = LiveGame()
        path = sorted(frames.glob("*.png"))[0]
        canvas = cv2.imread(str(path))
        if canvas is None:
            pytest.skip("кадр не читается")
        # В бою должен находиться ряд карточек и распознаваться энергия.
        assert len(game.card_slots(canvas)) == 5
        assert game.digits.read(canvas[682:698, 138:176]).ok

    def test_кнопка_призыва_над_рядом_карточек(self):
        from rrbot.live import CANVAS_H, CARD_ROW_Y, SUMMON_BUTTON

        # Кнопка призыва — в виджете маны, который стоит выше ряда карточек.
        assert 0 <= SUMMON_BUTTON[0] <= 492
        assert CARD_ROW_Y[0] - 60 <= SUMMON_BUTTON[1] < CARD_ROW_Y[0]
        assert SUMMON_BUTTON[1] < CANVAS_H

    def test_шаблон_кнопки_призва_загружается(self):
        game = LiveGame()
        assert game._load_summon_template() is not None


class TestCoordinates:
    """Переводы координат: экран <-> холст."""

    def test_экран_и_холст_совпадают_туда_и_обратно(self):
        game = LiveGame()
        game.canvas_offset = (1192, 177)
        game.canvas_size = (492, 804)
        for point in [(228, 685), (0, 0), (491, 803)]:
            screen = game.to_screen(*point)
            assert game.to_canvas(*screen) == point

    def test_смещение_холста_считается_от_окна(self):
        """Прямоугольник холста внутри окна -> экранные координаты."""
        game = LiveGame()
        game.window = GameWindow(967, 38, 941, 1030)
        game.set_canvas_rect((225, 139, 492, 804))
        assert game.canvas_offset == (1192, 177)
        assert game.to_screen(228, 685) == (1420, 862)


class TestCanvasDetection:
    """Холст находится по рамке на снимке окна — без обращения к странице."""

    @staticmethod
    def _window_with_frame(dark: bool = False, broken: bool = False) -> np.ndarray:
        """Окно браузера с тёмно-синей рамкой холста."""
        frame = np.full((1030, 941, 3), 180, dtype=np.uint8)
        # Рамка: BGR (60, 32, 10) -> H≈107, S≈213. На затемнённом экране
        # яркость падает почти до нуля, оттенок остаётся.
        colour = (16, 9, 3) if dark else (60, 32, 10)
        frame[139:148, 225:717] = colour
        frame[936:943, 225:717] = colour
        if not broken:
            frame[139:943, 225:234] = colour
            frame[139:943, 708:717] = colour
        else:
            # Окно перекрывает середину рамки — остаются полосы.
            frame[139:300, 225:234] = colour
            frame[800:943, 225:234] = colour
            frame[139:300, 708:717] = colour
            frame[800:943, 708:717] = colour
        return frame

    def test_рамка_находится(self):
        rect = find_canvas_rect(self._window_with_frame())
        assert rect == (225, 139, 492, 804)

    def test_рамка_находится_на_затемнённом_экране(self):
        rect = find_canvas_rect(self._window_with_frame(dark=True))
        assert rect == (225, 139, 492, 804)

    def test_рамка_собирается_из_полос(self):
        """Окно разрывает кольцо рамки — прямоугольник всё равно найден."""
        rect = find_canvas_rect(self._window_with_frame(broken=True))
        assert rect == (225, 139, 492, 804)

    def test_пустое_окно_не_даёт_холста(self):
        frame = np.full((1030, 941, 3), 180, dtype=np.uint8)
        assert find_canvas_rect(frame) is None

    def test_пустой_кадр_не_даёт_холста(self):
        assert find_canvas_rect(np.zeros((10, 10, 3), dtype=np.uint8)) is None


class TestSummonWidget:
    """Виджет призыва: по нему видно, хватает ли энергии."""

    @staticmethod
    def _canvas(widget_hsv: tuple[int, int, int]) -> np.ndarray:
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        # Оранжевый (или серый) виджет в области поиска.
        x0, y0, x1, y1 = SUMMON_SEARCH
        color = np.array(widget_hsv, dtype=np.uint8)
        canvas[y0 : y1, x0 : x1] = color
        return canvas

    def test_оранжевый_виджет_готов_к_призыву(self):
        game = LiveGame()
        canvas = self._canvas((0, 165, 255))  # оранжевый в BGR
        assert game.summon_ready(canvas)

    def test_серый_виджет_не_готов(self):
        game = LiveGame()
        canvas = self._canvas((160, 160, 160))  # серый: нет насыщенности
        assert not game.summon_ready(canvas)

    def test_пустой_кадр_не_готов(self):
        game = LiveGame()
        assert not game.summon_ready(np.zeros((804, 492, 3), dtype=np.uint8))


class TestPurchaseProtection:
    """Кнопки покупки нажимать нельзя."""

    @staticmethod
    def _gold_button(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Золотая кнопка как в игре: заливка и белый символ."""
        canvas[y : y + h, x : x + w] = (0, 165, 255)
        template = cv2.imread(str(ROOT / "assets" / "ui" / "currency_eur.png"), cv2.IMREAD_GRAYSCALE)
        if template is None:
            return
        th, tw = template.shape
        canvas[y + h // 2 - th // 2 : y + h // 2 - th // 2 + th,
               x + w - 14 : x + w - 14 + tw] = np.dstack([template] * 3)

    def test_золотая_кнопка_с_символом_не_нажимается(self):
        game = LiveGame()
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        self._gold_button(canvas, 180, 690, 120, 45)
        kinds = [b.kind for b in game.navigation_candidates(canvas)]
        assert "pvp" not in kinds
        assert "continue" not in kinds

    def test_кнопка_покупки_опознаётся(self):
        game = LiveGame()
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        self._gold_button(canvas, 180, 690, 120, 45)
        element = UiElement("pvp", 180, 690, 120, 45)
        assert game.is_purchase(canvas, element)

    def test_обычная_кнопка_не_считается_покупкой(self):
        game = LiveGame()
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        canvas[690:735, 180:300] = (220, 120, 40)  # синяя кнопка
        element = UiElement("continue", 180, 690, 120, 45)
        assert not game.is_purchase(canvas, element)
    """Порядок кнопок в меню: важно не промахнуться по покупке."""

    @staticmethod
    def _dialog_canvas() -> np.ndarray:
        """Диалог: красный крестик у правого края, синяя кнопка снизу."""
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        # Крестик: красный (H=0) круг 30x30 у правого края, на высоте 278.
        canvas[278:308, 409:439] = (0, 0, 220)
        # Синяя кнопка «Продолжить» внутри диалога.
        canvas[637:695, 91:229] = (220, 120, 40)
        return canvas

    def test_крестик_находится_не_только_в_углу(self):
        game = LiveGame()
        kinds = [b.kind for b in game.navigation_candidates(self._dialog_canvas())]
        assert "close" in kinds

    def test_крестик_идёт_раньше_кнопок_меню(self):
        game = LiveGame()
        candidates = game.navigation_candidates(self._dialog_canvas())
        assert candidates[0].kind == "close"


class FakeClock:
    """Часы, которые идут только когда «спит»."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestWaitForBattle:
    """Ожидание матча: ждём боя, а не «изменившегося экрана»."""

    def test_матч_начался_с_третьей_проверки(self):
        calls: list[int] = []

        def grab() -> np.ndarray:
            calls.append(1)
            return np.zeros((2, 2, 3), dtype=np.uint8)

        clock = FakeClock()
        started = wait_for_battle(
            grab,
            lambda frame: len(calls) >= 3,
            timeout=30.0,
            interval=1.0,
            sleep=clock.sleep,
            clock=clock,
        )
        assert started
        assert len(calls) == 3, "проверять надо до наступления таймаута"

    def test_анимированный_экран_не_считается_матчем(self):
        """Экран подбора соперника анимируется: «изменилось» всегда истинно."""
        counter = itertools.count()

        def grab() -> np.ndarray:
            # каждый кадр отличается от предыдущего
            return np.full((2, 2, 3), next(counter) % 255, dtype=np.uint8)

        clock = FakeClock()
        started = wait_for_battle(
            grab,
            lambda frame: False,
            timeout=5.0,
            interval=1.0,
            sleep=clock.sleep,
            clock=clock,
        )
        assert not started, "экран менялся, но боя не было — матчем это не считается"
        assert clock.now >= 5.0, "ждали до таймаута"

    def test_ожидание_кончается_по_таймауту(self):
        clock = FakeClock()
        started = wait_for_battle(
            lambda: np.zeros((2, 2, 3), dtype=np.uint8),
            lambda frame: False,
            timeout=2.5,
            interval=1.0,
            sleep=clock.sleep,
            clock=clock,
        )
        assert not started
        assert 2.5 <= clock.now < 3.5

    def test_без_кадра_матчем_не_считается(self):
        clock = FakeClock()
        started = wait_for_battle(
            lambda: None,
            lambda frame: True,
            timeout=1.0,
            interval=1.0,
            sleep=clock.sleep,
            clock=clock,
        )
        assert not started, "кадра нет — проверить нечего"

    def test_проверка_боя_идёт_по_кадру(self):
        """Признак начала боя — только состояние кадра."""
        canvas = np.zeros((2, 2, 3), dtype=np.uint8)
        seen: list[str] = []

        def is_battle(frame: np.ndarray) -> bool:
            seen.append("проверка")
            return True

        wait_for_battle(lambda: canvas, is_battle, timeout=1.0, interval=1.0)
        assert seen == ["проверка"]


class TestScreenChanged:
    """Смена экрана: нужна, чтобы понять, что клик куда-то попал."""

    def test_одинаковые_кадры_не_считаются_изменением(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert not screen_changed(frame, frame.copy())

    def test_разные_кадры_считаются_изменением(self):
        before = np.zeros((10, 10, 3), dtype=np.uint8)
        after = np.full((10, 10, 3), 255, dtype=np.uint8)
        assert screen_changed(before, after)

    def test_нет_кадра_считается_изменением(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert screen_changed(frame, None)
        assert screen_changed(None, frame)


class TestMatchmaking:
    """Подбор матча: PvP, Co-Op и переключение между ними."""

    def test_из_начала_берём_pvp(self):
        plan = Matchmaking()
        assert not plan.use_coop
        assert plan.mode_name == "PvP"

    def test_после_двух_неудачных_подборов_включается_coop(self):
        plan = Matchmaking()
        plan.begin_attempt()
        assert not plan.on_search_failed(), "первая неудача ещё не переключает"
        assert not plan.use_coop
        plan.begin_attempt()
        assert plan.on_search_failed(), "вторая неудача — переключаемся на Co-Op"
        assert plan.use_coop
        assert plan.mode_name == "Co-Op"

    def test_неудача_одной_попытки_считается_один_раз(self):
        """И «истекло время», и «нажали Отмена» — это одна и та же попытка."""
        plan = Matchmaking()
        plan.begin_attempt()
        assert not plan.on_search_failed()
        assert not plan.on_search_cancelled(), "неудача уже посчитана"
        assert plan.pvp_waits == 1
        assert not plan.use_coop

    def test_после_матча_счётчик_сбрасывается(self):
        plan = Matchmaking()
        plan.begin_attempt()
        plan.on_search_failed()
        plan.on_match_started()
        assert plan.pvp_waits == 0
        assert not plan.use_coop

    def test_отмена_подбора_считается_неудачей(self):
        plan = Matchmaking()
        plan.begin_attempt()
        assert not plan.on_search_cancelled()
        plan.begin_attempt()
        assert plan.on_search_cancelled()
        assert plan.use_coop

    def test_отмена_без_попытки_ничего_не_меняет(self):
        plan = Matchmaking()
        assert not plan.on_search_cancelled()
        assert plan.pvp_waits == 0

    def test_режим_pvp_не_переключается(self):
        plan = Matchmaking(mode="pvp")
        for _ in range(5):
            plan.begin_attempt()
            assert not plan.on_search_failed()
        assert not plan.use_coop
        assert plan.mode_name == "PvP"

    def test_режим_coop_сразу_берёт_coop(self):
        plan = Matchmaking(mode="coop")
        assert plan.use_coop
        assert plan.mode_name == "Co-Op"

    def test_неизвестный_режим_ошибка(self):
        with pytest.raises(ValueError):
            Matchmaking(mode="бог")


class FakeGame:
    """Подделка живой игры: кадр и ответы заданы заранее."""

    def __init__(self, in_battle: bool = False, canvas=None):
        self._in_battle = in_battle
        self.last_canvas = (
            canvas if canvas is not None else np.zeros((804, 492, 3), dtype=np.uint8)
        )
        self.mana_reads = 0
        self.summon_ready_flag = True
        self.mana_value = 120.0

    def grab_canvas(self, source=None):
        return self.last_canvas

    def in_battle(self, canvas) -> bool:
        return self._in_battle

    def read_mana(self, canvas):
        self.mana_reads += 1
        return self.mana_value, 0.9

    def summon_ready(self, canvas) -> bool:
        return self.summon_ready_flag

    def find_summon_button(self, canvas):
        return (228, 685)

    def to_screen(self, x: int, y: int):
        return (x, y)

    def field_image(self, canvas):
        return canvas

    def assign_teams(self, state) -> None:
        pass


class TestBattleGuard:
    """Пока бой не идёт, энергию читать и призывать нельзя."""

    @staticmethod
    def _bot(game: FakeGame, clicks: list):
        config = load_config(ROOT / "config" / "live.yaml")
        return LiveBot(config, game, clicker=lambda x, y: clicks.append((x, y)))

    def test_без_боя_энергия_не_читается(self):
        game = FakeGame(in_battle=False)
        clicks: list = []
        step = self._bot(game, clicks).step()
        assert not step.in_battle
        assert step.mana == 0.0
        assert game.mana_reads == 0, "на экране итогов в области энергии лежит другое"
        assert clicks == []

    def test_без_боя_призыва_не_бывает(self):
        game = FakeGame(in_battle=False)
        clicks: list = []
        bot = self._bot(game, clicks)
        assert bot.deploy(None, 120.0) == []
        assert clicks == []

    def test_в_бою_призыв_происходит(self):
        game = FakeGame(in_battle=True)
        clicks: list = []
        bot = self._bot(game, clicks)
        bot.summon_cooldown = 0.0
        # после призыва энергия падает — значит юнит встал
        game.mana_value = 20.0
        assert bot.deploy(None, 120.0) == [(228, 685)]
        assert clicks == [(228, 685)]

    def test_виджет_серый_значит_не_звали(self):
        game = FakeGame(in_battle=True)
        game.summon_ready_flag = False
        clicks: list = []
        bot = self._bot(game, clicks)
        assert bot.deploy(None, 120.0) == []
        assert clicks == []


class TestMenuButtons:
    """Кнопки главного меню: «PvP» и «Co-Op»."""

    @staticmethod
    def _menu_canvas() -> np.ndarray:
        """Меню: оранжевая кнопка «PvP» и ярко-синяя «Co-Op» справа.

        Фон меню тёмно-синий, но достаточно яркий, чтобы обычный диапазон
        «continue» сливал его с кнопкой в один кусок — именно из-за этого
        кнопка «Co-Op» раньше не находилась.
        """
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        canvas[:, :] = (135, 73, 14)  # фон меню (H≈106, V≈135)
        canvas[604:671, 79:239] = (0, 170, 245)  # оранжевая «PvP»
        canvas[604:671, 253:412] = (247, 133, 36)  # ярко-синяя «Co-Op»
        return canvas

    def test_кнопка_coop_находится_по_яркости(self):
        game = LiveGame()
        game._text_templates = {}  # надписи нет — работает запасной путь
        button = game.find_coop_button(self._menu_canvas())
        assert button is not None
        assert button.kind == "coop"
        assert button.x >= 253, "кнопка Co-Op стоит правее кнопки PvP"
        assert button.y >= 604 and button.y < 671

    def test_ярко_синяя_кнопка_не_сливается_с_фоном(self):
        """Фон меню ярче 120 — обычный диапазон «continue» его не отделяет."""
        game = LiveGame()
        game._text_templates = {}
        kinds = [(b.kind, b.x) for b in game.find_buttons(self._menu_canvas())]
        assert not any(kind == "continue" for kind, _ in kinds)

    def test_кнопка_pvp_находится_по_цвету(self):
        game = LiveGame()
        game._text_templates = {}
        button = game.find_pvp_button(self._menu_canvas())
        assert button is not None
        assert button.x == 79 and button.w == 160

    def test_в_меню_есть_кнопка_coop(self):
        game = LiveGame()
        game._text_templates = {}
        kinds = [b.kind for b in game.navigation_candidates(self._menu_canvas())]
        assert "coop" in kinds

    def test_оранжевая_pvp_не_считается_покупкой_по_надписи(self):
        """Кнопка «PvP» оранжевая — по цвету не отличить от кнопки магазина.

        Поэтому её надо искать по надписи: найденная по буквам кнопка
        проверку «покупка» проходит.
        """
        game = LiveGame()
        canvas = self._menu_canvas()
        colored = UiElement("pvp", 79, 604, 160, 67)
        assert game.is_purchase(canvas, colored), "по цвету это неотличимо от покупки"
        by_text = UiElement("pvp", 164, 624, 43, 27, by_text=True)
        assert not game.is_purchase(canvas, by_text)

    def test_кнопка_coop_не_считается_покупкой(self):
        game = LiveGame()
        canvas = self._menu_canvas()
        assert not game.is_purchase(canvas, UiElement("coop", 253, 604, 159, 67))


class TestTextButtons:
    """Поиск кнопок по надписи: надпись надёжнее цвета."""

    @staticmethod
    def _canvas_with_text(name: str, x: int, y: int, scale: float) -> np.ndarray:
        canvas = np.zeros((804, 492, 3), dtype=np.uint8)
        canvas[:, :] = (135, 73, 14)  # фон меню
        template = cv2.imread(str(ROOT / "assets" / "ui" / f"text_{name}.png"))
        assert template is not None, f"нет эталона надписи {name}"
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)
        small = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        canvas[y : y + height, x : x + width] = small
        return canvas

    def test_надпись_coop_находится_по_эталону(self):
        game = LiveGame()
        found = game.find_text(self._canvas_with_text("coop", 300, 620, 0.64), "coop")
        assert found is not None
        assert abs(found.x - 300) < 15
        assert abs(found.y - 620) < 15

    def test_надпись_pvp_находится_по_эталону(self):
        game = LiveGame()
        found = game.find_text(self._canvas_with_text("pvp", 160, 620, 0.68), "pvp")
        assert found is not None
        assert abs(found.x - 160) < 15
        assert abs(found.y - 620) < 15

    def test_кнопка_coop_находится_по_надписи(self):
        game = LiveGame()
        button = game.find_coop_button(self._canvas_with_text("coop", 300, 620, 0.64))
        assert button is not None
        assert button.kind == "coop"
        assert button.by_text, "кнопка найдена по буквам — её нельзя отбросить как покупку"

    def test_пустой_кадр_не_даёт_надписи(self):
        game = LiveGame()
        assert game.find_text(np.zeros((804, 492, 3), dtype=np.uint8), "coop") is None

    def test_неизвестная_надпись_не_находится(self):
        game = LiveGame()
        canvas = self._canvas_with_text("coop", 300, 620, 0.64)
        assert game.find_text(canvas, "такой-надписи-нет") is None


class FakeMenuGame:
    """Подделка игры для проверки оркестровки автобоя."""

    def __init__(self, battle_after: str | None = None):
        # После клика по какой кнопке начинается бой.
        self.battle_after = battle_after
        self.battle = False
        self.clicked: list[str] = []

    def in_battle(self, canvas) -> bool:
        return self.battle

    def navigation_candidates(self, canvas):
        return [
            UiElement("pvp", 79, 604, 160, 67),
            UiElement("coop", 253, 604, 160, 67),
        ]


class TestAutoplayFlow:
    """Оркестровка автобоя: ждём матч и переключаемся на Co-Op."""

    @staticmethod
    def _run(monkeypatch, mode: str = "auto", battle_after: str | None = None):
        import autoplay

        class _NoSleep:
            @staticmethod
            def sleep(seconds: float) -> None:
                pass

        monkeypatch.setattr(autoplay, "time", _NoSleep)
        game = FakeMenuGame(battle_after=battle_after)
        clicks: list[str] = []

        def click_canvas(cx: int, cy: int):
            kind = "pvp" if cx < 200 else "coop"
            clicks.append(kind)
            if kind == game.battle_after:
                game.battle = True
            return (cx, cy)

        plan = Matchmaking(mode=mode)
        started = autoplay.go_to_battle(
            game,
            lambda: np.zeros((804, 492, 3), dtype=np.uint8),
            click_canvas,
            plan,
            timeout_match=0.0,
        )
        return started, clicks, plan

    def test_после_двух_неудачных_подборов_берём_coop(self, monkeypatch):
        started, clicks, plan = self._run(monkeypatch, battle_after="coop")
        assert started
        assert clicks == ["pvp", "pvp", "coop"], "сначала PvP, после двух неудач — Co-Op"
        assert plan.use_coop

    def test_после_клика_pvp_ждём_матч(self, monkeypatch):
        started, clicks, plan = self._run(monkeypatch, battle_after="pvp")
        assert started
        assert clicks == ["pvp"], "бой начался сразу — второй раз кликать не надо"

    def test_режим_pvp_не_переключается(self, monkeypatch):
        started, clicks, plan = self._run(monkeypatch, mode="pvp")
        assert not started
        assert "coop" not in clicks
        assert not plan.use_coop

    def test_режим_coop_сразу_берёт_coop(self, monkeypatch):
        started, clicks, plan = self._run(monkeypatch, mode="coop", battle_after="coop")
        assert started
        assert clicks == ["coop"]
