"""Слой управления игрой.

По умолчанию бот работает в режиме **рекомендаций**: он считает, что и куда
нажать, и показывает это в интерфейсе, но не трогает мышь. Реальные клики
включаются флагом ``allow_input`` в конфиге и требуют установленного
``pyautogui`` — иначе остаётся только подсказка.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

from .models import Action

log = logging.getLogger(__name__)


def shutil_which(cmd: str) -> str | None:
    """Обёртка над shutil.which для точечного импорта."""
    import shutil

    return shutil.which(cmd)


@dataclass
class ClickTarget:
    """Экранная точка, куда нужно нажать."""

    x: int
    y: int
    label: str = ""
    kind: str = "spawn"


class InputBackend(Protocol):
    """Минимальный интерфейс устройства ввода."""

    available: bool

    def click(self, x: int, y: int) -> None: ...


class NullBackend:
    """Заглушка: клики не выполняются, но решения фиксируются."""

    available = False

    def click(self, x: int, y: int) -> None:
        log.info("рекомендация: нажать (%d, %d)", x, y)


class PyAutoGuiBackend:
    """Реальные клики через pyautogui (если пакет установлен)."""

    available = True

    def __init__(self) -> None:
        import pyautogui  # type: ignore[import-not-found]

        self._pyautogui = pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

    def click(self, x: int, y: int) -> None:
        self._pyautogui.click(x=x, y=y)


def desktop_size() -> tuple[int, int]:
    """Размер рабочей области (для диапазонов абсолютных осей)."""
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["hyprctl", "monitors"], capture_output=True, timeout=5, check=False
        ).stdout.decode()
    except Exception:  # pragma: no cover
        return (1920, 1080)
    sizes = [(int(w), int(h)) for w, h in re.findall(r"(\d+)x(\d+)@\d", out)]
    if not sizes:
        return (1920, 1080)
    return (max(w for w, _ in sizes), max(h for _, h in sizes))


class UinputMouseBackend:
    """Клики через собственное абсолютное устройство uinput.

    Почему не ydotool: тот эмулирует абсолютные координаты относительными
    смещениями, и компоновщик применяет к ним свой масштаб — курсор
    промахивался вдвое. Своё устройство с осями ABS_X/ABS_Y попадает в
    точку пиксель в пиксель, и координаты не зависят от того, где в этот
    момент находится настоящая мышь.
    """

    available = True

    def __init__(self, width: int | None = None, height: int | None = None):
        from evdev import AbsInfo, UInput, ecodes  # type: ignore[import-not-found]

        if width is None or height is None:
            width, height = desktop_size()
        self._ecodes = ecodes
        self._device = UInput(
            {
                ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE],
                ecodes.EV_ABS: [
                    (ecodes.ABS_X, AbsInfo(0, 0, int(width) - 1, 0, 0, 0)),
                    (ecodes.ABS_Y, AbsInfo(0, 0, int(height) - 1, 0, 0, 0)),
                ],
            },
            name="rrbot-virtual-mouse",
        )
        self.width = int(width)
        self.height = int(height)
        log.info("виртуальная мышь: %dx%d (%s)", self.width, self.height, self._device.device)

    def move(self, x: int, y: int) -> None:
        ecodes = self._ecodes
        x = max(0, min(self.width - 1, int(x)))
        y = max(0, min(self.height - 1, int(y)))
        self._device.write(ecodes.EV_ABS, ecodes.ABS_X, x)
        self._device.write(ecodes.EV_ABS, ecodes.ABS_Y, y)
        self._device.write(ecodes.EV_SYN, 0, 0)

    def click(self, x: int, y: int) -> None:
        """Кликнуть в экранных координатах."""
        ecodes = self._ecodes
        self.move(x, y)
        time.sleep(0.05)
        self._device.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
        self._device.write(ecodes.EV_SYN, 0, 0)
        time.sleep(0.06)
        self._device.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
        self._device.write(ecodes.EV_SYN, 0, 0)

    def drag(self, x1: int, y1: int, x2: int, y2: int, steps: int = 12,
             press_ms: int = 200, step_ms: int = 40, hold_ms: int = 200) -> None:
        """Перетащить: зажать, провести, отпустить.

        Как в боте (cpp/src/mouse.cpp): игра подхватывает юнита только после
        короткого нажатия, а быстрое движение считает свайпом.
        """
        ecodes = self._ecodes
        self.move(x1, y1)
        time.sleep(press_ms / 1000.0)
        self._device.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
        self._device.write(ecodes.EV_SYN, 0, 0)
        time.sleep(press_ms / 1000.0)
        for step in range(1, steps + 1):
            t = step / steps
            self.move(int(round(x1 + (x2 - x1) * t)), int(round(y1 + (y2 - y1) * t)))
            time.sleep(step_ms / 1000.0)
        time.sleep(hold_ms / 1000.0)
        self._device.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
        self._device.write(ecodes.EV_SYN, 0, 0)

    def close(self) -> None:
        try:
            self._device.close()
        except Exception:  # pragma: no cover
            pass


class YdotoolBackend:
    """Клики через виртуальное устройство ydotool (Wayland, без X11).

    Требует запущенного ``ydotoold`` и доступа к ``/dev/uinput``.
    Работает с любым приложением, а не только с браузером.
    """

    available = True

    def __init__(self) -> None:
        import shutil

        if not shutil.which("ydotool"):
            raise RuntimeError("ydotool не установлен")
        self._socket = os.environ.get("YDOTOOL_SOCKET", "")
        self._ensure_daemon()
        self.scale = self._calibrate_scale()

    # ------------------------------------------------------------- калибровка
    def _cursor(self) -> tuple[int, int] | None:
        """Положение настоящего курсора (нужен Hyprland)."""
        import subprocess

        if not shutil_which("hyprctl"):
            return None
        try:
            out = subprocess.run(
                ["hyprctl", "cursorpos"], capture_output=True, timeout=5, check=False
            ).stdout.decode()
            x, y = out.strip().split(",")
            return (int(x), int(y))
        except Exception:
            return None

    def _calibrate_scale(self) -> float:
        """Узнать, во сколько раз координаты устройства меньше экранных.

        Виртуальное устройство ydotool объявляет диапазон абсолютных
        координат, отличный от разрешения экрана (например, 960x540 при
        экране 1920x1080), поэтому координаты приходится пересчитывать.
        Масштаб измеряется по реакции курсора; замер повторяется, потому
        что курсор не всегда успевает доехать.
        """
        p1, p2 = (200, 200), (800, 400)
        ratios: list[float] = []
        for _ in range(3):
            c1 = self._measure(p1)
            c2 = self._measure(p2)
            if c1 is None or c2 is None:
                continue
            dx = abs(c2[0] - c1[0])
            dy = abs(c2[1] - c1[1])
            if dx < 50 or dy < 50:
                continue  # курсор не доехал — замер негодный
            ratios.append((dx / abs(p2[0] - p1[0]) + dy / abs(p2[1] - p1[1])) / 2)
        if not ratios:
            return 1.0
        ratios.sort()
        scale = ratios[len(ratios) // 2]
        if not (0.2 < scale < 10.0):
            return 1.0
        log.info("масштаб указателя ydotool: %.2f (координаты делятся на него)", scale)
        return scale

    def _measure(self, point: tuple[int, int]) -> tuple[int, int] | None:
        self._run("mousemove", "--absolute", str(point[0]), str(point[1]))
        time.sleep(0.6)
        return self._cursor()

    def _socket_path(self) -> str:
        return self._socket or os.path.expanduser("~/.ydotool_socket")

    def _ensure_daemon(self) -> None:
        """Поднять демон, если он не запущен."""
        import subprocess

        if self._probe():
            return
        # Сначала пробуем пользовательский юнит systemd.
        try:
            subprocess.run(
                ["systemctl", "--user", "start", "ydotool"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:  # pragma: no cover
            pass
        time.sleep(1.5)
        if self._probe():
            return
        # Юнит мог не подняться (нет группы input в этой сессии) — запускаем
        # демон напрямую: сокет и права на устройство у него те же.
        socket_path = self._socket_path()
        log.info("запускаю ydotoold напрямую (сокет %s)", socket_path)
        try:
            subprocess.Popen(
                ["ydotoold", f"--socket-path={socket_path}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"не удалось запустить ydotoold: {exc}") from exc
        time.sleep(1.5)
        if not self._probe():
            raise RuntimeError(
                "ydotoold не запустился: нет доступа к /dev/uinput. "
                "Нужно правило udev и членство в группе input "
                "(либо режим 0666 на устройстве)"
            )

    def _probe(self) -> bool:
        """Отвечает ли демон на сокет."""
        import subprocess

        if not os.path.exists(self._socket_path()):
            return False
        probe = subprocess.run(
            ["ydotool", "mousemove", "0", "0"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return probe.returncode == 0 and b"No such file" not in probe.stderr

    def _run(self, *args: str) -> None:
        import subprocess

        env = dict(os.environ)
        if self._socket:
            env["YDOTOOL_SOCKET"] = self._socket
        # ydotool печатает поток событий устройства в stdout — гасим его.
        subprocess.run(
            ["ydotool", *args],
            timeout=5,
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def click(self, x: int, y: int) -> None:
        """Кликнуть в экранных координатах."""
        scale = getattr(self, "scale", 1.0) or 1.0
        self._run("mousemove", "--absolute", str(int(x / scale)), str(int(y / scale)))
        time.sleep(0.05)
        self._run("click", "0xC0")
        time.sleep(0.03)
        self._run("click", "0x80")


def make_backend(allow_input: bool) -> InputBackend:
    """Выбрать реализацию ввода по конфигурации."""
    if not allow_input:
        return NullBackend()
    # На Wayland предпочтительнее своё устройство uinput: координаты
    # попадают точно. Дальше ydotool (если он есть) и pyautogui.
    for factory in (UinputMouseBackend, YdotoolBackend, PyAutoGuiBackend):
        try:
            backend = factory()
            log.warning("реальный ввод включён (%s) — бот будет кликать по экрану", factory.__name__)
            return backend
        except Exception as exc:  # pragma: no cover - зависит от окружения
            log.debug("ввод через %s недоступен: %s", factory.__name__, exc)
    log.warning("реальный ввод недоступен — остаюсь в режиме рекомендаций")
    return NullBackend()


class ActionExecutor:
    """Переводит решения планировщика в действия устройства ввода."""

    def __init__(self, backend: InputBackend, card_boxes: dict[str, ClickTarget] | None = None):
        self.backend = backend
        self.card_boxes = card_boxes or {}
        self.history: list[tuple[float, str, str]] = []

    def set_card_boxes(self, boxes: dict[str, ClickTarget]) -> None:
        self.card_boxes = boxes

    def execute(self, action: Action, lane_target: ClickTarget | None = None) -> bool:
        """Выполнить действие. Возвращает True, если действие отправлено."""
        if action.kind == "spawn" and action.unit_kind:
            card = self.card_boxes.get(action.unit_kind)
            if card is None:
                log.debug("нет координат карточки для %s", action.unit_kind)
                return False
            self.backend.click(card.x, card.y)
            if lane_target is not None:
                self.backend.click(lane_target.x, lane_target.y)
            self._record(action, f"клик по карточке {action.unit_kind} и точке {lane_target.label if lane_target else '—'}")
            return True
        if action.kind == "focus":
            self._record(action, f"фокус на юните #{action.target_uid}")
            return True
        return False

    def _record(self, action: Action, text: str) -> None:
        import time

        self.history.append((time.time(), action.kind, text))
        log.info("%s", text)
