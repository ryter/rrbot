"""Захват кадров: с экрана (X11 или Wayland) и из симулятора.

Важная деталь для Wayland: ``mss`` захватывает окна через XWayland и на
чистом Wayland-сеансе отдаёт чёрные кадры. Поэтому в режиме Wayland
используется ``grim`` (Hyprland/wlroots) — он снимает реальный кадр
композитора.

Оба бэкенда возвращают BGR-кадр (numpy), как и ожидает остальной код.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    """Источник кадров."""

    def grab(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


def session_type() -> str:
    """Тип графической сессии: ``wayland`` или ``x11``."""
    return os.environ.get("XDG_SESSION_TYPE", "x11").lower()


def _read_png(path: str) -> np.ndarray | None:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


class GrimSource:
    """Захват области экрана через grim (Wayland/wlroots)."""

    available = True

    def __init__(self, region: tuple[int, int, int, int]):
        self.region = region
        self._tmp = Path(tempfile.mkdtemp(prefix="rrbot-grim-"))

    def grab(self) -> np.ndarray | None:
        x, y, w, h = self.region
        if w <= 0 or h <= 0:
            return None
        out = self._tmp / "frame.png"
        geom = f"{x},{y} {w}x{h}"
        try:
            res = subprocess.run(
                ["grim", "-g", geom, str(out)],
                capture_output=True,
                timeout=15.0,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.error("grim не отработал: %s", exc)
            return None
        if res.returncode != 0:
            log.error("grim вернул %s: %s", res.returncode, res.stderr[:200])
            return None
        return _read_png(str(out))

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class MssSource:
    """Захват области экрана через mss (X11)."""

    available = True

    def __init__(self, region: tuple[int, int, int, int], monitor: int = 1):
        self.region = region
        self.monitor = monitor
        self._sct = None
        try:
            import mss  # type: ignore[import-not-found]

            self._sct = mss.MSS()
        except Exception as exc:  # pragma: no cover - зависит от окружения
            log.error("не удалось инициализировать mss: %s", exc)
            self._sct = None

    def grab(self) -> np.ndarray | None:
        if self._sct is None:
            return None
        try:
            raw = self._sct.grab(self.region)
        except Exception as exc:  # pragma: no cover
            log.error("ошибка захвата экрана: %s", exc)
            return None
        frame = np.asarray(raw)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


class WindowSource:
    """Захват окна по заголовку (Hyprland) с автоматическим поиском окна.

    Полезно для работы с игрой в браузере: не нужно вручную задавать
    координаты окна — источник сам следит за его положением и размером.
    """

    available = True

    def __init__(self, title_substring: str, region: tuple[int, int, int, int] | None = None):
        self.title_substring = title_substring
        self.fallback_region = region
        self.region = region or (0, 0, 1920, 1080)
        self._grim = GrimSource(self.region)

    def find_window(self) -> tuple[int, int, int, int] | None:
        """Найти окно через hyprctl и вернуть (x, y, w, h)."""
        if not shutil.which("hyprctl"):
            return None
        try:
            res = subprocess.run(
                ["hyprctl", "clients", "-j"],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            import json

            clients = json.loads(res.stdout or b"[]")
        except Exception as exc:  # pragma: no cover
            log.debug("hyprctl clients не отработал: %s", exc)
            return None
        for c in clients:
            title = (c.get("title") or "")
            if self.title_substring.lower() in title.lower():
                at = c.get("at", [0, 0])
                size = c.get("size", [0, 0])
                return (int(at[0]), int(at[1]), int(size[0]), int(size[1]))
        return None

    def grab(self) -> np.ndarray | None:
        found = self.find_window()
        if found is not None and found != self.region:
            self.region = found
            self._grim = GrimSource(self.region)
        return self._grim.grab()

    def close(self) -> None:
        self._grim.close()


def make_screen_source(
    region: tuple[int, int, int, int],
    monitor: int = 1,
    window_title: str | None = None,
) -> FrameSource:
    """Выбрать рабочий бэкенд захвата для текущей сессии."""
    if window_title:
        log.info("захват экрана: окно «%s» (grim)", window_title)
        return WindowSource(window_title, region)
    if session_type() == "wayland" and shutil.which("grim"):
        log.info("захват экрана: grim (Wayland)")
        return GrimSource(region)
    if shutil.which("grim"):
        log.info("захват экрана: grim")
        return GrimSource(region)
    log.info("захват экрана: mss (X11)")
    return MssSource(region, monitor)


class SimSource:
    """Источник из симулятора: мир прокручивается и рисуется."""

    def __init__(self, world, renderer, dt: float, available_fn=None):
        self.world = world
        self.renderer = renderer
        self.dt = dt
        self.available_fn = available_fn

    def grab(self) -> np.ndarray | None:
        self.world.step(self.dt)
        available = self.available_fn() if self.available_fn else None
        return self.renderer.render(self.world, available)

    def close(self) -> None:
        return None


class StaticSource:
    """Один и тот же кадр — удобно для отладки интерфейса и тестов."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def grab(self) -> np.ndarray | None:
        return self.frame

    def close(self) -> None:
        return None
