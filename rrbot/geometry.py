"""Геометрия: калибровка поля и гомография «экран <-> канонические координаты».

Идея: детектирование всегда идёт в каноническом пространстве (например,
512x288). Так шаблоны не зависят от разрешения экрана, размера окна и
положения игры на мониторе.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass
class Calibration:
    """Калибровка: углы поля на экране и размер канонического поля."""

    quad: Quad = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    board_width: int = 512
    board_height: int = 288

    def is_set(self) -> bool:
        return all(pt != (0.0, 0.0) for pt in self.quad)

    def to_dict(self) -> dict:
        return {
            "quad": [list(p) for p in self.quad],
            "board_width": self.board_width,
            "board_height": self.board_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        quad = tuple((float(p[0]), float(p[1])) for p in data["quad"])  # type: ignore[misc]
        return cls(
            quad=quad,  # type: ignore[arg-type]
            board_width=int(data.get("board_width", 512)),
            board_height=int(data.get("board_height", 288)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def fullscreen(cls, width: int, height: int, board_width: int = 512, board_height: int = 288) -> "Calibration":
        return cls(
            quad=((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))),
            board_width=board_width,
            board_height=board_height,
        )


def homography(cal: Calibration) -> np.ndarray:
    """Матрица 3x3: экран -> каноническое поле."""
    src = np.asarray(cal.quad, dtype=np.float32)
    dst = np.asarray(
        [
            (0.0, 0.0),
            (float(cal.board_width - 1), 0.0),
            (float(cal.board_width - 1), float(cal.board_height - 1)),
            (0.0, float(cal.board_height - 1)),
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp_to_board(frame: np.ndarray, cal: Calibration) -> np.ndarray:
    """Привести кадр экрана к каноническому виду поля."""
    h = homography(cal)
    return cv2.warpPerspective(
        frame,
        h,
        (cal.board_width, cal.board_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def board_to_screen(points: Iterable[Point], cal: Calibration) -> list[Point]:
    """Перевести точки из канонических координат в экранные."""
    h = homography(cal)
    h_inv = np.linalg.inv(h)
    pts = np.asarray([list(p) for p in points], dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def screen_to_board(points: Iterable[Point], cal: Calibration) -> list[Point]:
    """Перевести точки из экранных координат в канонические."""
    h = homography(cal)
    pts = np.asarray([list(p) for p in points], dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def order_quad(points: Sequence[Point]) -> Quad:
    """Упорядочить 4 точки: левый верхний, правый верхний, правый нижний, левый нижний."""
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return (
        tuple(pts[np.argmin(s)]),  # type: ignore[return-value]
        tuple(pts[np.argmin(d)]),  # type: ignore[return-value]
        tuple(pts[np.argmax(s)]),  # type: ignore[return-value]
        tuple(pts[np.argmax(d)]),  # type: ignore[return-value]
    )
