#!/usr/bin/env python3
"""Калибровка поля для реальной игры.

Делает снимок экрана, показывает его в окне и просит отметить 4 угла
игрового поля (левый верхний, правый верхний, правый нижний, левый нижний).
Результат сохраняется в JSON и подставляется в конфиг.

Запуск:
    python tools/calibrate.py --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter, QPen, QColor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rrbot.capture import make_screen_source  # noqa: E402
from rrbot.geometry import Calibration, order_quad  # noqa: E402

CORNER_NAMES = [
    "Левый верхний",
    "Правый верхний",
    "Правый нижний",
    "Левый нижний",
]


class Calibrator(QWidget):
    """Окно со снимком экрана: клики расставляют углы поля."""

    def __init__(self, frame: np.ndarray, offset: tuple[int, int]):
        super().__init__()
        self.frame = frame
        self.offset = offset
        self.points: list[tuple[int, int]] = []
        self.setWindowTitle("Калибровка поля — отметьте 4 угла")
        self.resize(1100, 900)

        layout = QVBoxLayout(self)
        self.view = ImageView(self)
        layout.addWidget(self.view, 1)

        self.status = QLabel("Отметьте 4 угла игрового поля по порядку: ЛВ → ПВ → ПН → ЛН")
        layout.addWidget(self.status)

        buttons = QVBoxLayout()
        undo = QPushButton("Отменить точку")
        undo.clicked.connect(self.undo)
        buttons.addWidget(undo)
        layout.addLayout(buttons)

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.view.points = list(self.points)
            self.view.update()

    def add_point(self, x: int, y: int) -> None:
        if len(self.points) >= 4:
            self.points.pop(0)
        self.points.append((x, y))
        self.view.points = list(self.points)
        self.view.update()
        self.status.setText(
            f"Отмечено {len(self.points)}/4. Следующая точка: {CORNER_NAMES[len(self.points) - 1]}"
            if self.points
            else "Отметьте 4 угла игрового поля по порядку: ЛВ → ПВ → ПН → ЛН"
        )


class ImageView(QWidget):
    """Отображение снимка с нарисованными углами."""

    def __init__(self, parent: Calibrator):
        super().__init__(parent)
        self.calibrator = parent
        self.points: list[tuple[int, int]] = []
        self.scale = 1.0
        self.origin = (0, 0)
        h, w = parent.frame.shape[:2]
        self.image = QImage(
            np.ascontiguousarray(parent.frame[:, :, ::-1]).data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888,
        ).copy()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 24))
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox = (self.width() - dw) // 2
        oy = (self.height() - dh) // 2
        self.scale, self.origin = scale, (ox, oy)
        painter.drawImage(QRect(ox, oy, dw, dh), self.image)

        if self.points:
            painter.setPen(QPen(QColor(255, 230, 100), 3))
            for i, (x, y) in enumerate(self.points):
                px = ox + int(x * scale)
                py = oy + int(y * scale)
                painter.drawEllipse(QPoint(px, py), 8, 8)
                painter.drawText(px + 12, py - 8, f"{i + 1}")
            painter.setPen(QPen(QColor(120, 255, 160), 2, Qt.DashLine))
            poly = [(ox + int(x * scale), oy + int(y * scale)) for x, y in self.points]
            if len(poly) >= 2:
                for i in range(len(poly) - 1):
                    painter.drawLine(poly[i][0], poly[i][1], poly[i + 1][0], poly[i + 1][1])

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        ox, oy = self.origin
        fx = (pos.x() - ox) / self.scale
        fy = (pos.y() - oy) / self.scale
        if fx < 0 or fy < 0 or fx >= self.image.width() or fy >= self.image.height():
            return
        self.calibrator.add_point(int(fx), int(fy))


from PySide6.QtCore import QRect  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка игрового поля")
    parser.add_argument("--out", default=str(ROOT / "calibration.json"))
    parser.add_argument("--region", default="0,0,1920,1080", help="x,y,w,h области захвата")
    parser.add_argument("--board", default="360,640", help="W,H канонического поля")
    args = parser.parse_args()

    x, y, w, h = (int(v) for v in args.region.split(","))
    source = make_screen_source((x, y, w, h))
    frame = source.grab()
    if frame is None:
        print("Не удалось получить кадр экрана", file=sys.stderr)
        return 1
    print(f"Снимок экрана: {frame.shape[1]}x{frame.shape[0]}")

    app = QApplication(sys.argv)
    cal = Calibrator(frame, (x, y))
    cal.show()
    app.exec()

    if len(cal.points) != 4:
        print("Калибровка не завершена", file=sys.stderr)
        return 1

    quad = order_quad(cal.points)
    bw, bh = (int(v) for v in args.board.split(","))
    calibration = Calibration(quad=quad, board_width=bw, board_height=bh)
    Path(args.out).write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    print(f"Сохранено в {args.out}")
    print(json.dumps(calibration.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
