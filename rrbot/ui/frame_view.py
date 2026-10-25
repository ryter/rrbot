"""Виджет отображения кадра с наложениями (детекции, решения, зоны)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..models import Action, Detection, GameState, Team

COLORS = {
    Team.ALLY: QColor(80, 170, 255),
    Team.ENEMY: QColor(255, 90, 90),
}
UNKNOWN = QColor(240, 210, 90)
LANE = QColor(120, 255, 140)


def numpy_to_qimage(frame: np.ndarray) -> QImage:
    """Конвертировать BGR-кадр в QImage без копирования данных (read-only)."""
    if frame is None or frame.size == 0:
        return QImage()
    if frame.ndim == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class FrameView(QWidget):
    """Показывает кадр и результаты работы бота."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._detections: list[Detection] = []
        self._state: GameState | None = None
        self._action: Action | None = None
        self._scale = 1.0
        self.setMinimumSize(320, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)

    # ------------------------------------------------------------- содержимое
    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._image = None
        else:
            self._image = numpy_to_qimage(frame)
        self.update()

    def set_results(
        self,
        detections: list[Detection] | None = None,
        state: GameState | None = None,
        action: Action | None = None,
    ) -> None:
        if detections is not None:
            self._detections = detections
        if state is not None:
            self._state = state
        if action is not None:
            self._action = action
        self.update()

    def clear(self) -> None:
        self._image = None
        self._detections = []
        self._state = None
        self._action = None
        self.update()

    # ------------------------------------------------------------- отрисовка
    def paintEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))

        if self._image is None or self._image.isNull():
            painter.setPen(QColor(160, 160, 170))
            painter.drawText(self.rect(), Qt.AlignCenter, "Нет кадра")
            return

        # Вписываем кадр в виджет с сохранением пропорций.
        iw, ih = self._image.width(), self._image.height()
        avail_w, avail_h = self.width() - 8, self.height() - 8
        scale = min(avail_w / iw, avail_h / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox = (self.width() - dw) // 2
        oy = (self.height() - dh) // 2
        self._scale = scale
        self._origin = (ox, oy)
        self._display_size = (dw, dh)

        painter.drawImage(QRect(ox, oy, dw, dh), self._image)

        # Сетка канонического поля (если известна разметка).
        self._draw_field_grid(painter, ox, oy, dw, dh)
        self._draw_lanes(painter)
        self._draw_detections(painter)
        self._draw_action(painter)
        self._draw_hud(painter, ox, oy, dw, dh)

    def _to_screen(self, x: float, y: float) -> QPoint:
        ox, oy = getattr(self, "_origin", (0, 0))
        return QPoint(int(ox + x * self._scale), int(oy + y * self._scale))

    def _draw_field_grid(self, painter: QPainter, ox: int, oy: int, dw: int, dh: int) -> None:
        state = self._state
        if state is None or not state.spawn_points:
            return
        pen = QPen(QColor(255, 255, 255, 40))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        # Иллюстрируем полосу поля: границы между нижней и верхней половиной.
        mid = self._to_screen(0, state.spawn_points[next(iter(state.spawn_points))].y)
        painter.drawLine(ox, mid.y(), ox + dw, mid.y())

    def _draw_lanes(self, painter: QPainter) -> None:
        state = self._state
        if state is None:
            return
        for lane in state.spawn_points.values():
            p = self._to_screen(lane.x, lane.y)
            painter.setPen(QPen(LANE, 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(120, 255, 140, 60)))
            painter.drawEllipse(p, 10, 10)

    def _draw_detections(self, painter: QPainter) -> None:
        for det in self._detections:
            x, y, w, h = det.box
            top_left = self._to_screen(x, y)
            color = COLORS.get(det.team, UNKNOWN) if det.team else UNKNOWN
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            rect = QRect(top_left, self._to_screen(x + w, y + h))
            painter.drawRect(rect)

            label = f"{det.name} {det.score:.2f}"
            if det.team:
                label = f"{'A' if det.team is Team.ALLY else 'E'} {label}"
            painter.setPen(color)
            painter.drawText(rect.topLeft().x() + 2, rect.topLeft().y() - 3, label)

    def _draw_action(self, painter: QPainter) -> None:
        action = self._action
        if action is None or action.kind != "spawn" or action.position is None:
            return
        p = self._to_screen(*action.position)
        painter.setPen(QPen(QColor(255, 230, 120), 3))
        painter.setBrush(QBrush(QColor(255, 230, 120, 70)))
        painter.drawEllipse(p, 14, 14)
        painter.drawLine(p.x() - 20, p.y(), p.x() + 20, p.y())
        painter.drawLine(p.x(), p.y() - 20, p.x(), p.y() + 20)

    def _draw_hud(self, painter: QPainter, ox: int, oy: int, dw: int, dh: int) -> None:
        state = self._state
        if state is None:
            return
        text = (
            f"t={state.time:5.1f}  мана={state.mana:6.1f}/{state.mana_max:.0f}  "
            f"своих={len(state.ally_units)}  врагов={len(state.enemy_units)}  "
            f"базы: {state.our_base_hp:.0%}/{state.enemy_base_hp:.0%}"
        )
        painter.setPen(QColor(230, 230, 235))
        painter.fillRect(QRect(ox, oy + dh - 26, dw, 26), QColor(0, 0, 0, 150))
        painter.drawText(QRect(ox + 6, oy + dh - 26, dw - 12, 26), Qt.AlignVCenter | Qt.AlignLeft, text)
