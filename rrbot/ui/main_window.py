"""Главное окно исследовательского бота."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..capture import SimSource, make_screen_source
from ..config import AppConfig, load_config
from ..geometry import Calibration
from ..input import ActionExecutor, make_backend
from ..models import Action, GameState
from ..pipeline import BotPipeline, FrameResult
from ..sim import SimRenderer, SimWorld
from .frame_view import FrameView

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


class Worker(QObject):
    """Фоновый поток: крутит конвейер и отдаёт результаты в GUI."""

    result_ready = Signal(object)
    finished = Signal()

    def __init__(self, pipeline: BotPipeline):
        super().__init__()
        self.pipeline = pipeline
        self._running = False

    def start(self) -> None:
        """Разрешить цикл обработки. Вызывается до запуска потока."""
        self._running = True

    @Slot()
    def run(self) -> None:
        while self._running:
            try:
                result = self._run_once()
            except Exception:  # noqa: BLE001 - поток не должен падать
                log.exception("ошибка в конвейере")
                self._running = False
                break
            self.result_ready.emit(result)
        self.finished.emit()

    def _run_once(self) -> FrameResult:
        """Поддержать и обычный конвейер, и живую игру."""
        target = self.pipeline
        if hasattr(target, "process_once"):
            return target.process_once()
        step = target.step()
        return FrameResult(
            frame=step.canvas,
            board=step.field,
            state=step.state,
            action=step.action,
            timestamp=time.time(),
            detections=list(getattr(step.state, "detections", []) or []),
        )

    def stop(self) -> None:
        self._running = False


class MainWindow(QMainWindow):
    """Окно с превью, настройками и журналом решений."""

    def __init__(self, config: AppConfig | None = None):
        super().__init__()
        self.config = config or load_config()
        self.setWindowTitle("Rush Royale — исследовательский бот")
        self.resize(1180, 820)

        self.pipeline: BotPipeline | None = None
        self.worker: Worker | None = None
        self.thread: QThread | None = None
        self._log_rows: list[tuple[str, str, str]] = []
        self._stats = {"fps": 0.0, "latency": 0.0, "detections": 0}

        self._build_ui()
        self._apply_config_to_ui()
        self._start_simulation()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # --- левая часть: превью и параметры
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.view = FrameView()
        left_layout.addWidget(self.view, 1)

        controls = QGroupBox("Управление")
        grid = QHBoxLayout(controls)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Симулятор", "sim")
        self.source_combo.addItem("Живая игра (браузер)", "live")
        self.source_combo.addItem("Захват экрана", "screen")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        grid.addWidget(QLabel("Источник:"))
        grid.addWidget(self.source_combo)

        self.play_button = QPushButton("Старт")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        grid.addWidget(self.play_button)

        self.step_button = QPushButton("Шаг")
        self.step_button.clicked.connect(self._on_step)
        grid.addWidget(self.step_button)

        self.reset_button = QPushButton("Сброс")
        self.reset_button.clicked.connect(self._on_reset)
        grid.addWidget(self.reset_button)

        self.allow_input = QCheckBox("Реальные клики")
        self.allow_input.setToolTip(
            "Выключено по умолчанию: бот только показывает рекомендации.\n"
            "Включайте на своём устройстве и в своей копии игры."
        )
        grid.addWidget(self.allow_input)
        grid.addStretch(1)
        left_layout.addWidget(controls)

        splitter.addWidget(left)

        # --- правая часть: вкладки настроек и журнала
        right = QWidget()
        right_layout = QVBoxLayout(right)
        tabs = QTabWidget()

        # Вкладка: зрение
        vision_box = QGroupBox("Зрение")
        vision_form = QFormLayout(vision_box)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.99)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        vision_form.addRow("Порог совпадения", self.threshold_spin)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.05, 0.9)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setDecimals(2)
        self.nms_spin.valueChanged.connect(self._on_nms_changed)
        vision_form.addRow("NMS (IoU)", self.nms_spin)
        tabs.addTab(vision_box, "Зрение")

        # Вкладка: планировщик
        planner_box = QGroupBox("Веса полезности")
        planner_form = QFormLayout(planner_box)
        self.weight_spins: dict[str, QDoubleSpinBox] = {}
        for key, label in (
            ("weight_counter", "Контр-счётство"),
            ("weight_pressure", "Давление"),
            ("weight_synergy", "Синергия"),
            ("weight_cost_efficiency", "Эффективность"),
            ("weight_spawn_value", "Доступность"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(-10.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.valueChanged.connect(self._on_weight_changed)
            self.weight_spins[key] = spin
            planner_form.addRow(label, spin)
        # Метрика эффективности по мане: какой отдачи ждём от юнита.
        self.cost_metric_combo = QComboBox()
        self.cost_metric_combo.addItem("живучесть / цена", "hp")
        self.cost_metric_combo.addItem("живучесть × урон / цена", "hp_dps")
        self.cost_metric_combo.currentIndexChanged.connect(self._on_weight_changed)
        planner_form.addRow("Эффективность по мане", self.cost_metric_combo)
        tabs.addTab(planner_box, "Планировщик")

        # Вкладка: журнал решений
        log_box = QGroupBox("Журнал")
        log_layout = QVBoxLayout(log_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Время", "Действие", "Обоснование", "Оценка"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        log_layout.addWidget(self.table)
        tabs.addTab(log_box, "Решения")

        # Вкладка: журнал сообщений
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setStyleSheet("background:#1b1b1f; color:#d6d6dd; font-family: monospace;")
        tabs.addTab(self.messages, "Сообщения")

        right_layout.addWidget(tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.status_label = QLabel("Готово")
        self.statusBar().addPermanentWidget(self.status_label)

        # Горячие клавиши.
        toggle = QAction("Пуск/пауза", self)
        toggle.setShortcut(QKeySequence("Space"))
        toggle.triggered.connect(self._on_play_toggled)
        self.addAction(toggle)

    # -------------------------------------------------------------- конфиг
    def _apply_config_to_ui(self) -> None:
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(self.config.vision.match_threshold)
        self.threshold_spin.blockSignals(False)

        self.nms_spin.blockSignals(True)
        self.nms_spin.setValue(self.config.vision.nms_iou)
        self.nms_spin.blockSignals(False)

        for key, spin in self.weight_spins.items():
            spin.blockSignals(True)
            spin.setValue(float(getattr(self.config.planner, key)))
            spin.blockSignals(False)

        idx = self.cost_metric_combo.findData(self.config.planner.cost_metric)
        if idx >= 0:
            self.cost_metric_combo.blockSignals(True)
            self.cost_metric_combo.setCurrentIndex(idx)
            self.cost_metric_combo.blockSignals(False)

        idx = self.source_combo.findData(self.config.capture.source)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)

    # ---------------------------------------------------------- симуляция
    def _make_calibration(self, source: str) -> Calibration:
        if source == "sim":
            w = float(self.config.sim.width)
            h = float(self.config.sim.height)
            return Calibration(
                quad=((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)),
                board_width=self.config.sim.width,
                board_height=self.config.sim.height,
            )
        x, y, w, h = self.config.capture.region
        return Calibration(
            quad=((float(x), float(y)), (float(x + w), float(y)),
                  (float(x + w), float(y + h)), (float(x), float(y + h))),
            board_width=self.config.sim.width,
            board_height=self.config.sim.height,
        )

    def _start_simulation(self) -> None:
        self._stop_thread()
        specs = self.config.vision.specs()
        self.world = SimWorld(self.config.sim, specs, seed=self.config.sim.seed)
        self.renderer = SimRenderer(self.config.sim, specs)
        source = SimSource(
            self.world,
            self.renderer,
            self.config.sim.dt,
            available_fn=lambda: self.world.affordable_kinds(),
        )
        backend = make_backend(self.config.allow_input or self.allow_input.isChecked())
        self.executor = ActionExecutor(backend)
        self.pipeline = BotPipeline(self.config, source, self._make_calibration("sim"), self.executor)
        self.pipeline.sync_card_boxes(self.renderer)
        self._start_thread()

    def _start_thread(self) -> None:
        if self.pipeline is None:
            return
        self.thread = QThread(self)
        self.worker = Worker(self.pipeline)
        self.worker.moveToThread(self.thread)
        self.worker.result_ready.connect(self._on_result)
        self.worker.finished.connect(self._thread_finished)
        self.thread.started.connect(self.worker.run)
        self.worker.start()
        self.thread.start()

    def _stop_thread(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            if not self.thread.wait(3000):
                # Кадр не успевает обработаться — останавливаем принудительно,
                # иначе Qt разрушит поток во время работы.
                log.warning("поток не остановился за 3 с — принудительная остановка")
                self.thread.terminate()
                self.thread.wait(1000)
        self.worker = None
        self.thread = None

    # ------------------------------------------------------------- события
    def _on_source_changed(self) -> None:
        source = self.source_combo.currentData()
        if source == "sim":
            self._start_simulation()
        elif source == "live":
            self._start_live()
        else:
            self._start_screen_capture()

    def _start_live(self) -> None:
        """Подключиться к живой игре в браузере."""
        self._stop_thread()
        try:
            from ..live import LiveBot, LiveGame, make_clicker
        except Exception as exc:  # pragma: no cover
            self._log(f"живой режим недоступен: {exc}")
            return

        game = LiveGame()
        clicker = (
            make_clicker(game=game)
            if (self.config.allow_input or self.allow_input.isChecked())
            else None
        )
        self.live_game = game
        self.live_bot = LiveBot(self.config, game, clicker=clicker)
        self.pipeline = self.live_bot
        self._log(f"живой режим: {game.describe()}")
        self._log("ввод: только рекомендации (включите чекбокс, чтобы кликать)")
        self._start_thread()

    def _start_screen_capture(self) -> None:
        self._stop_thread()
        region = tuple(self.config.capture.region)
        source = make_screen_source(region, self.config.capture.monitor)
        backend = make_backend(self.config.allow_input or self.allow_input.isChecked())
        self.executor = ActionExecutor(backend)
        self.pipeline = BotPipeline(self.config, source, self._make_calibration("screen"), self.executor)
        self._start_thread()

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            if self.thread is None:
                self._start_thread()
            self.thread.start() if self.thread is not None else None
            self.play_button.setText("Пауза")
        else:
            if self.worker is not None:
                self.worker.stop()
            if self.thread is not None:
                self.thread.quit()
            self.play_button.setText("Старт")

    def _on_step(self) -> None:
        if self.pipeline is None:
            return
        result = self.pipeline.process_once()
        self._on_result(result)

    def _on_reset(self) -> None:
        self._start_simulation()
        self.play_button.setChecked(True)

    def _thread_finished(self) -> None:
        pass

    def _on_result(self, result: FrameResult) -> None:
        if not result.ok:
            return
        state = result.state
        assert state is not None
        self.view.set_frame(result.frame)
        self.view.set_results(detections=result.detections, state=state, action=result.action)

        if result.action is not None and result.action.kind != "wait":
            self._append_log(state, result.action)

        now = time.time()
        self._stats["fps"] = self._stats.get("fps", 0.0) * 0.9 + (1000.0 / max(result.duration_ms, 1e-3)) * 0.1
        self._stats["latency"] = result.duration_ms
        self._stats["detections"] = len(result.detections)
        self.status_label.setText(
            f"fps≈{self._stats['fps']:.1f}  |  кадр {result.duration_ms:.1f} мс  |  "
            f"детекций {self._stats['detections']}  |  t={state.time:.1f}с"
        )

    def _append_log(self, state: GameState, action: Action) -> None:
        self._log_rows.append((f"{state.time:.2f}", action.kind, action.reason, f"{action.score:.2f}"))
        if len(self._log_rows) > 500:
            self._log_rows.pop(0)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate(self._log_rows[-1]):
            self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.scrollToBottom()
        if action.kind == "spawn":
            self._log.append(f"[{state.time:6.2f}] спавн {action.unit_kind} -> {action.lane}: {action.reason}")
        else:
            self._log.append(f"[{state.time:6.2f}] {action.describe()}")

    def _log(self, message: str) -> None:
        self.messages.append(message)

    # ------------------------------------------------------------ настройки
    def _on_threshold_changed(self, value: float) -> None:
        self.config.vision.match_threshold = value
        if self.pipeline is not None:
            self.pipeline.vision.vision.match_threshold = value

    def _on_nms_changed(self, value: float) -> None:
        self.config.vision.nms_iou = value
        if self.pipeline is not None:
            self.pipeline.vision.vision.nms_iou = value

    def _on_weight_changed(self) -> None:
        for key, spin in self.weight_spins.items():
            setattr(self.config.planner, key, spin.value())
        self.config.planner.cost_metric = str(self.cost_metric_combo.currentData())
        if self.pipeline is not None:
            self.pipeline.planner.weights = self.config.planner

    def closeEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        self._stop_thread()
        super().closeEvent(event)
