# UPDATE_FILE: whisker/gui/workflows/behavior_classification/widgets/behavior_labeling/media_viewer.py
#
# A focused video player for behavior hand-annotation. It mirrors the full
# WHISKER MediaViewerWidget's playback engine (Qt Multimedia QMediaPlayer +
# QGraphicsVideoItem, with OpenCV used only to read frame-rate/frame-count
# metadata) but drops the pose/detection overlays and side-by-side comparison
# that are irrelevant to labeling behavior bouts.
import logging
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QRectF, QTimer
from PyQt6.QtGui import QPainter, QResizeEvent
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QStyle,
    QSpinBox, QFormLayout, QDoubleSpinBox, QStyleOptionSlider, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsProxyWidget, QStackedWidget,
)

from whisker.gui.constants import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
from whisker.gui.widgets.info_overlay import InfoOverlay


class ClickableSlider(QSlider):
    """Standard QSlider that also jumps to the clicked position on the groove."""

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        hit = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_Slider, opt, event.pos(), self
        )

        if hit == QStyle.SubControl.SC_SliderGroove:
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderGroove, self
            )
            if self.orientation() == Qt.Orientation.Horizontal:
                ratio = (event.position().x() - groove.x()) / groove.width()
            else:
                ratio = (groove.height() - (event.position().y() - groove.y())) / groove.height()

            ratio = max(0.0, min(1.0, ratio))
            val = self.minimum() + (self.maximum() - self.minimum()) * ratio
            self.setValue(int(val))
            self.sliderMoved.emit(int(val))
            event.accept()
        else:
            super().mousePressEvent(event)


class MediaControls(QWidget):
    """Transport bar: slider, play/step/skip buttons, time/frame labels, speed."""

    play_toggled = pyqtSignal()
    seek_requested = pyqtSignal(int)       # ms
    step_requested = pyqtSignal(int)       # +1 / -1 frames
    skip_requested = pyqtSignal(int)       # +/- N frames
    goto_frame_requested = pyqtSignal(int)
    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 20, 5)

        self.slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.seek_requested.emit)
        layout.addWidget(self.slider)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(15)

        btns = QHBoxLayout()
        btns.setSpacing(2)

        self.btn_step_back = self._icon_btn(QStyle.StandardPixmap.SP_MediaSeekBackward, "Step Back")
        self.btn_step_back.clicked.connect(lambda: self.step_requested.emit(-1))

        self.btn_play = self._icon_btn(QStyle.StandardPixmap.SP_MediaPlay, "Play/Pause")
        self.btn_play.clicked.connect(self.play_toggled.emit)

        self.btn_step_fwd = self._icon_btn(QStyle.StandardPixmap.SP_MediaSeekForward, "Step Forward")
        self.btn_step_fwd.clicked.connect(lambda: self.step_requested.emit(1))

        btns.addWidget(self.btn_step_back)
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_step_fwd)
        btns.addSpacing(10)

        self.btn_skip_back = QPushButton("<<")
        self.btn_skip_back.clicked.connect(lambda: self.skip_requested.emit(-self.spin_step.value()))
        self.btn_skip_fwd = QPushButton(">>")
        self.btn_skip_fwd.clicked.connect(lambda: self.skip_requested.emit(self.spin_step.value()))

        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 10000)
        self.spin_step.setValue(10)
        self.spin_step.setToolTip("Frame skip size")

        btns.addWidget(self.btn_skip_back)
        btns.addWidget(self.btn_skip_fwd)
        btns.addWidget(self.spin_step)

        ctrl_layout.addLayout(btns)
        ctrl_layout.addStretch()

        info = QVBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_frame = QLabel("Frame: 0 / 0")
        info.addWidget(self.lbl_time)
        info.addWidget(self.lbl_frame)
        ctrl_layout.addLayout(info)

        form = QFormLayout()
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.valueChanged.connect(self.speed_changed.emit)
        form.addRow("Speed:", self.spin_speed)

        goto_box = QHBoxLayout()
        self.spin_goto = QSpinBox()
        self.btn_goto = QPushButton("Go")
        self.btn_goto.clicked.connect(lambda: self.goto_frame_requested.emit(self.spin_goto.value()))
        goto_box.addWidget(self.spin_goto)
        goto_box.addWidget(self.btn_goto)
        form.addRow("Go to:", goto_box)

        ctrl_layout.addLayout(form)
        layout.addLayout(ctrl_layout)

        self._interactive_widgets = [
            self.slider, self.btn_step_back, self.btn_step_fwd, self.btn_play,
            self.btn_skip_back, self.btn_skip_fwd, self.spin_step,
            self.spin_speed, self.spin_goto, self.btn_goto,
        ]

    def _icon_btn(self, icon, tip):
        b = QPushButton()
        b.setIcon(self.style().standardIcon(icon))
        b.setToolTip(tip)
        return b

    def set_controls_enabled(self, enabled: bool):
        for w in self._interactive_widgets:
            w.setEnabled(enabled)
        if not enabled:
            self.lbl_time.setText("00:00 / 00:00")
            self.lbl_frame.setText("Frame: 0 / 0")
        self.setVisible(enabled)

    def set_playing_state(self, is_playing: bool):
        icon = (QStyle.StandardPixmap.SP_MediaPause if is_playing
                else QStyle.StandardPixmap.SP_MediaPlay)
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def update_time(self, current_ms: int, duration_ms: int):
        self.lbl_time.setText(f"{self._fmt(current_ms)} / {self._fmt(duration_ms)}")
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(current_ms)
            self.slider.blockSignals(False)

    def update_frames(self, current_f: int, total_f: int):
        self.lbl_frame.setText(f"Frame: {current_f} / {max(0, total_f - 1)}")

    def set_duration(self, ms: int):
        self.slider.setRange(0, ms)

    def set_frame_limit(self, total_frames: int):
        self.spin_goto.setRange(0, max(0, total_frames - 1))

    def _fmt(self, ms: int):
        s = ms // 1000
        return f"{s // 60:02}:{s % 60:02}"


class MediaViewerWidget(QWidget):
    """Plays a single video and exposes a frame-accurate annotation API."""

    frame_changed = pyqtSignal(int)
    DEFAULT_OVERLAY_FONT_SIZE = InfoOverlay.DEFAULT_FONT_SIZE

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._media_player = QMediaPlayer()
        self._fps = 0.0
        self._total_frames = 0
        self._duration_ms = 0
        self._is_video_loaded = False

        self._setup_ui()
        self.set_media(None)
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.view_stack = QStackedWidget()
        layout.addWidget(self.view_stack)

        self.placeholder = QLabel("Select a video file to view")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_stack.addWidget(self.placeholder)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("border: none; background: #222;")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.info_overlay = InfoOverlay()
        self.overlay_proxy = QGraphicsProxyWidget()
        self.overlay_proxy.setWidget(self.info_overlay)
        self.overlay_proxy.setZValue(1)
        self.scene.addItem(self.overlay_proxy)

        self.controls = MediaControls()

        vbox.addWidget(self.view, 1)
        vbox.addWidget(self.controls)
        self.view_stack.addWidget(container)

    def _connect_signals(self):
        self._media_player.positionChanged.connect(self._on_position_changed)
        self._media_player.durationChanged.connect(self._on_duration_changed)
        self._media_player.errorOccurred.connect(self._handle_error)
        self._media_player.mediaStatusChanged.connect(self._handle_status)
        self._media_player.playbackStateChanged.connect(self._handle_state)
        self._media_player.setVideoOutput(self.video_item)

        self.controls.play_toggled.connect(self.toggle_play_pause)
        self.controls.seek_requested.connect(self._seek_ms)
        self.controls.step_requested.connect(self._step_frames)
        self.controls.skip_requested.connect(self._step_frames)
        self.controls.goto_frame_requested.connect(self.seek_to_frame)
        self.controls.speed_changed.connect(self._media_player.setPlaybackRate)

    # --- API ---

    def set_media(self, file_path: Optional[Path]):
        self._media_player.stop()
        self.video_item.setVisible(False)
        self.image_item.setVisible(False)
        self._is_video_loaded = False

        if not file_path or not file_path.is_file():
            self.controls.set_controls_enabled(False)
            self.view_stack.setCurrentWidget(self.placeholder)
            return

        self.view_stack.setCurrentWidget(self.view.parent())
        suffix = file_path.suffix.lower()

        if suffix in [e.lower() for e in IMAGE_EXTENSIONS]:
            self.controls.set_controls_enabled(False)
            self._load_image(file_path)
        elif suffix in [e.lower() for e in VIDEO_EXTENSIONS]:
            self.controls.set_controls_enabled(True)
            self._load_video(file_path)
        else:
            self.view_stack.setCurrentWidget(self.placeholder)
            self.placeholder.setText(f"Unsupported: {suffix}")

    def _load_image(self, path: Path):
        from PyQt6.QtGui import QPixmap
        pix = QPixmap(str(path))
        if not pix.isNull():
            self.image_item.setPixmap(pix)
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.image_item.setVisible(True)
            self._fit_view()

    def _load_video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                self._fps = cap.get(cv2.CAP_PROP_FPS)
                self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                if self._fps and self._fps > 0:
                    self._is_video_loaded = True
                    self.video_item.setSize(QRectF(0, 0, w, h).size())
                    self.scene.setSceneRect(self.video_item.boundingRect())
                    self.video_item.setVisible(True)
                    self.controls.set_frame_limit(self._total_frames)
                    self._media_player.setSource(QUrl.fromLocalFile(str(path)))
                    return
        finally:
            cap.release()

        logging.warning(f"Failed to load video: {path.name}")
        self.placeholder.setText("Error loading video metadata.")
        self.view_stack.setCurrentWidget(self.placeholder)

    # --- Playback ---

    def toggle_play_pause(self):
        if self._media_player.source().isEmpty():
            return
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
        else:
            if self._media_player.position() >= self._duration_ms - 100:
                self._media_player.setPosition(0)
            self._media_player.play()

    def _seek_ms(self, ms: int):
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
        self._media_player.setPosition(ms)
        if self._is_video_loaded:
            self.frame_changed.emit(self._ms_to_frame(ms))

    def _step_frames(self, delta_frames: int):
        if not self._is_video_loaded:
            return
        self.seek_to_frame(self.get_current_frame() + delta_frames)

    def seek_to_frame(self, frame_no: int):
        if not self._is_video_loaded:
            return
        target = max(0, min(frame_no, self._total_frames - 1))
        self._seek_ms(self._frame_to_ms(target))

    def _single_step_forward(self):
        self._step_frames(1)

    def _single_step_backward(self):
        self._step_frames(-1)

    def _skip_forward(self):
        self._step_frames(self.controls.spin_step.value())

    def _skip_backward(self):
        self._step_frames(-self.controls.spin_step.value())

    # --- Event handlers ---

    def _on_position_changed(self, ms: int):
        self.controls.update_time(ms, self._duration_ms)
        if self._is_video_loaded:
            frame = self._ms_to_frame(ms)
            self.controls.update_frames(frame, self._total_frames)
            self.frame_changed.emit(frame)

    def _on_duration_changed(self, ms: int):
        self._duration_ms = ms
        self.controls.set_duration(ms)

    def _handle_state(self, state):
        self.controls.set_playing_state(
            state == QMediaPlayer.PlaybackState.PlayingState
        )

    def _handle_status(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            QTimer.singleShot(0, self._kickstart_video)
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._media_player.setPosition(self._duration_ms)
            self._media_player.pause()

    def _kickstart_video(self):
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self._media_player.pause()
            self._media_player.setPosition(0)
        self._fit_view()

    def _handle_error(self, _, err_str):
        logging.error(f"Player Error: {err_str}")
        self.set_media(None)
        self.placeholder.setText(f"Playback Error: {err_str}")

    def resizeEvent(self, event: Optional[QResizeEvent]):
        super().resizeEvent(event)
        self._fit_view()

    def _fit_view(self):
        target_item = None
        if self.video_item.isVisible():
            target_item = self.video_item
        elif self.image_item.isVisible():
            target_item = self.image_item

        if target_item and not target_item.boundingRect().isEmpty():
            rect = target_item.boundingRect()
            self.view.fitInView(target_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.scene.setSceneRect(rect)
            self.overlay_proxy.setGeometry(rect)

    # --- Converters & accessors ---

    def _ms_to_frame(self, ms: int) -> int:
        return (min(round((ms / 1000.0) * self._fps), self._total_frames - 1)
                if self._fps else 0)

    def _frame_to_ms(self, frame: int) -> int:
        return int((frame / self._fps) * 1000) if self._fps else 0

    def get_current_frame(self) -> int:
        return (self._ms_to_frame(self._media_player.position())
                if self._is_video_loaded else -1)

    def set_overlay_visible(self, v):
        self.info_overlay.setVisible(v)

    def set_overlay_text(self, t):
        self.info_overlay.set_text(t)

    def set_overlay_font_size(self, s):
        self.info_overlay.set_font_size(s)

    @property
    def fps(self):
        return self._fps

    @property
    def total_frames(self):
        return self._total_frames
