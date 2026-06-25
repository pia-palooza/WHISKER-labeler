# UPDATE_FILE: whisker/gui/workflows/behavior_classification/widgets/behavior_labeling/timeline.py
#
# A dependency-light timeline strip that draws each behavior on its own row and
# paints the hand-annotated bouts as coloured spans, with a draggable playhead.
# WHISKER's full app uses a matplotlib ProbabilityPlotWidget here (because it
# also overlays model probabilities/predictions); since the labeler only does
# hand annotation, a plain QPainter strip keeps the dependency set unchanged.
from typing import List, Optional

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget

from whisker.gui.constants import KEYPOINT_QCOLORS


class BehaviorTimelineWidget(QWidget):
    """Per-behavior timeline of annotated bouts with a click/drag playhead."""

    seek_requested = pyqtSignal(int)  # frame index

    ROW_HEIGHT = 22
    LABEL_WIDTH = 120
    TOP_PAD = 6
    BOTTOM_PAD = 18

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._behaviors: List[str] = []
        self._bouts_df: Optional[pd.DataFrame] = None
        self._total_frames: int = 0
        self._fps: float = 30.0
        self._current_frame: int = 0
        self.setMinimumHeight(80)
        self.setMouseTracking(True)

    def set_project_behaviors(self, behaviors: List[str]):
        self._behaviors = list(behaviors)
        self.updateGeometry()
        self.update()

    def set_data(
        self,
        bouts_df: Optional[pd.DataFrame],
        total_frames: int,
        fps: float,
    ):
        self._bouts_df = bouts_df
        self._total_frames = max(0, int(total_frames))
        self._fps = fps if fps and fps > 0 else 30.0
        self.update()

    def set_current_frame(self, frame: int):
        self._current_frame = max(0, frame)
        self.update()

    def clear(self):
        self._bouts_df = None
        self._total_frames = 0
        self._current_frame = 0
        self.update()

    def _color_for(self, behavior: str) -> QColor:
        if behavior in self._behaviors:
            idx = self._behaviors.index(behavior)
        else:
            idx = abs(hash(behavior))
        return QKEYPOINT(idx)

    def sizeHint(self) -> QSize:
        rows = max(1, len(self._behaviors))
        return QSize(400, self.TOP_PAD + rows * self.ROW_HEIGHT + self.BOTTOM_PAD)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # --- Geometry helpers ---

    def _plot_left(self) -> int:
        return self.LABEL_WIDTH

    def _plot_width(self) -> int:
        return max(1, self.width() - self.LABEL_WIDTH - 10)

    def _frame_to_x(self, frame: int) -> float:
        if self._total_frames <= 1:
            return self._plot_left()
        frac = frame / (self._total_frames - 1)
        return self._plot_left() + frac * self._plot_width()

    def _x_to_frame(self, x: float) -> int:
        if self._total_frames <= 1:
            return 0
        frac = (x - self._plot_left()) / self._plot_width()
        frac = min(1.0, max(0.0, frac))
        return int(round(frac * (self._total_frames - 1)))

    # --- Interaction ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._total_frames > 0:
            if event.position().x() >= self._plot_left():
                self.seek_requested.emit(self._x_to_frame(event.position().x()))

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._total_frames > 0:
            if event.position().x() >= self._plot_left():
                self.seek_requested.emit(self._x_to_frame(event.position().x()))

    # --- Painting ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if not self._behaviors:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No behaviors defined in project.")
            return

        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        video_bouts = self._bouts_df
        for i, behavior in enumerate(self._behaviors):
            row_top = self.TOP_PAD + i * self.ROW_HEIGHT
            row_rect = QRectF(0, row_top, self.width(), self.ROW_HEIGHT)

            if i % 2 == 0:
                painter.fillRect(row_rect, QColor("#242424"))

            color = self._color_for(behavior)

            # Behavior label
            painter.setPen(QColor("#ddd"))
            painter.drawText(
                QRectF(4, row_top, self.LABEL_WIDTH - 8, self.ROW_HEIGHT),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                behavior,
            )

            # Track baseline
            painter.setPen(QPen(QColor("#333"), 1))
            track_y = row_top + self.ROW_HEIGHT / 2
            painter.drawLine(self._plot_left(), int(track_y),
                             self._plot_left() + self._plot_width(), int(track_y))

            # Bout spans
            if video_bouts is not None and not video_bouts.empty:
                beh_bouts = video_bouts[video_bouts["behavior"] == behavior]
                fill = QColor(color)
                fill.setAlpha(180)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                for row in beh_bouts.itertuples():
                    x1 = self._frame_to_x(int(row.start_frame))
                    x2 = self._frame_to_x(int(row.end_frame))
                    span = QRectF(x1, row_top + 3, max(2.0, x2 - x1),
                                  self.ROW_HEIGHT - 6)
                    painter.drawRoundedRect(span, 2, 2)

        # Playhead
        if self._total_frames > 0:
            x = self._frame_to_x(self._current_frame)
            bottom = self.TOP_PAD + len(self._behaviors) * self.ROW_HEIGHT
            painter.setPen(QPen(QColor("#ff5555"), 1.5))
            painter.drawLine(int(x), self.TOP_PAD, int(x), bottom)

            # Time axis label
            painter.setPen(QColor("#888"))
            secs = int(self._current_frame / self._fps)
            painter.drawText(
                QRectF(self._plot_left(), bottom, self._plot_width(), self.BOTTOM_PAD),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"Frame {self._current_frame}  ({secs // 60:02d}:{secs % 60:02d})",
            )


def QKEYPOINT(idx: int) -> QColor:
    """Pick a stable colour from the shared keypoint palette."""
    if not KEYPOINT_QCOLORS:
        return QColor("#4363D8")
    return QColor(KEYPOINT_QCOLORS[idx % len(KEYPOINT_QCOLORS)])
