import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QWidget


class PoseAgreementTimelineWidget(QWidget):
    """
    A custom timeline widget that draws a color-coded bar representing pose agreement 
    between two prediction runs (Green = Perfect agreement, Red = Swapped/Disagreement).
    Clicking or dragging on the widget seeks the video player to the corresponding frame.
    """
    frame_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.frame_distances: dict[int, float] = {}
        self.total_frames = 0
        self.current_frame = -1
        self.max_disagreement_distance = 50.0

        # Set a fixed height for the timeline bar
        self.setFixedHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._is_dragging = False

    def set_data(self, frame_distances: dict[int, float], total_frames: int):
        """Sets the computed frame distances and total frames, triggers a redraw."""
        self.frame_distances = frame_distances
        self.total_frames = total_frames
        self.current_frame = 0
        self.update()

    def set_current_frame(self, frame: int):
        """Updates the active frame playhead position."""
        if self.current_frame != frame:
            self.current_frame = frame
            self.update()

    def set_max_disagreement_distance(self, distance: float):
        """Sets the distance threshold corresponding to maximum disagreement (Red)."""
        if self.max_disagreement_distance != distance:
            self.max_disagreement_distance = max(1.0, distance)
            self.update()

    def get_color_for_distance(self, distance: float) -> QColor:
        """Maps a distance value to a color from Green to Red."""
        if pd.isna(distance) or distance < 0:
            # Gray for no data / missing predictions
            return QColor(80, 80, 80, 100)

        ratio = min(1.0, max(0.0, distance / self.max_disagreement_distance))

        # Premium palette: Interpolate RGB from Emerald Green (46, 204, 113) to Alizarin Red (231, 76, 60)
        r = int(46 + (231 - 46) * ratio)
        g = int(204 + (76 - 204) * ratio)
        b = int(113 + (60 - 113) * ratio)
        return QColor(r, g, b)

    def _frame_from_pos(self, x_pos: float) -> int:
        if self.total_frames <= 0:
            return 0
        ratio = x_pos / self.width()
        ratio = max(0.0, min(1.0, ratio))
        return int(ratio * (self.total_frames - 1))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            frame = self._frame_from_pos(event.position().x())
            self.frame_selected.emit(frame)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            frame = self._frame_from_pos(event.position().x())
            self.frame_selected.emit(frame)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        if self.total_frames <= 0:
            # Draw empty/placeholder gray bar
            painter.fillRect(0, 0, width, height, QColor(40, 40, 40))
            painter.setPen(QColor(70, 70, 70))
            painter.drawRect(0, 0, width - 1, height - 1)
            return

        # Draw the color-coded agreement timeline
        # To avoid performance bottleneck with huge number of frames,
        # we can step through pixels and sample the corresponding frame.
        for x in range(width):
            frame_idx = self._frame_from_pos(x)
            dist = self.frame_distances.get(frame_idx, float('nan'))
            color = self.get_color_for_distance(dist)

            painter.setPen(color)
            painter.drawLine(x, 0, x, height)

        # Draw border
        painter.setPen(QColor(50, 50, 50))
        painter.drawRect(0, 0, width - 1, height - 1)

        # Draw playhead line and handle
        if 0 <= self.current_frame < self.total_frames:
            playhead_x = int((self.current_frame / (self.total_frames - 1)) * (width - 1)) if self.total_frames > 1 else 0
            
            # White vertical indicator line
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(playhead_x, 0, playhead_x, height)

            # Circular playhead handle at top/bottom centers
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(playhead_x - 3, 0, 6, 6)
            painter.drawEllipse(playhead_x - 3, height - 6, 6, 6)
