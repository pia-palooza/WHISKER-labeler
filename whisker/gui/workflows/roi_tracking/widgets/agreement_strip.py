"""Ported from full WHISKER's ``whisker/gui/workflows/roi_tracking/widgets/agreement_strip.py``, with PySide6 ->
PyQt6 renames only (``Signal`` -> ``pyqtSignal``)."""
from typing import List, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from whisker.services.roi_tracking.internal.verify_tracks import AGREE, DISAGREE, NEITHER, ONLY_A, ONLY_B

STATUS_COLORS = {AGREE: "#2ECC71", DISAGREE: "#E74C3C", ONLY_A: "#F1C40F", ONLY_B: "#E67E22", NEITHER: "#BDC3C7"}
# What a pixel column shows when it covers several frames: the most telling status among them.
_PRIORITY = {AGREE: 0, NEITHER: 1, ONLY_A: 2, ONLY_B: 2, DISAGREE: 3}


class AgreementStrip(QWidget):
    """One colour strip per animal over the whole video: where two prediction runs agree (green), disagree (red), or
    only run A (yellow) / run B (orange) has the animal. Clicking a strip seeks the video to that frame."""

    frame_selected = pyqtSignal(int)
    ROW_HEIGHT = 14
    LABEL_WIDTH = 110
    STATUS_COLORS = STATUS_COLORS               # a subclass can recolour the statuses

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._names: List[str] = []
        self._rows: List[np.ndarray] = []
        self._frame = -1
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.ROW_HEIGHT + 6)

    def set_rows(self, names: Sequence[str], statuses: Sequence[np.ndarray]):
        self._names = list(names)
        self._rows = [np.asarray(s) for s in statuses]
        self.setFixedHeight(max(1, len(self._rows)) * (self.ROW_HEIGHT + 2) + 6)
        self.update()

    def set_current_frame(self, frame: int):
        if frame != self._frame:
            self._frame = frame
            self.update()

    def frame_count(self) -> int:
        return max((len(r) for r in self._rows), default=0)

    def _frame_at(self, x: float) -> int:
        n, width = self.frame_count(), max(1, self.width() - self.LABEL_WIDTH)
        return int(min(n - 1, max(0, (x - self.LABEL_WIDTH) / width * n))) if n else 0

    def mousePressEvent(self, event):
        if self.frame_count() and event.position().x() >= self.LABEL_WIDTH:
            self.frame_selected.emit(self._frame_at(event.position().x()))

    @staticmethod
    def column_statuses(status: np.ndarray, columns: int) -> np.ndarray:
        """The status shown by each of ``columns`` pixel columns over ``status``."""
        n = len(status)
        if n == 0 or columns <= 0:
            return np.zeros(0, dtype=np.int8)
        if n <= columns:
            return status[(np.arange(columns) * n // columns)]
        priority = np.vectorize(_PRIORITY.get)(status)
        edges = np.linspace(0, n, columns + 1).astype(int)
        worst = np.maximum.reduceat(priority, edges[:-1])
        by_priority = {0: AGREE, 1: NEITHER, 3: DISAGREE}
        out = np.empty(columns, dtype=np.int8)
        for column, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if worst[column] == 2:                              # one run only: say which (A if any frame is A only)
                out[column] = ONLY_A if np.any(status[lo:max(hi, lo + 1)] == ONLY_A) else ONLY_B
            else:
                out[column] = by_priority[int(worst[column])]
        return out

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        width = max(1, self.width() - self.LABEL_WIDTH)
        n = self.frame_count()
        for i, (name, status) in enumerate(zip(self._names, self._rows)):
            top = 3 + i * (self.ROW_HEIGHT + 2)
            painter.setPen(self.palette().text().color())
            painter.drawText(QRectF(0, top, self.LABEL_WIDTH - 6, self.ROW_HEIGHT), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), name)
            columns = self.column_statuses(status, width)
            if len(columns):
                image = QImage(len(columns), 1, QImage.Format.Format_RGB32)
                for x, code in enumerate(columns):
                    image.setPixelColor(x, 0, QColor(self.STATUS_COLORS[int(code)]))
                painter.drawImage(QRectF(self.LABEL_WIDTH, top, len(columns), self.ROW_HEIGHT), image)
        if n and 0 <= self._frame < n:
            x = self.LABEL_WIDTH + (self._frame + 0.5) / n * width
            painter.setPen(QPen(QColor("black"), 1.5))
            painter.drawLine(int(x), 0, int(x), self.height())
        painter.end()
