import pathlib
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtCore import Qt


class ScalableImageLabel(QLabel):
    """
    A custom QLabel that automatically scales its pixmap to fit the label's size
    while maintaining the original aspect ratio.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setScaledContents(False)
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.resizeEvent(None)

    def setImageFromPath(self, path: pathlib.Path):
        self.setPixmap(QPixmap(str(path)))

    def resizeEvent(self, event: QResizeEvent | None):
        if not self._pixmap or self._pixmap.isNull():
            return

        scaled_pixmap = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled_pixmap)
