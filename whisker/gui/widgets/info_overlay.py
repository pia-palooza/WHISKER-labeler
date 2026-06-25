from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QFont, QTextDocument
from PyQt6.QtWidgets import QWidget


class InfoOverlay(QWidget):
    """
    A transparent widget for displaying simple text information as an overlay.

    Designed to be placed on top of another widget (like a video player). It
    paints a text string with a semi-transparent background in its bottom-left
    corner for readability. Supports HTML for rich text formatting.
    """

    DEFAULT_FONT_SIZE = 10

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self._text = ""
        self._font = QFont("Arial", self.DEFAULT_FONT_SIZE, QFont.Weight.Bold)
        self._text_color = QColor("white")
        self._bg_color = QColor(0, 0, 0, 120)
        self._padding = 8
        self._doc = QTextDocument()

    def set_text(self, text: str):
        if self._text != text:
            self._text = text
            self._doc.setHtml(text)
            self._doc.setDefaultFont(self._font)
            self.update()

    def set_font_size(self, size: int):
        if size > 0:
            self._font.setPointSize(size)
            self._doc.setDefaultFont(self._font)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isVisible() or not self._text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        doc_size = self._doc.size()
        bg_rect_width = doc_size.width() + (2 * self._padding)
        bg_rect_height = doc_size.height() + (2 * self._padding)

        bg_x = self._padding
        bg_y = self.height() - bg_rect_height - self._padding
        bg_rect = QRectF(bg_x, bg_y, bg_rect_width, bg_rect_height)

        painter.setBrush(self._bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bg_rect, 5, 5)

        painter.save()
        painter.translate(bg_x + self._padding, bg_y + self._padding)
        self._doc.drawContents(painter)
        painter.restore()
