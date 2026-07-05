# START_DIFF: whisker/gui/widgets/info_overlay.py [Add font size control and increase default]
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QTextDocument
from PyQt6.QtWidgets import QWidget


class InfoOverlay(QWidget):
    DEFAULT_FONT_SIZE = 10

    """
    A transparent widget for displaying simple text information as an overlay.

    This widget is designed to be placed on top of another widget (like a
    video player) using a QStackedLayout. It paints a text string with a
    semi-transparent background in its bottom-left corner for readability.
    Supports HTML for rich text formatting.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # This makes the widget transparent, so we can see the video behind it.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self._text = ""
        self._font = QFont("Arial", self.DEFAULT_FONT_SIZE, QFont.Weight.Bold)
        self._text_color = QColor("white")
        self._bg_color = QColor(0, 0, 0, 120)  # Black with ~50% opacity
        self._padding = 8
        self._doc = QTextDocument()

    def set_text(self, text: str):
        """Updates the text to be displayed and schedules a repaint."""
        if self._text != text:
            self._text = text
            self._doc.setHtml(text)
            self._doc.setDefaultFont(self._font)
            self.update()  # Trigger a paintEvent

    def set_font_size(self, size: int):
        """Updates the font size of the overlay text."""
        if size > 0:
            self._font.setPointSize(size)
            self._doc.setDefaultFont(self._font)
            # We must call update to trigger a repaint with the new font size.
            self.update()

    def paintEvent(self, event):
        """Paints the text and its background when the widget is updated."""
        super().paintEvent(event)
        if not self.isVisible() or not self._text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate the size of the document
        doc_size = self._doc.size()
        
        # Create the background rectangle with padding
        bg_rect_width = doc_size.width() + (2 * self._padding)
        bg_rect_height = doc_size.height() + (2 * self._padding)
        
        bg_x = self._padding
        bg_y = self.height() - bg_rect_height - self._padding
        
        from PyQt6.QtCore import QRectF
        bg_rect = QRectF(bg_x, bg_y, bg_rect_width, bg_rect_height)

        # Draw the semi-transparent background
        painter.setBrush(self._bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bg_rect, 5, 5)

        # Draw the text on top of the background
        painter.save()
        painter.translate(bg_x + self._padding, bg_y + self._padding)
        self._doc.drawContents(painter)
        painter.restore()
