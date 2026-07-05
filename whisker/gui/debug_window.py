import logging
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QMainWindow, QApplication
from PyQt6.QtGui import QIcon

from whisker.gui.constants import ASSETS_DIR
from whisker.base.logger import get_logger
from whisker.gui.utils import QtLogHandler

class DebugWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("WHISKER Debug Window")

        screen = QApplication.primaryScreen()
        if screen:
            screen_geom = screen.availableGeometry()
            width = int(screen_geom.width() * 0.8)
            height = int(screen_geom.height() * 0.5)
            x = screen_geom.x() + (screen_geom.width() - width) // 2
            y = screen_geom.y() + (screen_geom.height() - height) // 2
            self.setGeometry(x, y, width, height)
        else:
            self.setGeometry(200, 200, 1920, 1080 // 2)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)

        # Use a monospaced font for better log alignment and readability
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(9)
        self.log_display.setFont(font)

        self.setCentralWidget(self.log_display)

        self.log_handler = QtLogHandler(get_logger())
        self.log_handler.log_received.connect(self.append_log)

        self._set_window_icon()

    @pyqtSlot(str)
    def append_log(self, html_message: str):
        """
        Appends a pre-formatted HTML message to the log display, ensuring
        all whitespace is preserved and each message appears on a new line.
        """
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)

        # 1. Replace all spaces with non-breaking spaces to ensure they are rendered.
        # This is the most reliable way to force whitespace preservation in HTML.
        # It's important to do this *before* potentially adding any other HTML tags,
        # unless you specifically want to preserve spaces within those tags.
        # Let's assume the incoming html_message might have multiple spaces that need preserving.
        whitespace_preserved_message = html_message.replace(" ", "&nbsp;")

        # 2. Wrap the message in a <p> tag to ensure it acts as a block and creates a new line.
        # Applying a minimal style to prevent any default paragraph margins/padding.
        # We can still keep the <pre> tag inside the <p> if there are internal newlines
        # that need to be preserved within the log message itself.
        # If your logs are single lines, the <pre> might not be strictly necessary
        # with &nbsp; but it doesn't hurt.
        formatted_html_entry = (
            f"<p style='margin:0; padding:0; white-space:pre-wrap;'>"
            f"{whitespace_preserved_message}\r\n"
            f"</p>"
        )

        # 3. Insert the fully formatted HTML.
        self.log_display.insertHtml(formatted_html_entry)

        # Optional: Scroll to the end to ensure the latest log is visible
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def _set_window_icon(self):
        icon_path = ASSETS_DIR / "favicon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logging.warning(f"Window icon 'favicon.ico' not found at {icon_path}")