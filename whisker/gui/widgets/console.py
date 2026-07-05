import logging
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import QWidget, QTextEdit, QLabel

from whisker.base.logger import get_logger
from whisker.gui.utils import QtLogHandler
from whisker.gui.base.collapsible_panel import HorizontalCollapsiblePanel

class ConsoleWidget(HorizontalCollapsiblePanel):
    """
    A collapsible panel at the bottom of the window that captures and displays log entries.
    Inherits from HorizontalCollapsiblePanel to support animated collapsing and top-edge resizing.
    """
    def __init__(self, parent: QWidget | None = None):
        title = QLabel("CONSOLE")
        title.setObjectName("HeaderLabel")
        super().__init__(title, parent=parent, drag_edges=Qt.Edge.TopEdge)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)

        # Use a monospaced font for better log alignment and readability
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(9)
        self.log_display.setFont(font)

        self.content_layout.addWidget(self.log_display)

        # Clear fixed height on startup so that it's resizable by the vertical splitter
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)

        self.log_handler = QtLogHandler(get_logger())
        self.log_handler.log_received.connect(self.append_log)

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
        whitespace_preserved_message = html_message.replace(" ", "&nbsp;")

        # 2. Wrap the message in a <p> tag to ensure it acts as a block and creates a new line.
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
