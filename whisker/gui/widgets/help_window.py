# START_FILE: whisker/gui/widgets/help_window.py [Defines the simple doc viewer window]
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from whisker.core.constants import WHISKER_BASE_SRC_DIR
from whisker.gui.constants import ASSETS_DIR

class HelpWindow(QMainWindow):
    """
    A simple native PyQt window to display documentation from the doc/ directory.
    It uses QTextEdit for fast, lightweight rendering of rich text (Markdown/HTML).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("WHISKER Documentation")
        self.setWindowIcon(QIcon(str(ASSETS_DIR / "favicon.ico")))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.text_editor = QTextEdit()
        self.text_editor.setReadOnly(True)
        # Use Markdown as the input format for documentation simplicity
        layout.addWidget(self.text_editor)
        self.load_documentation()

    def load_documentation(self):
        """Loads all text/Markdown files from the doc/ directory and displays them."""
        doc_dir = WHISKER_BASE_SRC_DIR.parent / "docs" # Assumes doc/ is next to whisker/
        
        if not doc_dir.is_dir():
            self.text_editor.setText(
                "# Documentation Not Found\n"
                f"Expected documentation directory not found at: `{doc_dir.resolve()}`"
            )
            logging.error(f"Documentation directory missing at: {doc_dir.resolve()}")
            return

        all_content = []
        
        # Define a consistent header structure for multiple files
        all_content.append("# WHISKER Documentation\n\n---")
        
        # Include common documentation extensions (e.g., Markdown and plain text)
        doc_extensions = ["*.md", "*.txt"]
        
        found_files = []
        for ext in doc_extensions:
            # Sort is crucial for consistent display order (e.g., INTRODUCTION.md first)
            found_files.extend(sorted(doc_dir.glob(ext)))
            
        if not found_files:
            self.text_editor.setText(
                "# No Documentation Files Found\n"
                f"Scanned directory: `{doc_dir.resolve()}`\n"
                f"No files matching extensions {doc_extensions} were found."
            )
            return
            
        for file_path in found_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Add a Markdown header for each file (Level 2 or 3)
                all_content.append(f"\n\n## {file_path.name}\n\n{content}")
            except Exception as e:
                logging.error(f"Failed to read documentation file {file_path}: {e}")
                all_content.append(f"\n\n## {file_path.name} (Error)\n\n*Could not read file content: {e}*")

        self.text_editor.setMarkdown("\n".join(all_content))

# END_FILE: whisker/gui/widgets/help_window.py