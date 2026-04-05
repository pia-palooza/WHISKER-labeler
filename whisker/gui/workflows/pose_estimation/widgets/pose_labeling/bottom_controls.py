import logging
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton,
)

class PoseLabelingBottomControlsWidget(QWidget):
    save_requested = pyqtSignal()
    swap_identities_requested = pyqtSignal(str, str)
    request_select_prev_image = pyqtSignal()
    request_select_next_image = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._identities = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 0)

        self.save_btn = QPushButton("Save Labels (`Ctrl+S`)")
        self.swap_btn = QPushButton("Swap Identities (`Ctrl+I`)")
        self.prev_btn = QPushButton("Prev Image (`N`)")
        self.next_btn = QPushButton("Next Image (`M`)")

        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.swap_btn)
        layout.addWidget(self.save_btn)

        self.prev_btn.clicked.connect(self.request_select_prev_image)
        self.next_btn.clicked.connect(self.request_select_next_image)
        self.swap_btn.clicked.connect(self._on_swap)
        self.save_btn.clicked.connect(self.save_requested)

    def set_identities(self, identities: Optional[list[str]]):
        self._identities = identities if identities else []

    def _on_swap(self):
        if len(self._identities) == 2:
            self.swap_identities_requested.emit(
                self._identities[0], self._identities[1]
            )
        else:
            logging.error("Shortcut swap requires exactly 2 identities in project.")

    def swap_identities_shortcut(self):
        self._on_swap()

