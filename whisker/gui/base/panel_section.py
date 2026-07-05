from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton


class PanelSection(QWidget):
    """A sleek, vertically collapsible container for nesting lists inside panels."""
    def __init__(self, title, allow_toggle=False, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)

        icon = "▼" if allow_toggle else ""
        self.toggle_btn = QPushButton(f"{icon} {title}")
        self.toggle_btn.setObjectName("SectionHeaderBtn")
        self.toggle_btn.setStyleSheet("text-align: left; font-weight: bold; padding: 4px;")
        if allow_toggle:
            self.toggle_btn.clicked.connect(self.toggle_content)
        self.layout.addWidget(self.toggle_btn)

        self.content_container = QWidget()
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(4, 2, 4, 6)
        self.content_layout.setSpacing(4)
        self.layout.addWidget(self.content_container)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def toggle_content(self):
        is_visible = self.content_container.isVisible()
        self.content_container.setVisible(not is_visible)
        self.toggle_btn.setText(f"▶ {self.toggle_btn.text()[2:]}" if is_visible else f"▼ {self.toggle_btn.text()[2:]}")