from typing import Optional
from PyQt6.QtCore import Qt, QVariantAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton
from PyQt6.QtGui import QPainter, QCursor, QPixmap
from whisker.gui.constants import ASSETS_DIR


class HelpIcon(QLabel):
    """A small help icon that displays a tooltip on hover."""

    def __init__(self, tooltip_text: str, parent=None):
        super().__init__(parent)
        self.setToolTip(tooltip_text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Try to load custom icon from the assets directory
        icon_path = ASSETS_DIR / "help_icon.png"
        pixmap = QPixmap(str(icon_path))
        if not pixmap.isNull():
            self.setPixmap(pixmap.scaled(
                16, 16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            # Fallback to the unicode text icon if help_icon.png is not found
            self.setText("ⓘ")
            self.setStyleSheet("""
                QLabel {
                    color: #7f8c8d;
                    font-size: 14px;
                    font-weight: bold;
                }
                QLabel:hover {
                    color: #2980b9;
                }
            """)


class VerticalLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(0, 0, self.height(), self.width(), self.alignment(), self.text())
        painter.end()

    def minimumSizeHint(self):
        return super().minimumSizeHint().transposed()


class ResizeHandle(QWidget):
    """A flexible handle supporting drag resizing on specified edges (Top, Bottom, Left, Right)."""
    def __init__(self, edge: Qt.Edge, mouse_move_callback, parent=None):
        super().__init__(parent)
        self.edge = edge
        self.callback = mouse_move_callback
        self._dragging = False
        self._start_global_pos = None

        if edge in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
            self.setFixedWidth(6)
        else:
            self.setCursor(QCursor(Qt.CursorShape.SplitVCursor))
            self.setFixedHeight(6)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            # Store the absolute global coordinate anchor when the drag begins
            self._start_global_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            # Send the current global position so we can calculate the true delta
            self.callback(self.edge, event.globalPosition().toPoint(), self._start_global_pos)
            # Update the anchor point dynamically to handle ongoing smooth movement
            self._start_global_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()


class BaseCollapsiblePanel(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, title_widget, collapsed_size=35, parent=None):
        super().__init__(parent)
        self.collapsed_size = collapsed_size
        self.expanded_size = None
        self.horiz_title = title_widget

        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("ToggleButton")
        self.toggle_button.clicked.connect(self.toggle)
        self.toggle_button.setSizePolicy(
            self.toggle_button.sizePolicy().Policy.Fixed, 
            self.toggle_button.sizePolicy().Policy.Fixed
        )

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.header_row = QWidget()
        self.header_layout = QHBoxLayout(self.header_row)
        self.header_layout.setContentsMargins(4, 4, 4, 4)
        self.header_layout.setSpacing(6)
        
        self.header_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.header_row)
        
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(220)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.valueChanged.connect(self._animate_size)
        self.animation.finished.connect(self._on_animation_finished)

        # Storage for instantiated active handles
        self.handles = {}

    def _animate_size(self, value): raise NotImplementedError
    def current_size(self): raise NotImplementedError
    def default_expanded_size(self): raise NotImplementedError
    def _on_animation_finished(self): pass
    
    def _on_drag(self, edge: Qt.Edge, current_global_pos, start_global_pos):
        # Calculate the direct movement difference in screen space
        delta_x = current_global_pos.x() - start_global_pos.x()
        delta_y = current_global_pos.y() - start_global_pos.y()
        
        if edge in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            # Left edge: moving right (positive delta) shrinks width. Moving left expands it.
            if edge == Qt.Edge.LeftEdge:
                new_width = max(self.collapsed_size, self.width() - delta_x)
            else:
                new_width = max(self.collapsed_size, self.width() + delta_x)
            self.setFixedWidth(new_width)
            self.expanded_size = new_width
            
        elif edge in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
            # Top edge: moving down (positive delta) shrinks height. Moving up expands it.
            if edge == Qt.Edge.TopEdge:
                new_height = max(self.collapsed_size, self.height() - delta_y)
            else:
                new_height = max(self.collapsed_size, self.height() + delta_y)
            self.setFixedHeight(new_height)
            self.expanded_size = new_height

    def update_ui_states(self, expanding):
        if expanding:
            self.header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.horiz_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.horiz_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        for handle in self.handles.values():
            handle.setVisible(expanding)

    def toggle(self):
        expanding = self.current_size() == self.collapsed_size
        if expanding:
            self.update_ui_states(True)
            start = self.collapsed_size
            end = self.expanded_size or self.default_expanded_size()
        else:
            self.expanded_size = self.current_size()
            self.update_ui_states(False)
            start = self.expanded_size
            end = self.collapsed_size

        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()
        self.toggled.emit(expanding)


class VerticalCollapsiblePanel(BaseCollapsiblePanel):
    def __init__(self, title_widget, parent=None, drag_edges: Optional[Qt.Edge] = Qt.Edge.RightEdge):
        super().__init__(title_widget, collapsed_size=35, parent=parent)
        
        self.expanded_header_content = QWidget()
        self.expanded_header_layout = QHBoxLayout(self.expanded_header_content)
        self.expanded_header_layout.setContentsMargins(0, 0, 0, 0)
        self.expanded_header_layout.addWidget(self.horiz_title)
        self.header_layout.addWidget(self.expanded_header_content)

        self.vert_title = VerticalLabel(self.horiz_title.text())
        self.vert_title.setObjectName("HeaderLabel")
        self.vert_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vert_title.hide()
        self.main_layout.addWidget(self.vert_title)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("PanelScroll")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.content_widget = QWidget()
        self.content_widget.setObjectName("PanelContent")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(4)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.content_widget)

        self.body_container = QWidget()
        self.body_layout = QHBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(0)
        
        # Add Left drag handle if designated
        if drag_edges is not None and drag_edges & Qt.Edge.LeftEdge:
            self.handles[Qt.Edge.LeftEdge] = ResizeHandle(Qt.Edge.LeftEdge, self._on_drag, self)
            self.body_layout.addWidget(self.handles[Qt.Edge.LeftEdge])
            
        self.body_layout.addWidget(self.scroll_area)
        
        # Add Right drag handle if designated (Default Behavior)
        if drag_edges is not None and drag_edges & Qt.Edge.RightEdge:
            self.handles[Qt.Edge.RightEdge] = ResizeHandle(Qt.Edge.RightEdge, self._on_drag, self)
            self.body_layout.addWidget(self.handles[Qt.Edge.RightEdge])
            
        self.main_layout.addWidget(self.body_container)
        self.toggle_button.setText("▼")

    def _animate_size(self, value): self.setFixedWidth(value)
    def current_size(self): return self.width()
    def default_expanded_size(self): return self.sizeHint().width()

    def _on_animation_finished(self):
        if self.current_size() != self.collapsed_size:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)

    def update_ui_states(self, expanding):
        super().update_ui_states(expanding)
        self.vert_title.setVisible(not expanding)
        self.expanded_header_content.setVisible(expanding)
        self.body_container.setVisible(expanding)
        self.toggle_button.setText("▼" if expanding else "▶")


class HorizontalCollapsiblePanel(BaseCollapsiblePanel):
    def __init__(self, title_widget, parent=None, drag_edges: Optional[Qt.Edge] = Qt.Edge.BottomEdge):
        super().__init__(title_widget, collapsed_size=35, parent=parent)
        self.header_layout.setSpacing(8)
        self.header_layout.addWidget(self.horiz_title)
        self.header_spacer = self.header_layout.addStretch()

        self.content_container = QWidget()
        self.content_container.setObjectName("PanelContent")
        self.content_layout = QHBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(8, 4, 8, 4)
        self.content_layout.setSpacing(16)
        
        # Add Top drag handle if designated
        if drag_edges is not None and drag_edges & Qt.Edge.TopEdge:
            self.handles[Qt.Edge.TopEdge] = ResizeHandle(Qt.Edge.TopEdge, self._on_drag, self)
            self.main_layout.insertWidget(0, self.handles[Qt.Edge.TopEdge])

        self.main_layout.addWidget(self.content_container)
        
        # Add Bottom drag handle if designated (Default Behavior)
        if drag_edges is not None and drag_edges & Qt.Edge.BottomEdge:
            self.handles[Qt.Edge.BottomEdge] = ResizeHandle(Qt.Edge.BottomEdge, self._on_drag, self)
            self.main_layout.addWidget(self.handles[Qt.Edge.BottomEdge])
            
        self.toggle_button.setText("▼")

    def _animate_size(self, value): self.setFixedHeight(value)
    def current_size(self): return self.height()
    def default_expanded_size(self): return self.sizeHint().height()

    def _on_animation_finished(self):
        if self.current_size() != self.collapsed_size:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)

    def update_ui_states(self, expanding):
        super().update_ui_states(expanding)
        self.content_container.setVisible(expanding)
        if not expanding:
            self.header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.horiz_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toggle_button.setText("▼" if expanding else "▶")