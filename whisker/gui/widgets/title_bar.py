import sys
import logging
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QMenu,
    QPushButton,
    QSizePolicy
)
from PyQt6.QtGui import QIcon, QAction

class CustomTitleBar(QWidget):
    settings_requested = pyqtSignal()
    help_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CustomTitleBar")
        self.setFixedHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Window Icon
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(16, 16)
        self.icon_label.setScaledContents(True)
        self.icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.icon_label)
        layout.addSpacing(8)

        # 2. Integrated QMenuBar
        self.menu_bar = QMenuBar(self)
        self._setup_menus()
        layout.addWidget(self.menu_bar)

        # 3. Draggable Left Spacer
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(left_spacer)

        # 4. Centered Title
        self.title_label = QLabel(self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # Apply bold accent style for modern look
        self.title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title_label)

        # 5. Draggable Right Spacer
        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(right_spacer)

        # 6. Window Control Buttons
        self.btn_minimize = QPushButton("🗕")
        self.btn_minimize.setObjectName("TitleBarButton")
        self.btn_minimize.setToolTip("Minimize")
        self.btn_minimize.clicked.connect(self._minimize_window)
        layout.addWidget(self.btn_minimize)

        self.btn_maximize = QPushButton("🗖")
        self.btn_maximize.setObjectName("TitleBarButton")
        self.btn_maximize.setToolTip("Maximize")
        self.btn_maximize.clicked.connect(self._toggle_maximize_window)
        layout.addWidget(self.btn_maximize)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("TitleBarCloseButton")
        self.btn_close.setToolTip("Close")
        self.btn_close.clicked.connect(self._close_window)
        layout.addWidget(self.btn_close)

    def _setup_menus(self):
        # File Menu (Stubbed)
        file_menu = self.menu_bar.addMenu("File")
        new_proj_action = file_menu.addAction("New Project...")
        new_proj_action.setEnabled(False)
        open_ws_action = file_menu.addAction("Open Workspace...")
        open_ws_action.setEnabled(False)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self._close_window)

        # Edit Menu (Settings)
        edit_menu = self.menu_bar.addMenu("Edit")
        settings_action = edit_menu.addAction("Settings...")
        settings_action.triggered.connect(self.settings_requested.emit)

        # Selection Menu (Stubbed)
        selection_menu = self.menu_bar.addMenu("Selection")
        select_all_action = selection_menu.addAction("Select All")
        select_all_action.setEnabled(False)

        # View Menu (Stubbed)
        view_menu = self.menu_bar.addMenu("View")
        toggle_console_action = view_menu.addAction("Toggle Console")
        toggle_console_action.setEnabled(False)

        # Help Menu (Stubbed)
        help_menu = self.menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About Whisker")
        about_action.setEnabled(False)

    def setIcon(self, icon: QIcon):
        # Retrieve a pixmap for the label
        pixmap = icon.pixmap(16, 16)
        self.icon_label.setPixmap(pixmap)

    def setTitle(self, title: str):
        self.title_label.setText(title)

    def update_maximize_button(self):
        if self.window().isMaximized():
            self.btn_maximize.setText("🗗")
            self.btn_maximize.setToolTip("Restore Down")
        else:
            self.btn_maximize.setText("🗖")
            self.btn_maximize.setToolTip("Maximize")

    def is_over_interactive(self, title_bar_local_pos: QPoint) -> bool:
        """Determines if a local point is over any interactive element (buttons, menus) in the title bar."""
        for child in [self.menu_bar, self.btn_minimize, self.btn_maximize, self.btn_close]:
            if child.geometry().contains(title_bar_local_pos):
                return True
        return False

    def _minimize_window(self):
        self.window().showMinimized()

    def _toggle_maximize_window(self):
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()

    def _close_window(self):
        self.window().close()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.is_over_interactive(event.position().toPoint()):
                self._toggle_maximize_window()

    def contextMenuEvent(self, event):
        # Only trigger context menu if right click is not over interactive elements
        if self.is_over_interactive(event.pos()):
            super().contextMenuEvent(event)
            return

        self.show_system_menu(event.globalPosition().toPoint())

    def show_system_menu(self, global_pos: QPoint):
        menu = QMenu(self)
        is_maximized = self.window().isMaximized()

        restore_action = menu.addAction("Restore")
        restore_action.setEnabled(is_maximized)

        move_action = menu.addAction("Move")
        move_action.setEnabled(not is_maximized)

        size_action = menu.addAction("Size")
        size_action.setEnabled(not is_maximized)

        minimize_action = menu.addAction("Minimize")

        maximize_action = menu.addAction("Maximize")
        maximize_action.setEnabled(not is_maximized)

        menu.addSeparator()

        close_action = menu.addAction("Close")
        close_action.setShortcut("Alt+F4")

        action = menu.exec(global_pos)
        window_handle = self.window().windowHandle()

        if action == restore_action:
            self.window().showNormal()
        elif action == move_action and window_handle:
            window_handle.startSystemMove()
        elif action == size_action and window_handle:
            # Default sizing from bottom right edge
            window_handle.startSystemResize(Qt.Edge.BottomEdge | Qt.Edge.RightEdge)
        elif action == minimize_action:
            self.window().showMinimized()
        elif action == maximize_action:
            self.window().showMaximized()
        elif action == close_action:
            self.window().close()
