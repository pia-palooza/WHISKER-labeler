import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QCloseEvent, QKeySequence, QAction
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QComboBox,
    QWidget,
    QLabel,
    QMessageBox,
    QPushButton,
    QInputDialog,
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QVBoxLayout,
    QToolBar,
    QSplitter,
    QStackedWidget,
    QMenu
)
from PyQt6.QtGui import QIcon, QGuiApplication

from whisker.core.workspace import Workspace
from whisker.base.logger import configure_console_logger
from whisker.gui.constants import ASSETS_DIR
from whisker.gui.widgets import data_explorer
from whisker.gui.dialogs.warn_if_exists_dialog import WarnIfExistsDialog
from whisker.gui.tabs import BaseTab
from whisker.third_party.server_manager import get_server_manager
from whisker.gui.signals import MessageBus
from whisker.gui.dialogs import SettingsDialog
from whisker.gui.widgets.help_window import HelpWindow
from whisker.gui.widgets.console import ConsoleWidget
from whisker.gui.panels.navigation_panel import NavigationPanel
from whisker.core.workflows.workflow_enum import Workflow
from whisker.gui.widgets.data_explorer.study_panel import StudyPanel
from typing import Any

_STATUS_BAR_WELCOME_MESSAGE = "Welcome to WHISKER!"

class MainWindow(QMainWindow):
    active_project_changed = pyqtSignal(object)

    def __init__(self, log_level: int = logging.INFO):
        super().__init__()
        self.settings = QSettings("whisker", "main")
        
        logging.debug("Blocking UI signals during main window initialization.")
        MessageBus.get().blockSignals(True)

        self._original_window_title = "WHISKER: Workbench for Holistic Insights via Skeletal Kinematics & Event Recognition"
        self.setWindowTitle(self._original_window_title)

        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        width = int(screen_geometry.width() * 0.8)
        height = int(screen_geometry.height() * 0.8)
        x = screen_geometry.x() + (screen_geometry.width() - width) // 2
        y = screen_geometry.y() + (screen_geometry.height() - height) // 2
        self.setGeometry(x, y, width, height)

        self.set_window_icon()

        configure_console_logger(level=log_level)
        logging.info("Logging initialized.")

        # --- Messaging setup ---
        self.bridge = MessageBus.get()
        self.bridge.subscribe("gui/request/register_view", self._on_register_view)
        self.bridge.subscribe("gui/request/switch_view", self._on_switch_view)
        self.bridge.subscribe("gui/request/workflow_selected", self._on_workflow_selected_telemetry)

        # --- Main Layout Construction ---
        self.central_widget = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setHandleWidth(1)
        self.central_widget.setChildrenCollapsible(False)

        # Middle: Data Explorer (Full Height)
        self.data_explorer = data_explorer.Widget(self)
        
        # STUDY panel containing workflow controls
        self.study_panel = StudyPanel(self)
        self.project_selector = self.study_panel.project_selector

        self.middle_container = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_container)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        self.middle_layout.setSpacing(0)
        self.middle_layout.addWidget(self.data_explorer)
        self.middle_layout.addWidget(self.study_panel)

        # Right Side: Toolbars + Stacked Widget + Console
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        self.views: dict[str, QWidget] = {}
        self.stacked_widget = QStackedWidget()
        self._workspace: Optional[Workspace] = None
        self._active_project_name: Optional[str] = None
        self._current_selection: Optional[data_explorer.Selection] = None

        # Right Vertical Splitter (Stacked Widget on top, Console on bottom)
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setHandleWidth(1)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.addWidget(self.stacked_widget)
        
        self.console_widget = ConsoleWidget(self)
        self.console_widget.toggle()  # Start collapsed by default
        self.right_splitter.addWidget(self.console_widget)
        
        # Set initial splitter sizes (80% for Stacked Widget, 20% for Console)
        self.right_splitter.setSizes([4 * height // 5, height // 5])

        # Leftmost: Navigation Panel (instantiated after stacked_widget and views)
        self.navigation_panel = NavigationPanel(self.bridge, self.settings, self)

        # Add widgets to central horizontal splitter in the correct order!
        self.central_widget.addWidget(self.navigation_panel)
        self.central_widget.addWidget(self.middle_container)
        self.central_widget.addWidget(self.right_container)

        # Set initial splitter sizes (smallest width for Navigation Panel, 20% for Data Explorer, remaining for Right Container)
        nav_width = self.navigation_panel.sizeHint().width()
        self.central_widget.setSizes([nav_width, width // 5, width - nav_width - width // 5])

        # Configure stretch factors: right container stretches to take up all collapsed space
        self.central_widget.setStretchFactor(0, 0)
        self.central_widget.setStretchFactor(1, 0)
        self.central_widget.setStretchFactor(2, 1)

        self.statusBar().showMessage(_STATUS_BAR_WELCOME_MESSAGE)

        self.right_layout.addWidget(self.right_splitter)

        # --- Menu Bar ---
        self._create_menu_bar()

        self._connect_signals()
        self._connect_global_refresh_signals()

        self.set_workspace(self._get_workspace_dir())
        self.navigation_panel.restore_from_settings()
        
        # Apply navigation panel and data explorer visibility from settings
        show_navigation = self.settings.value("show_navigation_panel", True, type=bool)
        self.navigation_panel.setVisible(show_navigation)
        if hasattr(self, "_toggle_nav_action"):
            self._toggle_nav_action.setChecked(show_navigation)
        
        show_explorer = self.settings.value("show_data_explorer", True, type=bool)
        self.middle_container.setVisible(show_explorer)
        if hasattr(self, "_toggle_explorer_action"):
            self._toggle_explorer_action.setChecked(show_explorer)
        
        # Apply console visibility from settings
        show_console = self.settings.value("show_console", True, type=bool)
        self.console_widget.setVisible(show_console)
        if hasattr(self, "_toggle_console_action"):
            self._toggle_console_action.setChecked(show_console)
        
        logging.debug("Re-enabling UI signals after main window initialization.")
        MessageBus.get().blockSignals(False)
        
        # Switch to the initial checked task button
        checked_task_btn = self.navigation_panel.task_group.checkedButton()
        if checked_task_btn:
            view_name = checked_task_btn.property("view_name") or checked_task_btn.text()
            from whisker.gui.topics import gui as gui_topics
            self._on_switch_view("", gui_topics.SwitchViewRequest(view_name=view_name))
            
        logging.info("Main window setup complete.")

    def closeEvent(self, event: QCloseEvent):
        """Override to check for unsaved changes before closing."""
        if not self._prompt_if_dirty(origin_view=None):
            event.ignore()
        else:
            event.accept()

    def _create_menu_bar(self):
        """Creates the application menu bar with File, Edit, Selection, View, Tools, Help menus."""
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("File")
        
        self._new_proj_action = QAction("New Project...", self)
        self._new_proj_action.setShortcut(QKeySequence("Ctrl+N"))
        self._new_proj_action.triggered.connect(self.data_explorer.show_create_project_dialog)
        file_menu.addAction(self._new_proj_action)
        
        self._open_ws_action = QAction("Open Workspace...", self)
        self._open_ws_action.setShortcut(QKeySequence("Ctrl+O"))
        self._open_ws_action.triggered.connect(self._show_set_workspace_dialog)
        file_menu.addAction(self._open_ws_action)
        
        file_menu.addSeparator()
        
        self._new_ds_action = QAction("New Dataset...", self)
        self._new_ds_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self._new_ds_action.triggered.connect(self.data_explorer.show_create_dataset_dialog)
        file_menu.addAction(self._new_ds_action)
        
        self._import_labels_action = QAction("Import Pose Labels...", self)
        self._import_labels_action.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self._import_labels_action.triggered.connect(self.data_explorer.show_import_pose_labels_dialog)
        file_menu.addAction(self._import_labels_action)
        
        self._create_detector_action = QAction("Create Detector Dataset...", self)
        file_menu.addAction(self._create_detector_action)
        
        file_menu.addSeparator()
        
        self._refresh_ws_action = QAction("Refresh Workspace", self)
        self._refresh_ws_action.setShortcut(QKeySequence("F5"))
        self._refresh_ws_action.triggered.connect(self._refresh_workspace)
        file_menu.addAction(self._refresh_ws_action)
        
        self._proj_settings_action = QAction("Project Settings...", self)
        self._proj_settings_action.setShortcut(QKeySequence("Ctrl+Alt+P"))
        self._proj_settings_action.triggered.connect(self._show_project_settings)
        file_menu.addAction(self._proj_settings_action)
        
        file_menu.addSeparator()
        
        # Recent Workspaces Submenu
        self._recent_menu = QMenu("Recent Workspaces", self)
        file_menu.addMenu(self._recent_menu)
        self._rebuild_recent_workspaces_menu()
        
        file_menu.addSeparator()
        
        # Export Submenu
        export_menu = QMenu("Export", self)
        file_menu.addMenu(export_menu)
        
        self._export_labels_action = QAction("Export Behavior Labels...", self)
        self._export_labels_action.triggered.connect(self._export_behavior_labels)
        export_menu.addAction(self._export_labels_action)
        
        self._export_charts_action = QAction("Export Charts (.png)...", self)
        self._export_charts_action.triggered.connect(self._export_behavior_charts)
        export_menu.addAction(self._export_charts_action)
        
        self._export_jitter_action = QAction("Export Jitter Analysis (.png)...", self)
        self._export_jitter_action.triggered.connect(self._export_jitter_analysis)
        export_menu.addAction(self._export_jitter_action)
        
        self._export_bouts_action = QAction("Export Bouts (.json)...", self)
        self._export_bouts_action.triggered.connect(self._export_bouts)
        export_menu.addAction(self._export_bouts_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Edit Menu ---
        edit_menu = menu_bar.addMenu("Edit")
        self._settings_action = QAction("Settings...", self)
        self._settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self._settings_action.triggered.connect(self._show_settings_dialog)
        edit_menu.addAction(self._settings_action)

        # --- Selection Menu ---
        selection_menu = menu_bar.addMenu("Selection")
        
        self._prev_item_action = QAction("Previous Item / Image", self)
        self._prev_item_action.setShortcut(QKeySequence("Left"))
        self._prev_item_action.triggered.connect(self.data_explorer.select_previous_image)
        selection_menu.addAction(self._prev_item_action)
        
        self._next_item_action = QAction("Next Item / Image", self)
        self._next_item_action.setShortcut(QKeySequence("Right"))
        self._next_item_action.triggered.connect(self.data_explorer.select_next_image)
        selection_menu.addAction(self._next_item_action)
        
        selection_menu.addSeparator()
        
        self._toggle_blind_action = QAction("Blind Mode", self)
        self._toggle_blind_action.setCheckable(True)
        self._toggle_blind_action.setShortcut(QKeySequence("Ctrl+B"))
        self._toggle_blind_action.triggered.connect(self._on_menu_blind_mode_toggled)
        selection_menu.addAction(self._toggle_blind_action)

        # --- View Menu ---
        view_menu = menu_bar.addMenu("View")
        
        self._toggle_nav_action = QAction("Toggle Navigation Panel", self)
        self._toggle_nav_action.setCheckable(True)
        self._toggle_nav_action.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self._toggle_nav_action.triggered.connect(self._toggle_navigation_panel_visibility)
        view_menu.addAction(self._toggle_nav_action)
        
        self._toggle_explorer_action = QAction("Toggle Data Explorer", self)
        self._toggle_explorer_action.setCheckable(True)
        self._toggle_explorer_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self._toggle_explorer_action.triggered.connect(self._toggle_data_explorer_visibility)
        view_menu.addAction(self._toggle_explorer_action)
        
        self._toggle_console_action = QAction("Toggle Console", self)
        self._toggle_console_action.setCheckable(True)
        self._toggle_console_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._toggle_console_action.triggered.connect(self._toggle_console_visibility)
        view_menu.addAction(self._toggle_console_action)
        
        view_menu.addSeparator()
        
        # Active Workflow Submenu
        workflow_menu = QMenu("Active Workflow", self)
        view_menu.addMenu(workflow_menu)
        
        from whisker.core.workflows.workflow_enum import Workflow
        workflows = [
            Workflow.POSE_ESTIMATION,
            Workflow.BEHAVIOR_CLASSIFICATION,
        ]
        
        self._workflow_actions = {}
        from PyQt6.QtGui import QActionGroup
        wf_group = QActionGroup(self)
        for wf in workflows:
            wf_name = wf.to_display_name()
            action = QAction(wf_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, name=wf_name: self._set_active_workflow_by_name(name))
            wf_group.addAction(action)
            workflow_menu.addAction(action)
            self._workflow_actions[wf_name] = action
            
        # Go To Task Submenu
        goto_menu = QMenu("Go To Task", self)
        view_menu.addMenu(goto_menu)
        
        tasks_list = [
            ("Welcome", "Ctrl+1", "❖ Welcome"),
            ("Info", "Ctrl+2", "❖ Info"),
            ("Jobs", "Ctrl+3", "❖ Jobs"),
            ("Label", "Ctrl+5", "❖ Label")
        ]
        for label, shortcut, view_name in tasks_list:
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda checked, name=view_name: self._switch_to_task_for_current_workflow(name))
            goto_menu.addAction(action)
            
        view_menu.addSeparator()
        
        # Theme Submenu
        theme_menu = QMenu("Style / Theme", self)
        view_menu.addMenu(theme_menu)
        
        theme_group = QActionGroup(self)
        self._theme_actions = {}
        
        themes = [
            "Follow Active Workflow",
            "Slate Blue",
            "Lilac Mist",
            "Sage Forest",
            "Peach Breeze",
            "Harvest Gold"
        ]
        for theme in themes:
            action = QAction(theme, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, name=theme: self._set_theme_selection(name))
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme] = action
            
        theme_menu.addSeparator()
        self._toggle_money_action = QAction("$$$$ Money Mode $$$", self)
        self._toggle_money_action.setCheckable(True)
        self._toggle_money_action.triggered.connect(self._toggle_money_mode_directly)
        theme_menu.addAction(self._toggle_money_action)

        # --- Tools Menu ---
        tools_menu = menu_bar.addMenu("Tools")
        
        tools_menu.addSeparator()
        
        self._run_jobs_action = QAction("Active Workers Monitor...", self)
        self._run_jobs_action.triggered.connect(lambda: self._switch_to_task_for_current_workflow("❖ Jobs"))
        tools_menu.addAction(self._run_jobs_action)

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("Help")
        
        doc_action = QAction("Documentation", self)
        doc_action.setShortcut(QKeySequence("F1"))
        doc_action.triggered.connect(self._show_help_window)
        help_menu.addAction(doc_action)
        
        help_menu.addSeparator()
        
        sys_info_action = QAction("System Diagnostics...", self)
        sys_info_action.triggered.connect(self._show_system_diagnostics)
        help_menu.addAction(sys_info_action)
        
        view_log_action = QAction("View Log File...", self)
        view_log_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        view_log_action.triggered.connect(self._view_log_file)
        help_menu.addAction(view_log_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About WHISKER...", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)


    def _toggle_console_visibility(self):
        """Toggles the visibility of the console widget and updates settings."""
        is_visible = self.console_widget.isVisible()
        self.console_widget.setVisible(not is_visible)
        self.settings.setValue("show_console", not is_visible)
        logging.info(f"Console visibility toggled to: {'Visible' if not is_visible else 'Hidden'}")

    def _show_settings_dialog(self):
        dialog = SettingsDialog(settings=self.settings, parent=self)
        logging.info("Opening settings dialog.")
        if dialog.exec():
            logging.info("Settings updated. Refreshing UI based on new settings.")
            self.navigation_panel.update_animal_detection_visibility()
            
            # Apply theme and sync checked states
            self._apply_current_theme()
            
            # Apply money mode to all views
            enable_money_mode = self.settings.value("enable_money_mode", False, type=bool)
            for view in self.views.values():
                if hasattr(view, "apply_money_mode"):
                    view.apply_money_mode(enable_money_mode)
            
            # Update console visibility and action check state
            show_console = self.settings.value("show_console", True, type=bool)
            self.console_widget.setVisible(show_console)
            if hasattr(self, "_toggle_console_action"):
                self._toggle_console_action.setChecked(show_console)
            
            self._update_menu_actions_state()
        else:
            logging.info("Settings dialog canceled. No changes applied.")

    def _show_create_project_dialog(self):
        # This is now handled through the Data Explorer's own actions
        pass

    def _show_set_workspace_dialog(self):
        if not self._prompt_if_dirty():
            return

        dialog = QFileDialog(
            self, "Select Workspace Directory", directory=str(self._get_workspace_dir())
        )
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                workspace_path = Path(selected_dirs[0])
                self.set_workspace(workspace_path)

    def _connect_signals(self):
        # --- Project and Data Explorer ---
        self.active_project_changed.connect(self._on_project_changed)
        self.active_project_changed.connect(self.data_explorer.set_active_project)
        self.active_project_changed.connect(self._update_menu_actions_state)
        self.project_selector.currentIndexChanged.connect(self._on_project_selected)
        self.data_explorer.toggled.connect(self.study_panel.setVisible)
        self.data_explorer.animation.valueChanged.connect(self._on_data_explorer_animation_value)
        self.navigation_panel.animation.valueChanged.connect(self._on_navigation_panel_animation_value)
        self.console_widget.animation.valueChanged.connect(self._on_console_animation_value)

        self.data_explorer.item_selected.connect(
            self._on_data_explorer_item_selected
        )
        self.data_explorer.item_selected.connect(self._update_selection_statusbar)
        self.data_explorer.item_selected.connect(lambda selection: self._update_menu_actions_state())
        self.data_explorer.item_deselected.connect(
            lambda: self._update_selection_statusbar(None)
        )
        self.data_explorer.item_deselected.connect(self._update_menu_actions_state)
        
        # Connect blind mode status from Data Explorer back to our action check state
        self.data_explorer.blind_mode_checkbox.toggled.connect(self._toggle_blind_action.setChecked)
        
        def _on_blind_mode_toggled(checked: bool):
            if checked:
                self.statusBar().showMessage("Selection: — (Hidden because Blind Mode is Enabled)")
        self.data_explorer.blind_mode_toggled.connect(_on_blind_mode_toggled)

    def _on_data_explorer_animation_value(self, value):
        sizes = self.central_widget.sizes()
        if len(sizes) > 1:
            sizes[1] = value
            self.central_widget.setSizes(sizes)

    def _on_navigation_panel_animation_value(self, value):
        sizes = self.central_widget.sizes()
        if len(sizes) > 0:
            sizes[0] = value
            self.central_widget.setSizes(sizes)

    def _on_console_animation_value(self, value):
        sizes = self.right_splitter.sizes()
        if len(sizes) > 1:
            sizes[1] = value
            self.right_splitter.setSizes(sizes)

    def _connect_global_refresh_signals(self):
        """Connects global refresh requests to workspace scan methods."""
        bus = MessageBus.get()
        bus.subscribe("request/workspace/projects/refresh", lambda t, p: self._on_refresh_projects())
        bus.subscribe("request/workspace/datasets/refresh", lambda t, p: self._on_refresh_datasets())
        bus.subscribe("request/workspace/labels/refresh", lambda t, p: self._on_refresh_labels())
        bus.subscribe("request/workspace/models/refresh", lambda t, p: self._on_refresh_models())
        bus.subscribe("request/workspace/predictions/refresh", lambda t, p: self._on_refresh_predictions())

        # Auto-refresh predictions when granular updates arrive from workers
        bus.subscribe("workspace/prediction/pose/video_completed", lambda t, p: self._on_refresh_predictions())
        bus.subscribe("workspace/prediction/pose/dataset_completed", lambda t, p: self._on_refresh_predictions())
        
        # Handle workspace set request from Settings dialog
        bus.subscribe("request/workspace/set", lambda t, p: self.set_workspace(Path(p["path"])))

        # Update action states on model run or workflow changes
        bus.subscribe("selection/model_run/changed", lambda t, p: self._update_menu_actions_state())
        bus.subscribe("gui/request/workflow_selected", lambda t, p: self._update_menu_actions_state())

    def _on_refresh_projects(self):
        if self._workspace:
            self._workspace.scan_projects()
            MessageBus.get().publish("workspace/projects/refreshed")
            self._update_project_selector()

    def _on_refresh_datasets(self):
        if self._workspace:
            self._workspace.scan_datasets()
            MessageBus.get().publish("workspace/datasets/refreshed")

    def _on_refresh_labels(self):
        if self._workspace:
            self._workspace.scan_labels()
            MessageBus.get().publish("workspace/labels/refreshed")

    def _on_refresh_models(self):
        if self._workspace:
            self._workspace.scan_models()
            MessageBus.get().publish("workspace/models/refreshed")

    def _on_refresh_predictions(self):
        if self._workspace:
            self._workspace.scan_predictions()
            MessageBus.get().publish("workspace/predictions/refreshed")

    def _update_project_selector(self):
        """Refreshes the project dropdown without changing the selection if possible."""
        if not self._workspace: return
        
        current_text = self.project_selector.currentText()
        self.project_selector.blockSignals(True)
        self.project_selector.clear()
        
        project_names = sorted(self._workspace.projects.keys())
        self.project_selector.addItems([""] + project_names)
        
        idx = self.project_selector.findText(current_text)
        if idx >= 0:
            self.project_selector.setCurrentIndex(idx)
        
        self.project_selector.setEnabled(bool(project_names))
        self.project_selector.blockSignals(False)

    def _on_labels_saved(self, dataset_name: str, media_path: str):
        MessageBus.get().publish("request/workspace/labels/refresh")

    def _on_register_view(self, topic: str, request: Any):
        view_name = request.view_name
        widget = request.widget
        
        self.stacked_widget.addWidget(widget)
        self.views[view_name] = widget
        
        # Connect signals if it is a specific view
        if view_name == "❖ Welcome":
            widget.request_set_workspace.connect(self._show_set_workspace_dialog)
            widget.request_create_project.connect(self.data_explorer.show_create_project_dialog)
            widget.request_create_dataset.connect(self.data_explorer.show_create_dataset_dialog)
            widget.request_import_labels.connect(self.data_explorer.show_import_pose_labels_dialog)
            
        # Connect general signals if present
        if hasattr(widget, "dirty_state_changed"):
            widget.dirty_state_changed.connect(self._on_any_view_dirty_state_changed)
        if hasattr(widget, "request_launch_worker"):
            widget.request_launch_worker.connect(self._on_request_launch_worker)
        if hasattr(widget, "labels_saved"):
            widget.labels_saved.connect(self._on_labels_saved)
        if hasattr(widget, "media_selected"):
            widget.media_selected.connect(self.data_explorer.select_item_by_path)
        if hasattr(widget, "request_select_prev_image"):
            widget.request_select_prev_image.connect(self.data_explorer.select_previous_image)
        if hasattr(widget, "request_select_next_image"):
            widget.request_select_next_image.connect(self.data_explorer.select_next_image)

        # Set workspace and project if already loaded
        if hasattr(widget, "set_workspace"):
            widget.set_workspace(self._workspace)
            widget.set_dirty(False)
        if hasattr(widget, "set_project"):
            widget.set_project(self._workspace.projects.get(self._active_project_name) if self._workspace and self._active_project_name else None)
            widget.set_dirty(False)
        if hasattr(widget, "apply_money_mode"):
            enable_money_mode = self.settings.value("enable_money_mode", False, type=bool)
            widget.apply_money_mode(enable_money_mode)

    def _on_switch_view(self, topic: str, request: Any):
        view_name = request.view_name
        if view_name not in self.views:
            return

        target_widget = self.views[view_name]
        current_widget = self.stacked_widget.currentWidget()
        
        if current_widget == target_widget:
            return
        
        if current_widget and current_widget != target_widget:
            if not self._prompt_if_dirty(origin_view=current_widget):
                # Revert selection in navigation panel
                current_view_name = next((k for k, v in self.views.items() if v == current_widget), None)
                if current_view_name:
                    self.navigation_panel.select_view(current_view_name)
                return

        # Perform switch
        self.stacked_widget.setCurrentWidget(target_widget)
        
        # Clean the view name for StudyPanel (strip icon prefix and workflow suffix)
        clean_tab_name = view_name
        if clean_tab_name.startswith("❖ "):
            clean_tab_name = clean_tab_name[2:]
        if "(" in clean_tab_name:
            clean_tab_name = clean_tab_name.split("(")[0].strip()
        self.study_panel.set_current_tab(clean_tab_name)
        
        # Adjust size policies (Expanding for active, Ignored for others)
        for w in self.views.values():
            if w == target_widget:
                w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            else:
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.stacked_widget.adjustSize()
        
        # Publish model run selection reset if switching to non-run views
        is_model_run_view = any(
            x in view_name for x in ["Predict", "Amend", "Evaluate", "Debug"]
        )
        if not is_model_run_view:
            MessageBus.get().publish("selection/model_run/changed", {"name": ""})

        # Apply current selection to the switched view
        if self._current_selection and hasattr(target_widget, "on_data_explorer_item_selected"):
            target_widget.on_data_explorer_item_selected(self._current_selection)

        # Update available data explorer item groups
        if hasattr(target_widget, "get_data_explorer_item_groups"):
            self.data_explorer.show_item_groups(target_widget.get_data_explorer_item_groups())

    def _on_workflow_selected_telemetry(self, topic: str, telemetry: Any):
        workflow = Workflow(telemetry.name)
        self.data_explorer.set_workflow(workflow)
        self.study_panel.set_workflow(workflow)
        
        # Pass the workflow selection change to all views
        for view in self.views.values():
            if hasattr(view, "set_active_workflow"):
                view.set_active_workflow(workflow)
        
        # Apply theme
        self._apply_current_theme()
        
        # Update checked state in menu
        display_name = workflow.to_display_name()
        if hasattr(self, "_workflow_actions") and display_name in self._workflow_actions:
            self._workflow_actions[display_name].setChecked(True)

    def _get_active_dataset_and_video(self) -> tuple[Optional[str], Optional[str]]:
        if not self._current_selection:
            return None, None
        
        from whisker.gui.widgets.data_explorer.constants import ItemGroupEnum
        if self._current_selection.group != ItemGroupEnum.DATASETS:
            return None, None
            
        hierarchy = self._current_selection.item
        if len(hierarchy) < 2:
            return None, None
            
        dataset_name = hierarchy[1]
        video_relative_path = hierarchy[2] if len(hierarchy) > 2 else None
        return dataset_name, video_relative_path

    def _update_menu_actions_state(self):
        if not hasattr(self, "_new_proj_action"):
            return
            
        has_workspace = self._workspace is not None
        has_project = bool(self._active_project_name)
        
        # File Menu Actions
        self._new_proj_action.setEnabled(has_workspace)
        self._open_ws_action.setEnabled(True)
        self._new_ds_action.setEnabled(has_workspace and has_project)
        self._import_labels_action.setEnabled(has_workspace and has_project)
        self._create_detector_action.setEnabled(has_workspace and has_project)
        self._refresh_ws_action.setEnabled(has_workspace)
        self._proj_settings_action.setEnabled(has_workspace and has_project)
        
        # Export Actions
        dataset_name, _ = self._get_active_dataset_and_video()
        has_dataset = dataset_name is not None
        
        model_run = ""
        if hasattr(self.data_explorer, "action_handler"):
            model_run = self.data_explorer.action_handler._current_model_run or ""
        has_model_run = bool(model_run)
        
        self._export_labels_action.setEnabled(has_workspace and has_dataset)
        self._export_charts_action.setEnabled(has_workspace and has_dataset and has_model_run)
        self._export_jitter_action.setEnabled(has_workspace and has_dataset and has_model_run)
        self._export_bouts_action.setEnabled(has_workspace and has_dataset and has_model_run)
        
        # Selection Actions
        self._prev_item_action.setEnabled(has_workspace)
        self._next_item_action.setEnabled(has_workspace)
        self._toggle_blind_action.setEnabled(True)
        
        # View Actions
        checked_wf_btn = self.navigation_panel.workflow_group.checkedButton()
        if checked_wf_btn:
            current_wf_name = checked_wf_btn.text()
            for wf_name, action in self._workflow_actions.items():
                action.setChecked(wf_name == current_wf_name)
        
        self._update_theme_menu_checked_state()

    def _show_project_settings(self):
        from whisker.gui.topics import gui as gui_topics
        self.navigation_panel.select_view("❖ Projects")
        self._on_switch_view("", gui_topics.SwitchViewRequest(view_name="❖ Projects"))

    def _refresh_workspace(self):
        if not self._workspace:
            return
        logging.info("Refreshing workspace from menu...")
        bus = MessageBus.get()
        bus.publish("request/workspace/projects/refresh")
        bus.publish("request/workspace/datasets/refresh")
        bus.publish("request/workspace/labels/refresh")
        bus.publish("request/workspace/models/refresh")
        bus.publish("request/workspace/predictions/refresh")

    def _export_behavior_labels(self):
        dataset_name, _ = self._get_active_dataset_and_video()
        if dataset_name and hasattr(self.data_explorer, "action_handler"):
            self.data_explorer.action_handler._export_behavior_labels(dataset_name)

    def _export_behavior_charts(self):
        dataset_name, _ = self._get_active_dataset_and_video()
        if dataset_name and hasattr(self.data_explorer, "action_handler"):
            self.data_explorer.action_handler._on_export_behavior_charts(dataset_name)

    def _export_jitter_analysis(self):
        dataset_name, _ = self._get_active_dataset_and_video()
        if dataset_name and hasattr(self.data_explorer, "action_handler"):
            self.data_explorer.action_handler._on_export_jitter_analysis(dataset_name)

    def _export_bouts(self):
        dataset_name, _ = self._get_active_dataset_and_video()
        if dataset_name and hasattr(self.data_explorer, "action_handler"):
            self.data_explorer.action_handler._on_export_bouts(dataset_name)

    def _update_recent_workspaces(self, workspace_path: Path):
        recent = self.settings.value("recent_workspaces", [])
        if not isinstance(recent, list):
            recent = []
        path_str = str(workspace_path.resolve())
        if path_str in recent:
            recent.remove(path_str)
        recent.insert(0, path_str)
        recent = recent[:5]
        self.settings.setValue("recent_workspaces", recent)
        self._rebuild_recent_workspaces_menu()

    def _rebuild_recent_workspaces_menu(self):
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        recent = self.settings.value("recent_workspaces", [])
        if not recent:
            no_recent_action = self._recent_menu.addAction("No Recent Workspaces")
            no_recent_action.setEnabled(False)
            return
        
        for path_str in recent:
            action = self._recent_menu.addAction(path_str)
            action.triggered.connect(lambda checked, p=path_str: self._open_recent_workspace(p))

    def _open_recent_workspace(self, path_str: str):
        path = Path(path_str)
        if path.exists():
            self.set_workspace(path)
        else:
            QMessageBox.warning(
                self,
                "Workspace Not Found",
                f"The directory '{path_str}' no longer exists."
            )
            recent = self.settings.value("recent_workspaces", [])
            if path_str in recent:
                recent.remove(path_str)
                self.settings.setValue("recent_workspaces", recent)
                self._rebuild_recent_workspaces_menu()

    def _on_menu_blind_mode_toggled(self, checked: bool):
        self.data_explorer.blind_mode_checkbox.setChecked(checked)

    def _toggle_navigation_panel_visibility(self):
        is_visible = self.navigation_panel.isVisible()
        self.navigation_panel.setVisible(not is_visible)
        self.settings.setValue("show_navigation_panel", not is_visible)
        self._toggle_nav_action.setChecked(not is_visible)
        logging.info(f"Navigation Panel visibility toggled to: {'Visible' if not is_visible else 'Hidden'}")

    def _toggle_data_explorer_visibility(self):
        is_visible = self.middle_container.isVisible()
        self.middle_container.setVisible(not is_visible)
        self.settings.setValue("show_data_explorer", not is_visible)
        self._toggle_explorer_action.setChecked(not is_visible)
        logging.info(f"Data Explorer visibility toggled to: {'Visible' if not is_visible else 'Hidden'}")

    def _set_active_workflow_by_name(self, name: str):
        for btn in self.navigation_panel.workflow_group.buttons():
            if btn.text() == name:
                btn.setChecked(True)
                self.navigation_panel._on_workflow_selected(btn)
                break

    def _switch_to_task_for_current_workflow(self, base_task_name: str):
        workflow_tasks = ["❖ Amend", "❖ Label", "❖ Predict", "❖ Evaluate"]
        if base_task_name in workflow_tasks:
            checked_wf_btn = self.navigation_panel.workflow_group.checkedButton()
            if checked_wf_btn:
                wf_name = checked_wf_btn.text()
                view_name = f"{base_task_name} ({wf_name})"
            else:
                return
        else:
            view_name = base_task_name
            
        from whisker.gui.topics import gui as gui_topics
        self.navigation_panel.select_view(view_name)
        self._on_switch_view("", gui_topics.SwitchViewRequest(view_name=view_name))

    def _apply_current_theme(self):
        theme_setting = self.settings.value("theme_selection", "Follow Active Workflow", type=str)
        money_mode_enabled = self.settings.value("enable_money_mode", False, type=bool)
        
        target_style = None
        if money_mode_enabled:
            target_style = "Harvest Gold"
        elif theme_setting == "Follow Active Workflow":
            checked_wf_btn = self.navigation_panel.workflow_group.checkedButton()
            if checked_wf_btn:
                workflow_name = checked_wf_btn.text()
                workflow_themes = {
                    "Pose Estimation": "Lilac Mist",
                    "Behavior Classification": "Sage Forest",
                }
                target_style = workflow_themes.get(workflow_name)
        else:
            target_style = theme_setting
            
        if target_style:
            from whisker.gui.utils.themes import THEMES, STYLESHEET_TEMPLATE
            theme_config = THEMES[target_style]
            self.setStyleSheet(STYLESHEET_TEMPLATE.format(**theme_config))

    def _set_theme_selection(self, theme_name: str):
        self.settings.setValue("theme_selection", theme_name)
        self._apply_current_theme()
        self._update_theme_menu_checked_state()

    def _toggle_money_mode_directly(self, checked: bool):
        self.settings.setValue("enable_money_mode", checked)
        self._apply_current_theme()
        for view in self.views.values():
            if hasattr(view, "apply_money_mode"):
                view.apply_money_mode(checked)
        self._toggle_money_action.setChecked(checked)
        self._update_theme_menu_checked_state()

    def _update_theme_menu_checked_state(self):
        if not hasattr(self, "_theme_actions"):
            return
        theme_setting = self.settings.value("theme_selection", "Follow Active Workflow", type=str)
        for name, action in self._theme_actions.items():
            action.setChecked(name == theme_setting)
        
        money_mode_enabled = self.settings.value("enable_money_mode", False, type=bool)
        self._toggle_money_action.setChecked(money_mode_enabled)


    def _show_system_diagnostics(self):
        import platform
        import sys
        
        torch_status = "Not Installed"
        cuda_status = "N/A"
        try:
            import torch
            torch_status = torch.__version__
            cuda_status = "Available (CUDA Enabled)" if torch.cuda.is_available() else "Not Available (CPU Only)"
        except ImportError:
            pass
            
        tf_status = "Not Installed"
        tf_gpus = "N/A"
        try:
            import tensorflow as tf
            tf_status = tf.__version__
            gpus = tf.config.list_physical_devices('GPU')
            tf_gpus = f"Available ({len(gpus)} GPU(s) found)" if gpus else "Not Available (CPU Only)"
        except ImportError:
            pass
            
        msg = (
            f"<b>WHISKER System Diagnostics</b><br><br>"
            f"<b>Operating System:</b> {platform.system()} {platform.release()}<br>"
            f"<b>Python Version:</b> {sys.version.split()[0]}<br><br>"
            f"<b>PyTorch Version:</b> {torch_status}<br>"
            f"<b>CUDA Status (PyTorch):</b> {cuda_status}<br><br>"
            f"<b>TensorFlow Version:</b> {tf_status}<br>"
            f"<b>GPU Status (TensorFlow):</b> {tf_gpus}<br>"
        )
        QMessageBox.information(self, "System Diagnostics", msg)

    def _view_log_file(self):
        import os
        import logging
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        
        root = logging.getLogger()
        log_file = None
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename
                break
                
        if log_file and os.path.exists(log_file):
            QDesktopServices.openUrl(QUrl.fromLocalFile(log_file))
        else:
            QMessageBox.information(
                self,
                "No Log File Found",
                "There is no active session log file for this workspace currently running."
            )

    def _show_about_dialog(self):
        msg = (
            "<h3>WHISKER</h3>"
            "Workbench for Holistic Insights via Skeletal Kinematics & Event Recognition<br><br>"
            "Version 1.0.0<br><br>"
            "A hybrid ML workbench for animal pose estimation and behavior classification.<br><br>"
            "&copy; 2026 WHISKER Developers"
        )
        QMessageBox.about(self, "About WHISKER", msg)

    def _on_request_launch_worker(self, worker_name: str, worker: Any):
        from whisker.gui.job_manager import JobManager
        JobManager.get().submit_worker(worker_name, worker)
        logging.info(f"Worker {worker_name} submitted to JobManager.")

    def _get_dirty_views(self) -> list:
        return [view for view in self.views.values() if hasattr(view, "is_dirty") and view.is_dirty()]

    def _prompt_if_dirty(self, origin_view: Optional[QWidget] = None) -> bool:
        dirty_views = self._get_dirty_views()
        if origin_view:
            dirty_views = [origin_view] if hasattr(origin_view, "is_dirty") and origin_view.is_dirty() else []

        if not dirty_views:
            return True

        if origin_view:
            view_name = next((k for k, v in self.views.items() if v == origin_view), "View")
            message = f"You have unsaved changes in the '{view_name}' view. Do you want to proceed without saving?"
        else:
            message = "You have unsaved changes. Are you sure you want to quit?"

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return reply == QMessageBox.StandardButton.Yes

    def _on_any_view_dirty_state_changed(self, dirty: bool):
        self._update_window_title()

    def _update_window_title(self):
        """Appends a '*' to the window title if any views are dirty."""
        dirty_views = self._get_dirty_views()
        if dirty_views:
            self.setWindowTitle(f"{self._original_window_title}*")
        else:
            self.setWindowTitle(self._original_window_title)

    def _on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        self._current_selection = selection
        active_view = self.stacked_widget.currentWidget()
        if active_view and hasattr(active_view, "on_data_explorer_item_selected"):
            active_view.on_data_explorer_item_selected(selection)

    def _on_project_changed(self, project: Optional[Any]):
        for view in self.views.values():
            if hasattr(view, "set_project"):
                view.set_project(project)
                view.set_dirty(False)

    ############################################################################

    def set_window_icon(self):
        icon_path = ASSETS_DIR / "favicon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logging.warning(f"Window icon 'favicon.ico' not found at {icon_path}")

    ############################################################################

    def set_workspace(self, workspace_path: Path):
        try:
            self._workspace = Workspace(workspace_path)
            self._set_workspace_dir(workspace_path)
            self._update_recent_workspaces(workspace_path)
        except Exception as _:
            logging.error(f"Error loading workspace: {traceback.format_exc()}")
            new_workspace = Workspace.create(
                workspace_path, warn_if_exists=WarnIfExistsDialog.run
            )
            if new_workspace:
                self._workspace = new_workspace
                self._set_workspace_dir(workspace_path)
                self._update_recent_workspaces(workspace_path)

        self._on_workspace_changed()

    def _get_workspace_dir(self) -> Path:
        return Path(self.settings.value("workspace_dir", str(Path.cwd())))

    def _set_workspace_dir(self, browse_dir: Path):
        self.settings.setValue("workspace_dir", str(browse_dir))
        get_server_manager().set_logging_dir(
            browse_dir / 'third_party' / 'logs'
        )

    def _on_workspace_changed(self):
        self.data_explorer.update_workspace(self._workspace)
        self.study_panel.set_workspace(self._workspace)
        for view in self.views.values():
            if hasattr(view, "set_workspace"):
                view.set_workspace(self._workspace)
                view.set_dirty(False)
        self._update_window_title()

        self.project_selector.blockSignals(True)
        self.project_selector.clear()
        if self._workspace:
            project_names = sorted(self._workspace.projects.keys())
            if project_names:
                self.project_selector.addItems([""] + project_names)
            self.project_selector.setEnabled(bool(project_names))
        else:
            self.project_selector.setEnabled(False)
        self.project_selector.blockSignals(False)

        # --- Restore Active Project ---
        last_project = self.settings.value("active_project", "")
        if last_project and self._workspace and last_project in self._workspace.projects.keys():
            all_projects = [""] + sorted(self._workspace.projects.keys())
            try:
                idx = all_projects.index(last_project)
                self.project_selector.setCurrentIndex(idx)
            except ValueError:
                self._on_project_selected(0)
        else:
            self._on_project_selected(self.project_selector.currentIndex())

    def _on_project_selected(self, index: int):
        if not self._prompt_if_dirty():
            if self._active_project_name:
                try:
                    all_projects = [""] + sorted(self._workspace.projects.keys())
                    previous_index = all_projects.index(self._active_project_name)
                    self.project_selector.blockSignals(True)
                    self.project_selector.setCurrentIndex(previous_index)
                    self.project_selector.blockSignals(False)
                except (ValueError, IndexError):
                    pass
            return

        project_name = self.project_selector.currentText()
        self.settings.setValue("active_project", project_name)

        if not self._workspace or not project_name:
            self._active_project_name = None
            self.active_project_changed.emit(None)
            return

        if self._active_project_name != project_name:
            self._active_project_name = project_name
            project = self._workspace.projects.get(project_name)
            logging.info(f"Active project changed to: '{project_name}'")
            self.active_project_changed.emit(project)

    def _show_help_window(self):
        """Initializes and shows the lightweight help/documentation window."""
        self._help_window = HelpWindow(self)
        self._help_window.show()

    def _update_selection_statusbar(self, selection: data_explorer.Selection | None):
        """Updates the window status bar with the current Data Explorer selection."""
        if self.data_explorer.is_blind_mode_enabled():
            return

        if selection is None:
            if self.statusBar().currentMessage() != _STATUS_BAR_WELCOME_MESSAGE:
                self.statusBar().showMessage("No data explorer item selected.")
            return

        self.statusBar().showMessage(f"Selection: — {' > '.join(selection.item)}")


