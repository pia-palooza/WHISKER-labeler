import logging
from typing import Optional
from PyQt6.QtCore import pyqtSignal, QSize, QSettings
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QApplication,
    QButtonGroup,
    QMainWindow,
)
from PyQt6.QtGui import QIcon

from whisker.gui.constants import ASSETS_DIR
from whisker.gui.tabs import  Workflow
from whisker.gui.utils.themes import THEMES, STYLESHEET_TEMPLATE
    
class WorkflowSelectionWidget(QButtonGroup):
    workflow_changed = pyqtSignal(Workflow)

    def __init__(self, toolbar, settings):
        super().__init__()
        self.setExclusive(True)
        self.settings = settings
        self.idClicked.connect(self._on_workflow_button_clicked)

        self._buttons: dict[Workflow, QPushButton] = {}
        self._workflows_by_ids: dict[int, Workflow] = {}
        self._themes_by_workflow: dict[Workflow, str] = {}
        self._toolbar = toolbar
        self._toolbar.addWidget(QLabel("   Workflow:   "))
        self._curr_style_sheet: Optional[str] = None

        for workflow, style_sheet in [
            (Workflow.POSE_ESTIMATION, "Lilac Mist"),
            (Workflow.BEHAVIOR_CLASSIFICATION, "Sage Forest"),
        ]:
            self._add_workflow(workflow, style_sheet)
    
    # ... (update_animal_detection_visibility and _add_workflow remain unchanged) ...

    def _add_workflow(self, workflow: Workflow, style_sheet: str):
        workflow_name = workflow.to_display_name().replace(' ', '_').lower()
        workflow_icon_path = str(ASSETS_DIR / f"{workflow_name}_icon.png")

        new_button = QPushButton(workflow.to_display_name())
        new_button.setCheckable(True)
        new_button.setIcon(QIcon(workflow_icon_path)) 
        new_button.setIconSize(QSize(24, 24))
        new_button.setToolTip(f"Switch to {workflow.to_display_name()} workflow.")

        button_id = len(self._workflows_by_ids)
        self._workflows_by_ids[button_id] = workflow
        self._buttons[workflow] = new_button
        self._themes_by_workflow[workflow] = style_sheet
        self.addButton(new_button, id=button_id)
        
        # Capture the action created by the toolbar
        action = self._toolbar.addWidget(new_button)
        # Store the action so we can hide/show it later
        new_button.setProperty("toolbar_action", action)

    def restore_active_workflow(self):
        last_workflow_val = self.settings.value("active_workflow", None)
        target_workflow = Workflow.POSE_ESTIMATION # Default to start of pipeline

        if last_workflow_val:
            for wf in Workflow:
                if wf.value == last_workflow_val:
                    target_workflow = wf
                    break

        workflow_button = self._buttons.get(target_workflow)
        if workflow_button:
            workflow_button.setChecked(True)
        
        main_window = self._find_main_window()
        if main_window:
            main_window.setUpdatesEnabled(False)
        try:
            self._apply_workflow_theme(target_workflow)
            self.workflow_changed.emit(target_workflow)
        finally:
            if main_window:
                main_window.setUpdatesEnabled(True)
                main_window.update()
    
    def _on_workflow_button_clicked(self, id: int):
        workflow = self._workflows_by_ids.get(id)
        if workflow:
            main_window = self._find_main_window()
            if main_window:
                main_window.setUpdatesEnabled(False)
            
            try:
                self.settings.setValue("active_workflow", workflow.value)
                self._apply_workflow_theme(workflow)
                self.workflow_changed.emit(workflow)
            finally:
                if main_window:
                    main_window.setUpdatesEnabled(True)
                    main_window.update()

    def _find_main_window(self) -> Optional[QMainWindow]:
        """Finds the QMainWindow by traversing up from the toolbar."""
        window = self._toolbar.window()
        if isinstance(window, QMainWindow):
            return window
        
        # Fallback for early initialization
        temp = self._toolbar.parentWidget()
        while temp:
            if isinstance(temp, QMainWindow):
                return temp
            temp = temp.parentWidget()
        return None

    def _apply_workflow_theme(self, workflow: Optional[Workflow] = None):
        """Applies the application stylesheet based on the workflow."""
        main_window = self._find_main_window()
        if not main_window:
            logging.warning("Could not find MainWindow to apply theme.")
            return

        money_mode_enabled = self.settings.value("enable_money_mode", False, type=bool)
        if money_mode_enabled:
            target_style = "Harvest Gold"
        else:
            workflow = workflow or self._workflows_by_ids.get(self.checkedId())
            target_style = self._themes_by_workflow.get(workflow)

        if not target_style or target_style == self._curr_style_sheet:
            return

        
        theme_config = THEMES[target_style]
        main_window.setStyleSheet(STYLESHEET_TEMPLATE.format(**theme_config))
        self._curr_style_sheet = target_style
