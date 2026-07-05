# UPDATE_FILE: gui/tabs/labeling_tab.py
import logging
from pathlib import Path
from typing import cast, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
)

from whisker.core.workspace import DatasetType, Workspace, Project
from whisker.gui.widgets import data_explorer
from whisker.gui.workflows.pose_estimation.tabs import LabelingPosesTab
from whisker.gui.workflows.behavior_classification.tabs import LabelingBehaviorsTab
from .base_tab import BaseTab, Workflow


class LabelingTab(BaseTab):
    labels_saved = pyqtSignal(str, str)
    request_data_explorer_selection = pyqtSignal(Path)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_selection: Optional[data_explorer.Selection] = None  # <--- Added state tracking

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        self.tab_widget.tabBar().setVisible(False)

        # --- Create and add the sub-tabs ---
        self.poses_view = LabelingPosesTab(self)
        self.tab_widget.addTab(self.poses_view, "Poses")

        self.behaviors_view = LabelingBehaviorsTab(self)
        self.tab_widget.addTab(self.behaviors_view, "Behaviors")


        self._connect_signals()

    def _connect_signals(self):
        """Connect signals from subtabs to this parent tab."""
        self.poses_view.dirty_state_changed.connect(self.set_dirty)
        self.behaviors_view.dirty_state_changed.connect(self.set_dirty)

        self.poses_view.labels_saved.connect(self.labels_saved)
        self.poses_view.media_selected.connect(self.request_data_explorer_selection)
        self.poses_view.request_select_next_image.connect(self.request_select_next_image)
        self.poses_view.request_select_prev_image.connect(self.request_select_prev_image)
        self.poses_view.request_launch_worker.connect(self.request_launch_worker)
        self.behaviors_view.labels_saved.connect(self.labels_saved)
        self.behaviors_view.media_selected.connect(self.request_data_explorer_selection)

    def set_workspace(self, workspace: Optional[Workspace]):
        super().set_workspace(workspace)
        self.poses_view.set_workspace(workspace)
        self.behaviors_view.set_workspace(workspace)
        self.set_dirty(False)

    def set_project(self, project: Optional[Project]):
        """Pass the active project down to the visible labeling sub-tab."""
        super().set_project(project)
        self.poses_view.set_project(project)
        self.behaviors_view.set_project(project)
        self.set_dirty(False)

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        """
        Switches the visible labeling UI based on the selected item.
        Handles files, collections, or no selection.
        """
        self._current_selection = selection  # <--- Store selection

        if any(
            (
                not self._workspace,
                not self._project,
                selection.group != data_explorer.ItemGroupEnum.DATASETS,
                not selection.item,
                len(selection.item) <= 1,
            )
        ):
            return

        curr_widget = cast(BaseTab, self.tab_widget.currentWidget())
        curr_widget.on_data_explorer_item_selected(selection)

    def set_active_workflow(self, workflow: Workflow):
        # 1. Switch Tab
        if workflow == Workflow.POSE_ESTIMATION:
            self.tab_widget.setCurrentWidget(self.poses_view)
        elif workflow == Workflow.BEHAVIOR_CLASSIFICATION:
            self.tab_widget.setCurrentWidget(self.behaviors_view)
        
        # 2. Refresh Context (ensure the new tab sees the current selection)
        if self._current_selection:
            curr_widget = cast(BaseTab, self.tab_widget.currentWidget())
            curr_widget.on_data_explorer_item_selected(self._current_selection)