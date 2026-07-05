import logging
from typing import Optional
import enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QSizePolicy, QWidget

from whisker.core.workspace import Workspace, Project
from whisker.core.workflows.workflow_enum import Workflow
from whisker.gui.widgets import data_explorer
from whisker.gui.worker_wrapper import Worker


class BaseTab(QWidget):
    # DEV_NOTE: This signal is emitted whenever the tab's dirty state changes.
    dirty_state_changed = pyqtSignal(bool)
    request_select_prev_image = pyqtSignal()
    request_select_next_image = pyqtSignal()
    request_launch_worker = pyqtSignal(str, Worker)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None
        self._project: Optional[Project] = None
        self._is_dirty = False

        # On creation the tab will be assumed to be hidden and its size policy
        # is to be adjusted based on whether it is visible. 
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        pass

    def get_data_explorer_item_groups(self) -> list[data_explorer.ItemGroupEnum]:
        return []

    def set_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace

    def set_project(self, project: Optional[Project]):
        """
        Sets the active project for this tab. Subclasses should override this
        to handle project-specific data loading.
        """
        self._project = project

    def is_dirty(self) -> bool:
        """Returns True if the tab has unsaved changes."""
        return self._is_dirty

    def set_dirty(self, dirty: bool):
        """
        Sets the dirty state of the tab.

        If the state changes, it emits the `dirty_state_changed` signal.
        """
        if self._is_dirty != dirty:
            logging.debug(f"{self.__class__} dirty state changing from {self._is_dirty} to {dirty}")
            self._is_dirty = dirty
            self.dirty_state_changed.emit(self._is_dirty)

    def set_active_workflow(self, workflow: Workflow):
        """
        Sets the active workflow for this tab. Subclasses should override this
        to handle workflow-specific data loading.
        """
        del workflow  # Unused variable
