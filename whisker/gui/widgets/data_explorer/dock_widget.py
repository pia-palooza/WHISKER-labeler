from typing import Optional

from PyQt6.QtCore import Qt, pyqtBoundSignal
from PyQt6.QtWidgets import (
    QWidget,
    QDockWidget,
)

from whisker.core.workspace import Workspace, Project
from .widget import Widget
from .constants import ItemGroupEnum


class DockWidget(QDockWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Data Explorer", parent)
        self.data_explorer = Widget(self)
        self.setWidget(self.data_explorer)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

    def update_workspace(self, workspace: Optional[Workspace]):
        self.data_explorer.update_workspace(workspace)

    def set_active_project(self, project: Optional[Project]):
        self.data_explorer.set_active_project(project)

    @property
    def item_selected(self) -> pyqtBoundSignal:
        return self.data_explorer.item_selected

    @property
    def item_deselected(self) -> pyqtBoundSignal:
        return self.data_explorer.item_deselected

    @property
    def blind_mode_toggled(self) -> pyqtBoundSignal:
        return self.data_explorer.blind_mode_toggled

    def set_item_group(self, group: ItemGroupEnum):
        self.data_explorer.set_item_group(group)
    
    def set_workflow(self, workflow):
        self.data_explorer.set_workflow(workflow)

    def set_run_name(self, run_name: Optional[str]):
        self.data_explorer.set_run_name(run_name)

    def is_blind_mode_enabled(self):
        return self.data_explorer.is_blind_mode_enabled()
    
    def show_item_groups(self, groups: list[ItemGroupEnum]):
        self.data_explorer.show_item_groups(groups)