import dataclasses
import logging
from typing import Dict, Tuple, Type, TypeVar
from typing import Callable
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from whisker.core.workspace import Workspace, Project
from whisker.gui.tabs.base_tab import BaseTab, Workflow
from whisker.gui.widgets import data_explorer
from whisker.gui.worker_wrapper import Worker

T = TypeVar('T', bound=BaseTab)

@dataclasses.dataclass
class _SubtabRecord:
    dtype: Type
    obj: BaseTab

WorkflowItemHandler = tuple[Callable[[Workspace, data_explorer.Selection], bool], QWidget]

class BaseWorkflowTab(BaseTab):
    """
    The main container tab for "Train", which holds sub-tabs for
    Pose and Behavior model training.
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Use 0 margins

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        self.tab_widget.tabBar().setVisible(False)

        self.subtab_records: Dict[str, _SubtabRecord] = {}

    def _current_subtab(self) -> Optional[BaseTab]:
        """Helper to get the currently active sub-tab."""
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, BaseTab):
            return current_widget
        return None

    def add_subtab(self, dtype: Type[T], name: str, info: str):
        new_record = _SubtabRecord(
            dtype=dtype,
            obj=dtype(self)
        )
        self.subtab_records[name] = new_record
        self.tab_widget.addTab(new_record.obj, info)
        new_record.obj.request_launch_worker.connect(self.request_launch_worker)
        
    def set_workspace(self, workspace: Optional[Workspace]):
        """Pass the workspace down to both sub-tabs."""
        super().set_workspace(workspace)
        for record in self.subtab_records.values():
            record.obj.set_workspace(workspace)

    def set_project(self, project: Optional[Project]):
        """Pass the project down to both sub-tabs."""
        super().set_project(project)
        for record in self.subtab_records.values():
            record.obj.set_project(project)

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        """Pass the selection event to the currently visible sub-tab."""
        if subtab := self._current_subtab():
            subtab.on_data_explorer_item_selected(selection)

    def set_active_workflow(self, workflow: Workflow):
        subtab = self.subtab_records.get(workflow.value)
        if subtab:
            self.tab_widget.setCurrentWidget(subtab.obj)
            subtab.obj.on_data_explorer_item_selected(None)
