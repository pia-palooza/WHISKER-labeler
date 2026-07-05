import logging
from pathlib import Path

from PyQt6.QtWidgets import QWidget

from whisker.core.workspace import Workspace
from whisker.gui.widgets import data_explorer

class BaseInfoItemHandlerWidget(QWidget):
    def __init__(self):
        super().__init__()

    def show_workflow_item_info(self, workspace: Workspace, selection: data_explorer.Selection) -> bool:
        if selection.group == data_explorer.ItemGroupEnum.WORKSPACE_FILES:
            relative_path = Path(*selection.item[1:])
            file_path = workspace.base_dir / relative_path
            return self.show_workflow_file_info(workspace, file_path)
        return False

    def show_workflow_file_info(self, workspace: Workspace, file_path: Path) -> bool:
        return False